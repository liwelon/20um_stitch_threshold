from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QDateTime, QEvent, QObject, QSettings, Qt, QTemporaryDir, QThread
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView, QAbstractScrollArea, QAbstractSpinBox, QApplication,
    QButtonGroup, QComboBox, QFileDialog, QFormLayout, QGraphicsView, QGridLayout,
    QGroupBox, QHeaderView, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
    QMainWindow, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSlider,
    QSpinBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem, QTabWidget,
    QVBoxLayout, QWidget,
)

from core.first_pad import (
    DEFAULT_THRESH,
    estimate_pad_rotation,
    find_pad_column,
)
from core.image_utils import (
    IMAGE_EXTENSIONS, imread_unicode, imwrite_unicode, natural_sort_key,
    numpy_to_qimage, rotate_image,
)
from core.traditional_detection import (
    AutoCropOutput,
    DEFAULT_BLIND_CIRCUIT_MODEL_PATH,
    DetectionOutput,
    _rect,
)
from core.wafer_recipe_db import (
    DEFAULT_DATABASE_CONFIG,
    WaferRecipe,
    load_exported_wafer_recipes,
)
from dialogs.recipe_manager import RecipeManagerDialog
from dialogs.stitch_calibration import StitchCalibrationDialog
from core.stitching.params import (
    get_default_machine, get_joint_params, get_machine_presets, list_machine_ids,
    params_json_path,
)
from widgets.image_view import ImageView
from workers.crop_worker import BatchRegionCropWorker, CropWorker
from workers.stitch_worker import StitchWorker
from workers.traditional_detection_worker import TraditionalDetectionWorker
from workers.wafer_recipe_worker import WaferRecipeWorker


@dataclass(frozen=True)
class _ParameterField:
    value: Any
    path: tuple[str, ...]


class _HoldWKeyFilter(QObject):
    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self._window = window

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() in {QEvent.Type.KeyPress, QEvent.Type.KeyRelease}:
            if event.key() == Qt.Key.Key_R:  # type: ignore[attr-defined]
                if not event.isAutoRepeat():  # type: ignore[attr-defined]
                    self._window.set_measure_mode(event.type() == QEvent.Type.KeyPress)
                event.accept()
                return True
        if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():  # type: ignore[attr-defined]
            key = event.key()  # type: ignore[attr-defined]
            if key == Qt.Key.Key_W:
                self._window.set_roi_draw_mode(True)
                return False
        if event.type() == QEvent.Type.KeyRelease and not event.isAutoRepeat():  # type: ignore[attr-defined]
            if event.key() == Qt.Key.Key_W:  # type: ignore[attr-defined]
                self._window.set_roi_draw_mode(False)
                return False
        return super().eventFilter(watched, event)


class _NoWheelFilter(QObject):
    """拦截控件滚轮改值，并将滚轮转给外层滚动区域。"""

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() != QEvent.Type.Wheel:
            return super().eventFilter(watched, event)
        parent = watched.parent()
        while parent is not None:
            if isinstance(parent, QAbstractScrollArea):
                QApplication.sendEvent(parent.viewport(), event)
                return True
            parent = parent.parent()
        return True


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._source_image: np.ndarray | None = None
        self._original_image: np.ndarray | None = None
        self._gray_image: np.ndarray | None = None
        self._down_image: np.ndarray | None = None
        self._up_image: np.ndarray | None = None
        self._stitched_images: list[np.ndarray] = []
        self._stitched_up_images: list[np.ndarray | None] = []
        self._stitch_layer = "down"
        self._ignored_image_count = 0
        self._stitch_all = False
        self._stitch_finished = True
        self._settings = QSettings("ImageBinarizationTool", "ImageBinarizationTool")
        saved_mode = self._settings.value("last_resolution_mode", "20um", type=str)
        self._resolution_mode = saved_mode if saved_mode in {"20um", "5um"} else "20um"
        self._display_mode = "original"
        self._syncing_views = False
        self._first_pad: tuple[int, int, int, int] | None = None
        self._pad_rects: list[tuple[int, int, int, int]] = []
        self._pad_band_mode: str | None = None  # "left" | "both"
        self._pad_params: dict[str, int] = {}
        self._pad_top_to_cmos_edge_pix: int | None = None
        self._roi_rect: tuple[int, int, int, int] | None = None
        self._syncing_measurement = False
        self._measure_mode = False
        self._crop_image: np.ndarray | None = None
        self._crop_gray: np.ndarray | None = None
        self._stitch_thread: QThread | None = None
        self._stitch_worker: StitchWorker | None = None
        self._recipe_thread: QThread | None = None
        self._recipe_worker: WaferRecipeWorker | None = None
        self._recipe_imported_during_refresh = False
        self._detection_thread: QThread | None = None
        self._detection_worker: TraditionalDetectionWorker | None = None
        self._crop_thread: QThread | None = None
        self._crop_worker: CropWorker | None = None
        self._region_crop_thread: QThread | None = None
        self._region_crop_worker: BatchRegionCropWorker | None = None
        self._region_crop_stop_requested = False
        self._region_crop_configured_zones = False
        self._auto_crop_output: AutoCropOutput | None = None
        self._last_auto_crop_angle = 0.0
        self._wafer_recipes: list[WaferRecipe] = []
        self._runtime_recipe_params: dict[str, Any] | None = None
        self._recipe_region_items: list[tuple[str, dict[str, Any] | list[Any], str]] = []
        self._parameter_editors: dict[tuple[str, ...], QLineEdit] = {}
        self._five_um_zone_limit_table: QTableWidget | None = None
        self._five_um_other_limit_table: QTableWidget | None = None
        self._detection_output: DetectionOutput | None = None
        self._detection_outputs: list[tuple[DetectionOutput, str]] = []
        self._detection_cache_dir: QTemporaryDir | None = None
        self._detection_cache_paths: list[tuple[Path, Path] | None] = []
        self._detection_output_index = -1
        self._single_detection_file_name: str | None = None
        self._single_detection_layer_name: str | None = None
        self._crop_folder_paths: list[Path] = []
        self._crop_folder_index = -1
        self._detection_all = False
        self._detection_stop_requested = False
        self._parameter_group_fields: list[
            tuple[
                QGridLayout,
                list[tuple[QLabel, QLineEdit, bool, QPushButton | None]],
            ]
        ] = []
        self._calibration_dialog: StitchCalibrationDialog | None = None
        self.setWindowTitle("图像拼接与二值化工具")
        self.resize(1500, 800)
        self.setAcceptDrops(True)
        self._build_ui()
        self._build_shortcuts()
        self._w_filter = _HoldWKeyFilter(self)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self._w_filter)
        self._restore_last_paths()
        self.setStyleSheet(
            """
            QMainWindow { background: #eef2f7; }
            QWidget#controlPanel {
                background: #ffffff;
                border: 1px solid #dbe3ee;
                border-radius: 12px;
            }
            QWidget#leftPanel {
                background: #ffffff;
                border: 1px solid #dbe3ee;
                border-radius: 12px;
            }
            QGroupBox#sectionGroup {
                background: #f8fafc;
                border: 1px solid #dbe3ee;
                border-radius: 9px;
                margin-top: 8px;
                padding: 10px 10px 8px 10px;
                color: #334155;
                font-weight: 600;
            }
            QGroupBox#sectionGroup::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background: #ffffff;
            }
            QLabel#title { color: #1f2937; font-size: 20px; font-weight: 600; }
            QLabel#subtitle { color: #718096; font-size: 13px; }
            QLabel#roiTitle { color: #1f2937; font-weight: 600; }
            QSlider::groove:horizontal { height: 5px; background: #dbe3ee; border-radius: 2px; }
            QSlider::handle:horizontal { width: 16px; margin: -6px 0; background: #3b82f6; border-radius: 8px; }
            QPushButton, QComboBox, QLineEdit {
                min-height: 34px; border-radius: 7px;
            }
            QComboBox, QLineEdit {
                padding: 0 8px; border: 1px solid #cbd5e1; background: #f8fafc;
            }
            QSpinBox, QDoubleSpinBox {
                min-height: 34px; background: #f8fafc; color: #1f2937;
            }
            QLineEdit#parameterEditor {
                min-height: 28px; max-height: 28px; padding: 0 6px;
            }
            QTableWidget#limitTable {
                border: 1px solid #dbe3ee; background: #ffffff;
                alternate-background-color: #f8fafc; gridline-color: #e2e8f0;
            }
            QTableWidget#limitTable QHeaderView::section {
                min-height: 28px; background: #f1f5f9; border: none;
                border-right: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0;
                padding: 0 6px; font-weight: 600;
            }
            QPlainTextEdit#operationLog {
                border: 1px solid #cbd5e1; border-radius: 7px;
                background: #f8fafc; color: #334155; padding: 6px;
            }
            QPushButton#spinArrowButton {
                min-width: 28px; max-width: 28px; min-height: 16px; max-height: 16px;
                border-radius: 4px; padding: 0; font-size: 9px; font-weight: 600;
            }
            QPushButton { border: 1px solid #3b82f6; color: #2563eb; background: #eff6ff; padding: 0 12px; }
            QPushButton:hover { background: #dbeafe; }
            QPushButton:checked { color: #ffffff; background: #2563eb; border-color: #1d4ed8; }
            QPushButton:disabled { color: #94a3b8; border-color: #cbd5e1; background: #f1f5f9; }
            QPushButton#modeLabel {
                min-height: 28px; border: none; background: transparent;
                color: #94a3b8; font-weight: 600; padding: 0 4px;
            }
            QPushButton#modeLabel:checked { color: #2563eb; background: transparent; }
            QSlider#modeSwitch { min-height: 28px; max-height: 28px; }
            QSlider#modeSwitch::groove:horizontal {
                height: 16px; background: #dbeafe; border: 1px solid #93c5fd;
                border-radius: 8px;
            }
            QSlider#modeSwitch::handle:horizontal {
                width: 24px; height: 24px; margin: -5px 0;
                background: #2563eb; border: 2px solid #ffffff; border-radius: 12px;
            }
            """
        )

    def _build_ui(self) -> None:
        central_widget = QWidget(self)
        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(12)

        image_panel = QWidget()
        image_panel_layout = QVBoxLayout(image_panel)
        image_panel_layout.setContentsMargins(0, 0, 0, 0)
        image_panel_layout.setSpacing(8)

        display_row = QHBoxLayout()
        display_row.setContentsMargins(0, 0, 0, 0)
        display_row.setSpacing(8)
        display_label = QLabel("显示模式")
        self.mode_original_btn = QPushButton("原图")
        self.mode_binary_btn = QPushButton("二值图")
        self.mode_parallel_btn = QPushButton("并排")
        self.mode_crop_btn = QPushButton("裁剪图")
        self.mode_detection_btn = QPushButton("检测结果")
        self.mode_detection_btn.setEnabled(False)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        for button, mode in (
            (self.mode_original_btn, "original"),
            (self.mode_binary_btn, "binary"),
            (self.mode_parallel_btn, "parallel"),
            (self.mode_crop_btn, "crop"),
            (self.mode_detection_btn, "detection"),
        ):
            button.setCheckable(True)
            self.mode_group.addButton(button)
            display_row.addWidget(button)
            button.clicked.connect(lambda _checked=False, m=mode: self.set_display_mode(m))
        self.mode_original_btn.setChecked(True)
        display_row.insertWidget(0, display_label)
        self.previous_detection_button = QPushButton("上一张")
        self.previous_detection_button.setEnabled(False)
        self.previous_detection_button.clicked.connect(
            lambda _checked=False: self.switch_detection_output(-1)
        )
        self.detection_page_label = QLabel("0/0")
        self.next_detection_button = QPushButton("下一张")
        self.next_detection_button.setEnabled(False)
        self.next_detection_button.clicked.connect(
            lambda _checked=False: self.switch_detection_output(1)
        )
        display_row.addWidget(self.previous_detection_button)
        display_row.addWidget(self.detection_page_label)
        display_row.addWidget(self.next_detection_button)
        display_row.addStretch()
        image_panel_layout.addLayout(display_row)

        image_layout = QHBoxLayout()
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(8)

        self.original_view = ImageView()
        self.original_view.set_roi_interactive(True)
        self.original_view.roi_changed.connect(self._on_roi_changed)
        self.original_view.recipe_region_changed.connect(
            self._on_recipe_region_dragged
        )
        self.original_view.measurement_changed.connect(
            lambda x, y, width, height: self._on_measurement_changed(
                self.original_view, x, y, width, height
            )
        )
        self.original_view.transform_changed.connect(
            lambda: self._sync_views_from(self.original_view)
        )
        image_layout.addWidget(self.original_view, 1)

        self.image_view = ImageView()
        self.image_view.set_roi_interactive(False)
        self.image_view.recipe_region_changed.connect(
            self._on_recipe_region_dragged
        )
        self.image_view.measurement_changed.connect(
            lambda x, y, width, height: self._on_measurement_changed(
                self.image_view, x, y, width, height
            )
        )
        self.image_view.setVisible(False)
        self.image_view.transform_changed.connect(
            lambda: self._sync_views_from(self.image_view)
        )
        image_layout.addWidget(self.image_view, 1)
        image_panel_layout.addLayout(image_layout, 1)
        control_panel = QWidget()
        control_panel.setObjectName("controlPanel")
        control_panel.setMinimumWidth(0)
        control_panel.setMaximumWidth(360)
        panel_layout = QVBoxLayout(control_panel)
        panel_layout.setContentsMargins(8, 8, 8, 8)
        panel_layout.setSpacing(6)

        left_panel = QWidget()
        left_panel.setObjectName("leftPanel")
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(12)

        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(4, 0, 4, 0)
        self.mode_20um_button = QPushButton("20um")
        self.mode_20um_button.setObjectName("modeLabel")
        self.mode_20um_button.setCheckable(True)
        self.mode_20um_button.clicked.connect(
            lambda _checked=False: self.set_resolution_mode("20um")
        )
        self.mode_switch = QSlider(Qt.Orientation.Horizontal)
        self.mode_switch.setObjectName("modeSwitch")
        self.mode_switch.setRange(0, 1)
        self.mode_switch.setFixedWidth(58)
        self.mode_switch.setPageStep(1)
        self.mode_switch.valueChanged.connect(
            lambda value: self.set_resolution_mode("5um" if value else "20um")
        )
        self.mode_5um_button = QPushButton("5um")
        self.mode_5um_button.setObjectName("modeLabel")
        self.mode_5um_button.setCheckable(True)
        self.mode_5um_button.clicked.connect(
            lambda _checked=False: self.set_resolution_mode("5um")
        )
        mode_row.addStretch()
        mode_row.addWidget(self.mode_20um_button)
        mode_row.addWidget(self.mode_switch)
        mode_row.addWidget(self.mode_5um_button)
        mode_row.addStretch()
        self._sync_mode_switch()
        left_layout.addLayout(mode_row)

        machine_group = QGroupBox("机台选择")
        machine_group.setObjectName("sectionGroup")
        machine_layout = QVBoxLayout(machine_group)
        machine_layout.setContentsMargins(4, 8, 4, 2)
        machine_layout.setSpacing(8)

        stitch_group = QGroupBox("拼接小图")
        stitch_group.setObjectName("sectionGroup")
        stitch_layout = QVBoxLayout(stitch_group)
        stitch_layout.setContentsMargins(4, 8, 4, 2)
        stitch_layout.setSpacing(8)

        self.machine_combo = QComboBox()
        try:
            machine_ids = list_machine_ids(self._resolution_mode)
            default_machine = get_default_machine(self._resolution_mode)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "参数加载失败",
                f"无法读取拼接参数：\n{params_json_path()}\n\n{exc}",
            )
            machine_ids = []
            default_machine = ""
        self.machine_combo.addItems(machine_ids)
        if default_machine:
            self.machine_combo.setCurrentText(default_machine)
        machine_layout.addWidget(QLabel("机台配方"))
        machine_row = QHBoxLayout()
        machine_row.addWidget(self.machine_combo, 1)
        self.manage_recipe_button = QPushButton("管理配方")
        self.manage_recipe_button.clicked.connect(self.open_recipe_manager)
        machine_row.addWidget(self.manage_recipe_button)
        machine_layout.addLayout(machine_row)
        self.calibrate_stitch_button = QPushButton("确认拼接参数")
        self.calibrate_stitch_button.clicked.connect(self.open_stitch_calibration)
        machine_layout.addWidget(self.calibrate_stitch_button)
        left_layout.addWidget(machine_group)

        recipe_header = QGroupBox("wafer_cpp 配方")
        recipe_header.setObjectName("sectionGroup")
        recipe_header_layout = QVBoxLayout(recipe_header)
        recipe_header_layout.setContentsMargins(4, 6, 4, 4)
        recipe_header_layout.setSpacing(4)
        recipe_row = QHBoxLayout()
        recipe_row.setSpacing(4)
        self.wafer_recipe_combo = QComboBox()
        self.wafer_recipe_combo.currentIndexChanged.connect(
            self._show_current_wafer_recipe
        )
        self.load_wafer_recipes_button = QPushButton("刷新配方")
        self.load_wafer_recipes_button.clicked.connect(self.refresh_wafer_recipes)
        self.import_wafer_recipes_button = QPushButton("导入配方")
        self.import_wafer_recipes_button.clicked.connect(
            self.import_wafer_recipes
        )
        recipe_row.addWidget(QLabel("配方："))
        recipe_row.addWidget(self.wafer_recipe_combo, 1)
        recipe_header_layout.addLayout(recipe_row)
        recipe_header_layout.addWidget(self.load_wafer_recipes_button)
        recipe_header_layout.addWidget(self.import_wafer_recipes_button)
        left_layout.addWidget(recipe_header)

        image_source_group = QGroupBox("图像来源")
        image_source_group.setObjectName("sectionGroup")
        image_source_layout = QVBoxLayout(image_source_group)
        image_source_layout.setContentsMargins(4, 8, 4, 2)
        image_source_layout.setSpacing(6)

        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("小图文件夹路径")
        self.folder_edit.setReadOnly(True)
        image_source_layout.addWidget(self.folder_edit)
        browse_folder_button = QPushButton("选择文件夹")
        browse_folder_button.clicked.connect(self.select_folder)
        image_source_layout.addWidget(browse_folder_button)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("单张裁剪图路径（可选）")
        self.path_edit.setReadOnly(True)
        image_source_layout.addWidget(self.path_edit)
        browse_image_button = QPushButton("选择单张图")
        browse_image_button.clicked.connect(self.select_image)
        image_source_layout.addWidget(browse_image_button)

        self.cropped_up_edit = QLineEdit()
        self.cropped_up_edit.setPlaceholderText("20um 已裁剪上图路径")
        self.cropped_up_edit.setReadOnly(True)
        image_source_layout.addWidget(self.cropped_up_edit)
        self.cropped_up_edit.setVisible(self._resolution_mode == "20um")
        self.browse_cropped_up_button = QPushButton("选择已裁剪上图")
        self.browse_cropped_up_button.clicked.connect(
            lambda _checked=False: self.select_cropped_pair_image("up")
        )
        self.clear_cropped_up_button = QPushButton("清除上图")
        self.clear_cropped_up_button.clicked.connect(
            lambda _checked=False: self.clear_cropped_pair_image("up")
        )
        cropped_up_row = QHBoxLayout()
        cropped_up_row.addWidget(self.browse_cropped_up_button)
        cropped_up_row.addWidget(self.clear_cropped_up_button)
        image_source_layout.addLayout(cropped_up_row)
        self.browse_cropped_up_button.setVisible(self._resolution_mode == "20um")
        self.clear_cropped_up_button.setVisible(self._resolution_mode == "20um")

        self.cropped_down_edit = QLineEdit()
        self.cropped_down_edit.setPlaceholderText("20um 已裁剪下图路径")
        self.cropped_down_edit.setReadOnly(True)
        image_source_layout.addWidget(self.cropped_down_edit)
        self.cropped_down_edit.setVisible(self._resolution_mode == "20um")
        self.browse_cropped_down_button = QPushButton("选择已裁剪下图")
        self.browse_cropped_down_button.clicked.connect(
            lambda _checked=False: self.select_cropped_pair_image("down")
        )
        self.clear_cropped_down_button = QPushButton("清除下图")
        self.clear_cropped_down_button.clicked.connect(
            lambda _checked=False: self.clear_cropped_pair_image("down")
        )
        cropped_down_row = QHBoxLayout()
        cropped_down_row.addWidget(self.browse_cropped_down_button)
        cropped_down_row.addWidget(self.clear_cropped_down_button)
        image_source_layout.addLayout(cropped_down_row)
        self.browse_cropped_down_button.setVisible(self._resolution_mode == "20um")
        self.clear_cropped_down_button.setVisible(self._resolution_mode == "20um")

        self.crop_folder_edit = QLineEdit()
        self.crop_folder_edit.setPlaceholderText("5um 裁剪图文件夹路径")
        self.crop_folder_edit.setReadOnly(True)
        self.crop_folder_edit.setVisible(self._resolution_mode == "5um")
        image_source_layout.addWidget(self.crop_folder_edit)
        self.browse_crop_folder_button = QPushButton("选择裁剪图文件夹")
        self.browse_crop_folder_button.setVisible(self._resolution_mode == "5um")
        self.browse_crop_folder_button.clicked.connect(self.select_crop_folder)
        image_source_layout.addWidget(self.browse_crop_folder_button)
        left_layout.addWidget(image_source_group)

        log_group = QGroupBox("操作日志")
        log_group.setObjectName("sectionGroup")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(4, 8, 4, 2)
        self.operation_log = QPlainTextEdit()
        self.operation_log.setObjectName("operationLog")
        self.operation_log.setReadOnly(True)
        self.operation_log.setPlaceholderText("操作记录将在此显示")
        log_layout.addWidget(self.operation_log)
        left_layout.addWidget(log_group, 1)

        stitch_action_row = QHBoxLayout()
        self.stitch_button = QPushButton("拼接单张")
        self.stitch_button.clicked.connect(self.start_stitch)
        self.stitch_all_button = QPushButton("全部拼接")
        self.stitch_all_button.clicked.connect(
            lambda _checked=False: self.start_stitch(stitch_all=True)
        )
        stitch_action_row.addWidget(self.stitch_button)
        stitch_action_row.addWidget(self.stitch_all_button)
        stitch_layout.addLayout(stitch_action_row)

        self.save_stitch_button = QPushButton("保存当前拼接图")
        self.save_stitch_button.setEnabled(False)
        self.save_stitch_button.clicked.connect(self.save_stitched_image)
        stitch_layout.addWidget(self.save_stitch_button)

        preview_row = QHBoxLayout()
        self.previous_stitch_button = QPushButton("上一张")
        self.previous_stitch_button.setEnabled(False)
        self.previous_stitch_button.clicked.connect(
            lambda _checked=False: self.switch_stitched_image(-1)
        )
        self.stitch_preview_combo = QComboBox()
        self.stitch_preview_combo.setEnabled(False)
        self.stitch_preview_combo.currentIndexChanged.connect(
            self.preview_stitched_image
        )
        self.next_stitch_button = QPushButton("下一张")
        self.next_stitch_button.setEnabled(False)
        self.next_stitch_button.clicked.connect(
            lambda _checked=False: self.switch_stitched_image(1)
        )
        preview_row.addWidget(self.previous_stitch_button)
        preview_row.addWidget(self.stitch_preview_combo, 1)
        preview_row.addWidget(self.next_stitch_button)
        stitch_layout.addLayout(preview_row)

        self.stitch_layer_widget = QWidget()
        stitch_layer_layout = QHBoxLayout(self.stitch_layer_widget)
        stitch_layer_layout.setContentsMargins(0, 0, 0, 0)
        stitch_layer_layout.setSpacing(8)
        self.stitch_down_button = QPushButton("下图")
        self.stitch_down_button.setCheckable(True)
        self.stitch_down_button.setChecked(True)
        self.stitch_down_button.setEnabled(False)
        self.stitch_down_button.clicked.connect(
            lambda _checked=False: self.show_stitched_layer("down")
        )
        self.stitch_up_button = QPushButton("上图")
        self.stitch_up_button.setCheckable(True)
        self.stitch_up_button.setEnabled(False)
        self.stitch_up_button.clicked.connect(
            lambda _checked=False: self.show_stitched_layer("up")
        )
        self.stitch_layer_group = QButtonGroup(self)
        self.stitch_layer_group.setExclusive(True)
        self.stitch_layer_group.addButton(self.stitch_down_button)
        self.stitch_layer_group.addButton(self.stitch_up_button)
        stitch_layer_layout.addWidget(self.stitch_down_button)
        stitch_layer_layout.addWidget(self.stitch_up_button)
        stitch_layout.addWidget(self.stitch_layer_widget)
        self.stitch_layer_widget.setVisible(self._resolution_mode == "20um")
        panel_layout.addWidget(stitch_group)
        self.machine_combo.currentTextChanged.connect(self._clear_stitch_results)
        self.machine_combo.currentTextChanged.connect(self._remember_machine)

        threshold_group = QGroupBox("阈值调整")
        threshold_group.setObjectName("sectionGroup")
        threshold_group_layout = QVBoxLayout(threshold_group)
        threshold_group_layout.setContentsMargins(4, 8, 4, 2)
        threshold_group_layout.setSpacing(4)

        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(DEFAULT_THRESH)

        self.threshold_spinbox = QSpinBox()
        self.threshold_spinbox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.threshold_spinbox.setRange(0, 255)
        self.threshold_spinbox.setValue(DEFAULT_THRESH)
        self.threshold_spinbox.setFixedWidth(88)
        self.threshold_spinbox.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.threshold_slider.valueChanged.connect(self.threshold_spinbox.setValue)
        self.threshold_spinbox.valueChanged.connect(self.threshold_slider.setValue)
        self.threshold_slider.valueChanged.connect(self.update_binary_image)
        self._no_wheel_filter = _NoWheelFilter(self)
        self.threshold_slider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.threshold_spinbox.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.threshold_slider.installEventFilter(self._no_wheel_filter)
        self.threshold_spinbox.installEventFilter(self._no_wheel_filter)
        self.machine_combo.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.machine_combo.installEventFilter(self._no_wheel_filter)

        threshold_layout = QFormLayout()
        threshold_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        threshold_value_widget = QWidget()
        threshold_value_layout = QHBoxLayout(threshold_value_widget)
        threshold_value_layout.setContentsMargins(0, 0, 0, 0)
        threshold_value_layout.setSpacing(4)
        threshold_arrows = QVBoxLayout()
        threshold_arrows.setContentsMargins(0, 0, 0, 0)
        threshold_arrows.setSpacing(2)
        threshold_up_button = QPushButton("▲")
        threshold_up_button.setObjectName("spinArrowButton")
        threshold_up_button.clicked.connect(self.threshold_spinbox.stepUp)
        threshold_down_button = QPushButton("▼")
        threshold_down_button.setObjectName("spinArrowButton")
        threshold_down_button.clicked.connect(self.threshold_spinbox.stepDown)
        threshold_arrows.addWidget(threshold_up_button)
        threshold_arrows.addWidget(threshold_down_button)
        threshold_value_layout.addWidget(self.threshold_spinbox)
        threshold_value_layout.addLayout(threshold_arrows)
        self.write_thresh_button = QPushButton("写入当前配方")
        self.write_thresh_button.setToolTip(
            "将当前阈值应用到当前 wafer_cpp 运行配方（仅本次运行）"
        )
        self.write_thresh_button.clicked.connect(self.apply_threshold_to_current_recipe)
        threshold_value_layout.addWidget(self.write_thresh_button, 1)
        threshold_layout.addRow("阈值：", threshold_value_widget)
        threshold_layout.addRow(self.threshold_slider)
        thresh_hint = QLabel("亦为 padGrayThresh；切换 wafer_cpp 配方会自动载入")
        thresh_hint.setStyleSheet("color: #94a3b8; font-weight: 400; font-size: 11px;")
        threshold_layout.addRow(thresh_hint)
        self.gray_mean_label = QLabel("灰度均值：暂无图像")
        self.gray_mean_label.setStyleSheet("color: #475569; font-weight: 600;")
        threshold_layout.addRow(self.gray_mean_label)
        calculate_gray_mean_button = QPushButton("计算灰度均值")
        calculate_gray_mean_button.clicked.connect(self._update_gray_mean_display)
        threshold_layout.addRow(calculate_gray_mean_button)
        threshold_group_layout.addLayout(threshold_layout)
        panel_layout.addWidget(threshold_group)

        pad_group = QGroupBox("定位 Pad")
        pad_group.setObjectName("sectionGroup")
        pad_layout = QVBoxLayout(pad_group)
        pad_layout.setContentsMargins(4, 8, 4, 2)
        pad_layout.setSpacing(8)
        pad_btn_row = QHBoxLayout()
        pad_btn_row.setSpacing(8)
        self.find_pad_button = QPushButton("定位第一个Pad")
        self.find_pad_button.clicked.connect(self.locate_first_pad)
        pad_btn_row.addWidget(self.find_pad_button)
        self.find_all_pads_button = QPushButton("定位所有 Pad")
        self.find_all_pads_button.setToolTip("检测左右条带全部 pad 并绘制")
        self.find_all_pads_button.clicked.connect(self.locate_all_pads)
        pad_btn_row.addWidget(self.find_all_pads_button)
        pad_layout.addLayout(pad_btn_row)
        panel_layout.addWidget(pad_group)

        roi_group = QGroupBox("ROI 裁剪")
        roi_group.setObjectName("sectionGroup")
        roi_layout = QVBoxLayout(roi_group)
        roi_layout.setContentsMargins(4, 8, 4, 2)
        roi_layout.setSpacing(6)
        roi_title = QLabel("ROI 裁剪（仅原图可画框；并排时左右视图联动）")
        roi_title.setObjectName("roiTitle")
        roi_layout.addWidget(roi_title)

        self.rotation_spin = QDoubleSpinBox()
        self.rotation_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.rotation_spin.setDecimals(2)
        self.rotation_spin.setRange(-180.0, 180.0)
        self.rotation_spin.setSingleStep(0.05)
        self.rotation_spin.setValue(0.0)
        self.rotation_spin.setSuffix(" °")
        self.rotation_spin.setMinimumWidth(160)
        self.rotation_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rotation_spin.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.rotation_spin.valueChanged.connect(self._on_rotation_changed)
        self.rotation_spin.installEventFilter(self._no_wheel_filter)
        rotate_form = QFormLayout()
        rotate_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        rotation_value_widget = QWidget()
        rotation_value_layout = QHBoxLayout(rotation_value_widget)
        rotation_value_layout.setContentsMargins(0, 0, 0, 0)
        rotation_value_layout.setSpacing(4)
        rotation_arrows = QVBoxLayout()
        rotation_arrows.setContentsMargins(0, 0, 0, 0)
        rotation_arrows.setSpacing(2)
        rotation_up_button = QPushButton("▲")
        rotation_up_button.setObjectName("spinArrowButton")
        rotation_up_button.clicked.connect(self.rotation_spin.stepUp)
        rotation_down_button = QPushButton("▼")
        rotation_down_button.setObjectName("spinArrowButton")
        rotation_down_button.clicked.connect(self.rotation_spin.stepDown)
        rotation_arrows.addWidget(rotation_up_button)
        rotation_arrows.addWidget(rotation_down_button)
        rotation_value_layout.addWidget(self.rotation_spin, 1)
        rotation_value_layout.addLayout(rotation_arrows)
        rotate_form.addRow("旋转：", rotation_value_widget)
        roi_layout.addLayout(rotate_form)

        crop_action_row = QHBoxLayout()
        crop_action_row.setSpacing(8)
        self.auto_crop_button = QPushButton("自动裁剪")
        self.auto_crop_button.setToolTip(
            "按当前 wafer_cpp 配方执行模板匹配和 Pad 旋正，在原图上生成裁剪框"
        )
        self.auto_crop_button.setEnabled(False)
        self.auto_crop_button.clicked.connect(self.start_auto_crop)
        crop_action_row.addWidget(self.auto_crop_button)

        crop_button = QPushButton("裁剪保存")
        crop_button.clicked.connect(self.crop_and_save)
        crop_action_row.addWidget(crop_button)
        roi_layout.addLayout(crop_action_row)
        batch_crop_row = QHBoxLayout()
        batch_crop_row.setSpacing(8)
        self.batch_region_crop_button = QPushButton("批量裁剪盲元电路外围区")
        self.batch_region_crop_button.setEnabled(False)
        self.batch_region_crop_button.setVisible(self._resolution_mode == "5um")
        self.batch_region_crop_button.clicked.connect(self._handle_region_crop_clicked)
        batch_crop_row.addWidget(self.batch_region_crop_button)
        self.batch_configured_region_crop_button = QPushButton("按配置框批量裁剪")
        self.batch_configured_region_crop_button.setEnabled(False)
        self.batch_configured_region_crop_button.setVisible(self._resolution_mode == "5um")
        self.batch_configured_region_crop_button.clicked.connect(
            self._handle_configured_region_crop_clicked
        )
        batch_crop_row.addWidget(self.batch_configured_region_crop_button)
        roi_layout.addLayout(batch_crop_row)
        panel_layout.addWidget(roi_group)

        roi_info_group = QGroupBox("ROI 信息")
        roi_info_group.setObjectName("sectionGroup")
        roi_info_layout = QVBoxLayout(roi_info_group)
        roi_info_layout.setContentsMargins(4, 8, 4, 2)
        roi_info_layout.setSpacing(6)
        self.roi_size_label = QLabel("裁剪框宽高：尚未选择裁剪区域")
        self.measurement_label = QLabel("测量框宽高：尚未选择测量区域")
        self.roi_position_label = QLabel("左上角坐标：尚未选择裁剪区域")
        self.horizontal_offset_label = QLabel("水平偏移：尚未选择裁剪区域")
        self.vertical_offset_label = QLabel("垂直偏移：尚未选择裁剪区域")
        self.roi_size_label.setStyleSheet("color: #475569; font-weight: 600;")
        self.measurement_label.setStyleSheet("color: #c2410c; font-weight: 600;")
        self.roi_position_label.setStyleSheet("color: #475569; font-weight: 600;")
        self.horizontal_offset_label.setStyleSheet("color: #475569; font-weight: 600;")
        self.vertical_offset_label.setStyleSheet("color: #475569; font-weight: 600;")
        roi_info_layout.addWidget(self.roi_size_label)
        roi_info_layout.addWidget(self.measurement_label)
        roi_info_layout.addWidget(self.roi_position_label)
        roi_info_layout.addWidget(self.horizontal_offset_label)
        roi_info_layout.addWidget(self.vertical_offset_label)
        panel_layout.addWidget(roi_info_group)

        panel_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
        )
        scroll.setWidget(control_panel)

        parameter_page = QWidget()
        parameter_layout = QVBoxLayout(parameter_page)
        parameter_layout.setContentsMargins(6, 6, 6, 6)
        parameter_layout.setSpacing(6)

        self.parameter_content = QWidget()
        self.parameter_content_layout = QVBoxLayout(self.parameter_content)
        self.parameter_content_layout.setContentsMargins(0, 0, 0, 0)
        self.parameter_content_layout.setSpacing(4)
        self.parameter_content_layout.addStretch()
        self.recipe_region_group = QGroupBox("区域框微调")
        self.recipe_region_group.setVisible(self._resolution_mode == "5um")
        recipe_region_layout = QGridLayout(self.recipe_region_group)
        recipe_region_layout.setContentsMargins(4, 8, 4, 4)
        self.recipe_region_combo = QComboBox()
        self.recipe_region_combo.currentIndexChanged.connect(
            self._on_selected_recipe_region_changed
        )
        self.original_view.recipe_region_selected.connect(
            self.recipe_region_combo.setCurrentIndex
        )
        self.image_view.recipe_region_selected.connect(
            self.recipe_region_combo.setCurrentIndex
        )
        recipe_region_layout.addWidget(self.recipe_region_combo, 0, 0, 1, 4)
        self.recipe_region_spins: list[QSpinBox] = []
        for index, name in enumerate(("x1", "y1", "x2", "y2")):
            spin = QSpinBox()
            spin.setRange(0, 1000000)
            spin.valueChanged.connect(self._on_recipe_region_coordinate_changed)
            recipe_region_layout.addWidget(QLabel(name), 1 + index // 2, (index % 2) * 2)
            recipe_region_layout.addWidget(spin, 1 + index // 2, (index % 2) * 2 + 1)
            self.recipe_region_spins.append(spin)
        self.recipe_region_group.setEnabled(False)
        parameter_layout.addWidget(self.recipe_region_group)
        self.parameter_scroll = QScrollArea()
        self.parameter_scroll.setWidgetResizable(True)
        self.parameter_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.parameter_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.parameter_scroll.setWidget(self.parameter_content)
        self.parameter_scroll.viewport().installEventFilter(self)
        parameter_layout.addWidget(self.parameter_scroll, 1)

        detection_action_row = QHBoxLayout()
        self.run_detection_button = QPushButton("检测单张")
        self.run_detection_button.setEnabled(False)
        self.run_detection_button.clicked.connect(
            lambda _checked=False: self.run_traditional_detection()
        )
        self.run_detection_all_button = QPushButton(
            "批量检测" if self._resolution_mode == "5um" else "全部检测"
        )
        self.run_detection_all_button.setEnabled(False)
        self.run_detection_all_button.clicked.connect(
            self._handle_detect_all_clicked
        )
        detection_action_row.addWidget(self.run_detection_button)
        detection_action_row.addWidget(self.run_detection_all_button)
        parameter_layout.addLayout(detection_action_row)
        self.detection_result_edit = QPlainTextEdit()
        self.detection_result_edit.setReadOnly(True)
        self.detection_result_edit.setPlaceholderText("检测结果将在此显示")
        self.detection_result_edit.setMaximumHeight(110)
        parameter_layout.addWidget(self.detection_result_edit)

        self.control_tabs = QTabWidget()
        self.control_tabs.setFixedWidth(360)
        self.control_tabs.addTab(scroll, "操作")
        self.control_tabs.addTab(parameter_page, "参数预览")
        self.control_tabs.currentChanged.connect(self._update_recipe_overlay)
        root_layout.addWidget(left_panel)
        root_layout.addWidget(image_panel, 1)
        root_layout.addWidget(self.control_tabs)
        self.setCentralWidget(central_widget)
        self._append_log("界面已就绪")
        self.refresh_wafer_recipes()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if (
            watched is self.parameter_scroll.viewport()
            and event.type() == QEvent.Type.Resize
        ):
            self._reflow_parameter_groups(event.size().width())  # type: ignore[attr-defined]
        return super().eventFilter(watched, event)

    def _append_log(self, message: str) -> None:
        timestamp = QDateTime.currentDateTime().toString("HH:mm:ss")
        self.operation_log.appendPlainText(f"[{timestamp}] {message}")

    def _set_status(self, message: str) -> None:
        self._append_log(message)

    def refresh_wafer_recipes(self) -> None:
        if self._recipe_thread is not None and self._recipe_thread.isRunning():
            return
        self._recipe_imported_during_refresh = False
        self.load_wafer_recipes_button.setEnabled(False)
        self.mode_switch.setEnabled(False)
        self.mode_20um_button.setEnabled(False)
        self.mode_5um_button.setEnabled(False)
        self._set_status(
            f"正在连接 MySQL 并读取 {self._resolution_mode} 配方"
        )
        self._recipe_thread = QThread(self)
        self._recipe_worker = WaferRecipeWorker(
            DEFAULT_DATABASE_CONFIG,
            self._resolution_mode,
        )
        self._recipe_worker.moveToThread(self._recipe_thread)
        self._recipe_thread.started.connect(self._recipe_worker.run)
        self._recipe_worker.finished.connect(self._on_wafer_recipes_loaded)
        self._recipe_worker.failed.connect(self._on_wafer_recipes_failed)
        self._recipe_worker.finished.connect(self._recipe_thread.quit)
        self._recipe_worker.failed.connect(self._recipe_thread.quit)
        self._recipe_thread.finished.connect(self._cleanup_recipe_thread)
        self._recipe_thread.start()

    def _cleanup_recipe_thread(self) -> None:
        if self._recipe_worker is not None:
            self._recipe_worker.deleteLater()
            self._recipe_worker = None
        if self._recipe_thread is not None:
            self._recipe_thread.deleteLater()
            self._recipe_thread = None
        self.load_wafer_recipes_button.setEnabled(True)
        self.import_wafer_recipes_button.setEnabled(True)
        self.mode_switch.setEnabled(True)
        self.mode_20um_button.setEnabled(True)
        self.mode_5um_button.setEnabled(True)

    def _on_wafer_recipes_loaded(self, recipes: object) -> None:
        if self._recipe_imported_during_refresh:
            self._set_status("已保留文件导入的配方，忽略本次 MySQL 读取结果")
            return
        self._wafer_recipes = list(recipes)  # type: ignore[arg-type]
        self.wafer_recipe_combo.blockSignals(True)
        self.wafer_recipe_combo.clear()
        for recipe in self._wafer_recipes:
            self.wafer_recipe_combo.addItem(recipe.recipe_name, recipe.recipe_id)
        self.wafer_recipe_combo.blockSignals(False)
        if self._wafer_recipes:
            self.wafer_recipe_combo.setCurrentIndex(0)
            self._show_current_wafer_recipe(0)
        else:
            self._clear_parameter_groups()
            self.detection_result_edit.setPlainText(
                f"数据库中没有匹配 {self._resolution_mode} 结构的配方。"
            )
        self._update_detection_button()
        self._set_status(
            f"MySQL 连接成功，已加载 {len(self._wafer_recipes)} 个"
            f" {self._resolution_mode} 配方"
        )

    def _on_wafer_recipes_failed(self, message: str) -> None:
        self._set_status(f"MySQL 配方读取失败：{message}；可使用导入配方")

    def import_wafer_recipes(self) -> None:
        last_path = self._settings.value("last_wafer_recipe_path", "", type=str)
        initial_path = (
            last_path
            if last_path and Path(last_path).exists()
            else str(Path(last_path).parent) if last_path else ""
        )
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "导入 wafer_cpp 配方",
            initial_path,
            "JSON 文件 (*.json);;所有文件 (*)",
        )
        if not file_path:
            return
        try:
            recipes = load_exported_wafer_recipes(
                file_path,
                self._resolution_mode,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._set_status(f"配方导入失败：{exc}")
            QMessageBox.critical(self, "配方导入失败", str(exc))
            return

        self._recipe_imported_during_refresh = True
        self._wafer_recipes = recipes
        self.wafer_recipe_combo.blockSignals(True)
        self.wafer_recipe_combo.clear()
        for recipe in recipes:
            self.wafer_recipe_combo.addItem(recipe.recipe_name, recipe.recipe_id)
        self.wafer_recipe_combo.blockSignals(False)
        self.wafer_recipe_combo.setCurrentIndex(0)
        self._show_current_wafer_recipe(0)
        self._settings.setValue("last_wafer_recipe_path", file_path)
        self._set_status(
            f"已从文件导入 {len(recipes)} 个 {self._resolution_mode} 配方"
        )

    def _show_current_wafer_recipe(self, index: int) -> None:
        self._clear_parameter_groups()
        if not 0 <= index < len(self._wafer_recipes):
            self._update_detection_button()
            return
        recipe = self._wafer_recipes[index]
        self._auto_crop_output = None
        self._runtime_recipe_params = deepcopy(recipe.params)
        if self._resolution_mode == "5um":
            detection = self._runtime_recipe_params.get("DetectionParams")
            if isinstance(detection, dict):
                detection.setdefault(
                    "blindCircuitModelPath", DEFAULT_BLIND_CIRCUIT_MODEL_PATH
                )
                detection.setdefault("blindCircuitThreshold", 0.35)
                detection.setdefault("blindCircuitScoreThreshold", 0.0)
                detection.setdefault("coreMinArea", 300)
        self._apply_wafer_pad_params()
        for title, values in self._parameter_groups(self._runtime_recipe_params):
            if values:
                self._add_parameter_group(title, values)
            if self._resolution_mode == "5um" and title == "裁剪参数":
                detection = self._runtime_recipe_params.get("DetectionParams", {})
                if isinstance(detection, dict):
                    self._add_5um_limit_tables(detection)
        self.parameter_content_layout.addStretch()
        self._load_recipe_region_controls()
        self._update_auto_crop_button()
        self._update_detection_button()

    def _clear_parameter_groups(self) -> None:
        self._runtime_recipe_params = None
        self._recipe_region_items.clear()
        self.recipe_region_combo.blockSignals(True)
        self.recipe_region_combo.clear()
        self.recipe_region_combo.blockSignals(False)
        self.recipe_region_group.setEnabled(False)
        self.original_view.clear_recipe_overlay()
        self.image_view.clear_recipe_overlay()
        self._pad_params.clear()
        self._pad_top_to_cmos_edge_pix = None
        self._set_pad_actions_enabled(False)
        self._parameter_editors.clear()
        self._parameter_group_fields.clear()
        self._five_um_zone_limit_table = None
        self._five_um_other_limit_table = None
        self.auto_crop_button.setEnabled(False)
        while self.parameter_content_layout.count():
            item = self.parameter_content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _load_recipe_region_controls(self) -> None:
        self._recipe_region_items.clear()
        params = self._runtime_recipe_params
        if self._resolution_mode == "5um" and isinstance(params, dict):
            detection = params.get("DetectionParams", {})
            if isinstance(detection, dict):
                coords = detection.get("regionCoords", [])
                names = ("A", "B", "C", "Dummy", "Detection")
                colors = ("#ef4444", "#22c55e", "#3b82f6", "#eab308", "#06b6d4")
                if isinstance(coords, list):
                    for index, value in enumerate(coords[:5]):
                        if isinstance(value, (dict, list)):
                            _, _, width, height = _rect(value)
                            if width > 0 and height > 0:
                                self._recipe_region_items.append(
                                    (names[index], value, colors[index])
                                )
                for label, list_key, old_key, color in (
                    ("盲元区", "blindDetectionZones", "blindDetectionZone", "#f97316"),
                    ("电路区", "circuitDetectionZones", "circuitDetectionZone", "#14b8a6"),
                    ("外围区", "outerDetectionZones", None, "#a855f7"),
                    ("排除框", "excludeZones", None, "#94a3b8"),
                ):
                    configured = detection.get(list_key)
                    if isinstance(configured, list):
                        values = configured
                    elif old_key is not None and isinstance(detection.get(old_key), dict):
                        values = [detection[old_key]]
                    else:
                        values = []
                    for index, value in enumerate(values, start=1):
                        if isinstance(value, (dict, list)):
                            _, _, width, height = _rect(value)
                            if width > 0 and height > 0:
                                self._recipe_region_items.append(
                                    (f"{label} {index}", value, color)
                                )
        self.recipe_region_combo.blockSignals(True)
        self.recipe_region_combo.clear()
        for name, _, _ in self._recipe_region_items:
            self.recipe_region_combo.addItem(name)
        self.recipe_region_combo.blockSignals(False)
        self.recipe_region_group.setEnabled(bool(self._recipe_region_items))
        self._on_selected_recipe_region_changed(0 if self._recipe_region_items else -1)

    def _on_selected_recipe_region_changed(self, index: int) -> None:
        if not 0 <= index < len(self._recipe_region_items):
            self._update_recipe_overlay()
            return
        _, value, _ = self._recipe_region_items[index]
        x, y, width, height = _rect(value)
        for spin, coordinate in zip(
            self.recipe_region_spins, (x, y, x + width, y + height)
        ):
            spin.blockSignals(True)
            spin.setValue(coordinate)
            spin.blockSignals(False)
        self._update_recipe_overlay()

    def _on_recipe_region_coordinate_changed(self, _value: int) -> None:
        index = self.recipe_region_combo.currentIndex()
        if not 0 <= index < len(self._recipe_region_items):
            return
        x1, y1, x2, y2 = (spin.value() for spin in self.recipe_region_spins)
        if x1 >= x2 or y1 >= y2:
            self._on_selected_recipe_region_changed(index)
            return
        _, value, _ = self._recipe_region_items[index]
        if isinstance(value, list):
            value[:4] = [x1, y1, x2, y2]
        elif "x2" in value or "y2" in value:
            value.update(x1=x1, y1=y1, x2=x2, y2=y2)
        else:
            value["x1" if "x1" in value else "x"] = x1
            value["y1" if "y1" in value else "y"] = y1
            value.update(w=x2 - x1, h=y2 - y1)
        self._auto_crop_output = None
        self._update_recipe_overlay()

    def _on_recipe_region_dragged(
        self, index: int, x: int, y: int, width: int, height: int
    ) -> None:
        if not 0 <= index < len(self._recipe_region_items):
            return
        if self.recipe_region_combo.currentIndex() != index:
            self.recipe_region_combo.setCurrentIndex(index)
        for spin, coordinate in zip(
            self.recipe_region_spins, (x, y, x + width, y + height)
        ):
            spin.blockSignals(True)
            spin.setValue(coordinate)
            spin.blockSignals(False)
        self._on_recipe_region_coordinate_changed(0)

    def _update_recipe_overlay(self, _index: int = -1) -> None:
        if (
            self._resolution_mode != "5um"
            or self.control_tabs.currentIndex() != 1
            or self._display_mode not in {"original", "binary", "parallel"}
            or self._original_image is None
        ):
            self.original_view.clear_recipe_overlay()
            self.image_view.clear_recipe_overlay()
            self.original_view.set_roi_interactive(
                self._display_mode in {"original", "parallel"}
            )
            return
        self.original_view.set_roi_interactive(False)
        regions = [
            (name, _rect(value), color)
            for name, value, color in self._recipe_region_items
        ]
        selected = self.recipe_region_combo.currentIndex()
        self.original_view.set_recipe_overlay(regions, selected)
        self.image_view.set_recipe_overlay(regions, selected)

    def _flatten_parameters(
        self,
        value: Any,
        prefix: str = "",
    ) -> list[tuple[str, Any]]:
        if isinstance(value, _ParameterField):
            return [(prefix or "值", value)]
        if isinstance(value, dict):
            rows: list[tuple[str, Any]] = []
            for key, child_value in value.items():
                name = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(child_value, dict):
                    rows.extend(self._flatten_parameters(child_value, name))
                else:
                    rows.append((name, child_value))
            return rows
        return [(prefix or "值", value)]

    def _add_parameter_group(self, title: str, values: dict[str, Any]) -> None:
        group = QGroupBox(title)
        group.setObjectName("sectionGroup")
        layout = QGridLayout(group)
        layout.setContentsMargins(
            2 if self._resolution_mode == "5um" else 4,
            4 if self._resolution_mode == "5um" else 6,
            2 if self._resolution_mode == "5um" else 4,
            2 if self._resolution_mode == "5um" else 4,
        )
        layout.setHorizontalSpacing(2)
        layout.setVerticalSpacing(2 if self._resolution_mode == "5um" else 4)
        fields: list[tuple[QLabel, QLineEdit, bool, QPushButton | None]] = []
        for name, value in self._flatten_parameters(values):
            label_text = f"{name}："
            label = QLabel(label_text)
            label.setWordWrap(True)
            label.setToolTip(label_text)
            label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            editor = QLineEdit()
            if self._resolution_mode == "5um":
                editor.setObjectName("parameterEditor")
                editor.setFixedHeight(28)
            full_width = (
                isinstance(value, _ParameterField)
                and value.path[-1] in {
                    "tempPath", "modelPath", "onnxPath", "blindCircuitModelPath"
                }
            )
            if full_width:
                editor.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Fixed,
                )
            else:
                editor.setFixedWidth(
                    editor.fontMetrics().horizontalAdvance("000000") + 18
                )
            editor.setAlignment(Qt.AlignmentFlag.AlignLeft)
            if isinstance(value, _ParameterField):
                parameter_value = value.value
                editor.editingFinished.connect(
                    lambda e=editor, path=value.path: self._apply_local_parameter(
                        e,
                        path,
                    )
                )
                self._parameter_editors[value.path] = editor
            else:
                parameter_value = value
                editor.setReadOnly(True)
            model_button = None
            if (
                isinstance(value, _ParameterField)
                and value.path[-1] in {
                    "modelPath", "onnxPath", "blindCircuitModelPath"
                }
            ):
                model_button = QPushButton("选择模型文件")
                model_button.clicked.connect(
                    lambda _checked=False, path=value.path: self.select_runtime_model(path)
                )
            text = self._parameter_text(parameter_value)
            editor.setText(text)
            editor.setToolTip(text)
            fields.append((label, editor, full_width, model_button))
        self._parameter_group_fields.append((layout, fields))
        self._layout_parameter_fields(
            layout,
            fields,
            self._parameter_column_count(self.parameter_scroll.viewport().width()),
        )
        self.parameter_content_layout.addWidget(group)

    def _add_5um_limit_tables(self, detection: dict[str, Any]) -> None:
        zone_group = QGroupBox("分区卡控标准（单位：μm）")
        zone_group.setObjectName("sectionGroup")
        zone_layout = QVBoxLayout(zone_group)
        zone_layout.setContentsMargins(2, 4, 2, 2)
        zone_layout.setSpacing(2)
        zone_table = QTableWidget(4, 5)
        zone_table.setObjectName("limitTable")
        zone_table.setHorizontalHeaderLabels(
            ["缺陷类型", "A区", "B区", "C区", "dummy区"]
        )
        zone_table.verticalHeader().setVisible(False)
        zone_table.verticalHeader().setDefaultSectionSize(28)
        zone_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        zone_table.horizontalHeader().setSectionResizeMode(
            0,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        zone_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        zone_table.setAlternatingRowColors(True)
        zone_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        zone_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        zone_table.setFixedHeight(142)
        region_thresholds = detection.get("regionThresh", [])
        visible_rows = (("坏簇", 0), ("划伤", 1), ("磨砂", 2), ("异物", 4))
        zone_table.blockSignals(True)
        for display_row, (label, source_row) in enumerate(visible_rows):
            type_item = QTableWidgetItem(label)
            type_item.setData(Qt.ItemDataRole.UserRole, source_row)
            type_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            zone_table.setItem(display_row, 0, type_item)
            source_values = (
                region_thresholds[source_row]
                if isinstance(region_thresholds, list)
                and source_row < len(region_thresholds)
                and isinstance(region_thresholds[source_row], list)
                else []
            )
            for column in range(4):
                if column < len(source_values):
                    value_item = QTableWidgetItem(str(source_values[column]))
                    value_item.setFlags(
                        Qt.ItemFlag.ItemIsSelectable
                        | Qt.ItemFlag.ItemIsEditable
                        | Qt.ItemFlag.ItemIsEnabled
                    )
                else:
                    value_item = QTableWidgetItem("/")
                    value_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                zone_table.setItem(display_row, column + 1, value_item)
        zone_table.blockSignals(False)
        zone_table.itemChanged.connect(self._apply_5um_zone_limit_item)
        zone_layout.addWidget(zone_table)
        self._five_um_zone_limit_table = zone_table
        self.parameter_content_layout.addWidget(zone_group)

        other_group = QGroupBox("其他区域卡控标准")
        other_group.setObjectName("sectionGroup")
        other_layout = QVBoxLayout(other_group)
        other_layout.setContentsMargins(2, 4, 2, 2)
        other_layout.setSpacing(2)
        other_table = QTableWidget(2, 1)
        other_table.setObjectName("limitTable")
        other_table.setHorizontalHeaderLabels([""])
        other_table.horizontalHeader().setVisible(False)
        other_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        other_table.setVerticalHeaderLabels(["盲元破损", "崩边"])
        other_table.verticalHeader().setDefaultSectionSize(28)
        other_table.verticalHeader().setMinimumWidth(80)
        other_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        other_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        other_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        other_table.setFixedHeight(58)
        other_table.blockSignals(True)
        damaged_item = QTableWidgetItem(str(detection.get("damTh", "/")))
        if "damTh" in detection:
            damaged_item.setFlags(
                Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEditable
                | Qt.ItemFlag.ItemIsEnabled
            )
        else:
            damaged_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        other_table.setItem(0, 0, damaged_item)
        edge_item = QTableWidgetItem("/")
        edge_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        other_table.setItem(1, 0, edge_item)
        other_table.blockSignals(False)
        other_table.itemChanged.connect(self._apply_5um_other_limit_item)
        other_layout.addWidget(other_table)
        self._five_um_other_limit_table = other_table
        self.parameter_content_layout.addWidget(other_group)

    def _apply_5um_zone_limit_item(self, item: QTableWidgetItem) -> None:
        table = self._five_um_zone_limit_table
        params = self._runtime_recipe_params
        if table is None or params is None or item.column() < 1:
            return
        detection = params.get("DetectionParams")
        type_item = table.item(item.row(), 0)
        if not isinstance(detection, dict) or type_item is None:
            return
        source_row = type_item.data(Qt.ItemDataRole.UserRole)
        region_thresholds = detection.get("regionThresh")
        if (
            not isinstance(source_row, int)
            or not isinstance(region_thresholds, list)
            or source_row >= len(region_thresholds)
            or not isinstance(region_thresholds[source_row], list)
            or item.column() - 1 >= len(region_thresholds[source_row])
        ):
            return
        source_column = item.column() - 1
        original = region_thresholds[source_row][source_column]
        try:
            value = int(item.text().strip())
        except ValueError:
            table.blockSignals(True)
            item.setText(str(original))
            table.blockSignals(False)
            QMessageBox.warning(self, "参数格式错误", "分区卡控值必须是整数。")
            return
        region_thresholds[source_row][source_column] = value
        table.blockSignals(True)
        item.setText(str(value))
        table.blockSignals(False)
        self._auto_crop_output = None
        zone_name = table.horizontalHeaderItem(item.column()).text()
        self._append_log(f"本地参数已修改：{type_item.text()}.{zone_name} = {value}")

    def _apply_5um_other_limit_item(self, item: QTableWidgetItem) -> None:
        table = self._five_um_other_limit_table
        params = self._runtime_recipe_params
        if table is None or params is None or item.row() != 0 or item.column() != 0:
            return
        detection = params.get("DetectionParams")
        if not isinstance(detection, dict) or "damTh" not in detection:
            return
        original = detection["damTh"]
        try:
            value = int(item.text().strip())
        except ValueError:
            table.blockSignals(True)
            item.setText(str(original))
            table.blockSignals(False)
            QMessageBox.warning(self, "参数格式错误", "盲元破损值必须是整数。")
            return
        detection["damTh"] = value
        table.blockSignals(True)
        item.setText(str(value))
        table.blockSignals(False)
        editor = self._parameter_editors.get(("DetectionParams", "damTh"))
        if editor is not None:
            editor.setText(str(value))
            editor.setToolTip(str(value))
        self._auto_crop_output = None
        self._append_log(f"本地参数已修改：盲元破损 = {value}")

    def _parameter_column_count(self, width: int) -> int:
        if self._resolution_mode == "5um":
            return 3
        if width < 420:
            return 2
        if width < 760:
            return 3
        return 4

    @staticmethod
    def _layout_parameter_fields(
        layout: QGridLayout,
        fields: list[tuple[QLabel, QLineEdit, bool, QPushButton | None]],
        column_count: int,
    ) -> None:
        for label, editor, _full_width, model_button in fields:
            layout.removeWidget(label)
            layout.removeWidget(editor)
            if model_button is not None:
                layout.removeWidget(model_button)
        for column in range(8):
            layout.setColumnStretch(column, 0)
        row = 0
        column = 0
        for label, editor, full_width, model_button in fields:
            label.setMinimumWidth(0)
            label.setMaximumWidth(16777215)
            if full_width:
                if column:
                    row += 2
                    column = 0
                layout.addWidget(label, row, 0, 1, column_count)
                if model_button is None:
                    layout.addWidget(editor, row + 1, 0, 1, column_count)
                else:
                    layout.addWidget(editor, row + 1, 0, 1, column_count - 1)
                    layout.addWidget(model_button, row + 1, column_count - 1)
                layout.setColumnStretch(0, 1)
                row += 2
                continue
            layout.addWidget(label, row, column)
            layout.addWidget(
                editor,
                row + 1,
                column,
                Qt.AlignmentFlag.AlignLeft,
            )
            layout.setColumnStretch(column, 1)
            column += 1
            if column == column_count:
                row += 2
                column = 0
        layout.invalidate()

    def _reflow_parameter_groups(self, width: int | None = None) -> None:
        available_width = (
            self.parameter_scroll.viewport().width()
            if width is None
            else width
        )
        column_count = self._parameter_column_count(available_width)
        for layout, fields in self._parameter_group_fields:
            self._layout_parameter_fields(layout, fields, column_count)

    @staticmethod
    def _parameter_text(value: Any) -> str:
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return f"{value:.2f}".rstrip("0").rstrip(".")
        return str(value)

    def _parameter_value(self, path: tuple[str, ...]) -> Any:
        current: Any = self._runtime_recipe_params
        for key in path:
            if not isinstance(current, dict) or key not in current:
                raise KeyError(".".join(path))
            current = current[key]
        return current

    def _set_parameter_value(self, path: tuple[str, ...], value: Any) -> None:
        current: Any = self._runtime_recipe_params
        for key in path[:-1]:
            if not isinstance(current, dict) or key not in current:
                raise KeyError(".".join(path))
            current = current[key]
        if not isinstance(current, dict):
            raise KeyError(".".join(path))
        current[path[-1]] = value

    @staticmethod
    def _parse_parameter_text(text: str, original: Any) -> Any:
        stripped = text.strip()
        if isinstance(original, bool):
            normalized = stripped.lower()
            if normalized in {"true", "1", "是"}:
                return True
            if normalized in {"false", "0", "否"}:
                return False
            raise ValueError("布尔值请输入 true/false、1/0 或 是/否")
        if isinstance(original, int):
            return int(stripped)
        if isinstance(original, float):
            return float(stripped)
        if isinstance(original, (dict, list)):
            parsed = json.loads(stripped)
            if not isinstance(parsed, type(original)):
                expected = "对象" if isinstance(original, dict) else "数组"
                raise ValueError(f"请输入 JSON {expected}")
            return parsed
        return text

    def _apply_local_parameter(
        self,
        editor: QLineEdit,
        path: tuple[str, ...],
    ) -> None:
        if not editor.isModified():
            return
        try:
            original = self._parameter_value(path)
            value = self._parse_parameter_text(editor.text(), original)
            if path == ("DetectionParams", "blindCircuitThreshold") and not 0.0 <= value <= 1.0:
                raise ValueError("盲元/电路/外围区概率阈值必须在 0 到 1 之间。")
            if path == ("DetectionParams", "blindCircuitScoreThreshold") and not 0.0 <= value <= 1.0:
                raise ValueError("结果平均像素概率下限必须在 0 到 1 之间。")
            if path == ("DetectionParams", "coreMinArea") and not 0 <= value <= 99999:
                raise ValueError("外围区缺陷最小面积必须在 0 到 99999 之间。")
            self._set_parameter_value(path, value)
            self._auto_crop_output = None
            if path[-1] in {
                "padW",
                "padH",
                "padRange",
                "padSearchRange",
                "padMinNum",
                "padGrayThresh",
                "padTopToCmosEdgePix",
            }:
                self._apply_wafer_pad_params()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            try:
                editor.setText(self._parameter_text(self._parameter_value(path)))
            except KeyError:
                editor.clear()
            QMessageBox.warning(self, "参数格式错误", str(exc))
            return
        text = self._parameter_text(value)
        editor.setText(text)
        editor.setToolTip(text)
        if path == ("DetectionParams", "damTh"):
            table = self._five_um_other_limit_table
            if table is not None:
                table.blockSignals(True)
                table.item(0, 0).setText(text)
                table.blockSignals(False)
        self._append_log(f"本地参数已修改：{'.'.join(path)} = {text}")

    def _model_parameter_path(self) -> tuple[str, ...] | None:
        params = self._runtime_recipe_params
        if params is None:
            return None
        if self._resolution_mode == "5um":
            detection = params.get("DetectionParams")
            if isinstance(detection, dict) and "modelPath" in detection:
                return "DetectionParams", "modelPath"
            return None
        for section_name in ("detectParams", "DetectParams"):
            detection = params.get(section_name)
            if isinstance(detection, dict) and "onnxPath" in detection:
                return section_name, "onnxPath"
        return None

    def select_runtime_model(self, path: tuple[str, ...] | None = None) -> None:
        if path is None:
            path = self._model_parameter_path()
        if path is None:
            QMessageBox.warning(self, "提示", "当前配方没有模型路径参数。")
            return
        current_path = str(self._parameter_value(path))
        initial_path = (
            current_path
            if current_path and Path(current_path).exists()
            else self._settings.value("last_model_dir", "", type=str)
        )
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型文件",
            initial_path,
            (
                "TorchScript 模型 (*.pt);;所有文件 (*)"
                if self._resolution_mode == "20um"
                else (
                    "TensorRT Engine (*.engine);;ONNX 模型 (*.onnx);;所有文件 (*)"
                    if path[-1] == "blindCircuitModelPath"
                    else "ONNX 模型 (*.onnx);;所有文件 (*)"
                )
            ),
        )
        if not file_path:
            return
        self._set_parameter_value(path, file_path)
        editor = self._parameter_editors.get(path)
        if editor is not None:
            editor.setText(file_path)
            editor.setToolTip(file_path)
        self._settings.setValue("last_model_dir", str(Path(file_path).parent))
        self._append_log(f"本地模型路径已修改：{file_path}")

    def _select_labeled_parameters(
        self,
        source: dict[str, Any],
        fields: tuple[tuple[str, str], ...],
        prefix: tuple[str, ...],
    ) -> dict[str, _ParameterField]:
        return {
            self._parameter_display_label(label): _ParameterField(
                source[key],
                prefix + (key,),
            )
            for key, label in fields
            if key in source
        }

    def _parameter_display_label(self, label: str) -> str:
        if self._resolution_mode != "20um":
            return label
        if "（" in label and label.endswith("）"):
            prefix, description = label.split("（", 1)
            if any(char.isascii() and char.isalpha() for char in prefix):
                return description[:-1]
        return label

    def _twenty_um_cmos_parameters(
        self,
        detection: dict[str, Any],
        prefix: tuple[str, ...],
    ) -> dict[str, Any]:
        values: dict[str, Any] = {}
        values.update(
            self._select_labeled_parameters(
                detection,
                (
                    ("cmosAdhesiveDx0", "cmosAdhesiveDx0（胶水范围 X0，px）"),
                    ("cmosAdhesiveDx1", "cmosAdhesiveDx1（胶水范围 X1，px）"),
                    ("cmosAdhesiveDy0", "cmosAdhesiveDy0（胶水范围 Y0，px）"),
                    ("cmosAdhesiveDy1", "cmosAdhesiveDy1（胶水范围 Y1，px）"),
                    ("cmosDetectGrayThresh", "cmosDetectGrayThresh（灰度阈值）"),
                    ("cmosChipMinLengthUm", "cmosChipMinLengthUm（CMOS 崩边最小尺寸，um）"),
                    ("cmosChipMaxLengthUm", "cmosChipMaxLengthUm（CMOS 崩边最大尺寸，um）"),
                    ("cmosChipShrinkPix", "cmosChipShrinkPix（收缩像素，px）"),
                    ("cmosDirtyMinWidthPix", "cmosDirtyMinWidthPix（脏污最小宽度，px）"),
                ),
                prefix,
            )
        )
        return values

    def _parameter_groups(
        self,
        params: dict[str, Any],
    ) -> list[tuple[str, dict[str, Any]]]:
        if self._resolution_mode == "20um":
            split_name = "splitParams" if "splitParams" in params else "SplitParams"
            detect_name = "detectParams" if "detectParams" in params else "DetectParams"
            split = params.get(split_name, {})
            detect = params.get(detect_name, {})
            split = split if isinstance(split, dict) else {}
            detect = detect if isinstance(detect, dict) else {}
            split_prefix = (split_name,)
            detect_prefix = (detect_name,)
            split_fields = (
                ("ratio", "ratio（图像压缩比例）"),
                ("ksize", "ksize（高斯核大小）"),
                ("sigmax", "sigmax（高斯 sigma）"),
                ("matchedThresh", "matchedThresh（模板匹配阈值）"),
                ("tempPath", "tempPath（模板路径）"),
                ("tempW0", "tempW0（模板宽度）"),
                ("tempH0", "tempH0（模板高度）"),
                ("padW", "padW（pad 期望宽度，px）"),
                ("padH", "padH（pad 期望高度，px）"),
                ("padRange", "padRange（尺寸容差，宽高∈[pad±range]）"),
                ("padGrayThresh", "padGrayThresh（first pad/旋正二值化阈值）"),
                ("padSearchRange", "padSearchRange（搜索条带宽度，px）"),
                ("padMinNum", "padMinNum（列内最少 pad 数，first pad/旋正共用）"),
                ("maxRotateAngle", "maxRotateAngle（最大旋转角，度）"),
                ("capTopOffset", "capTopOffset（顶部偏移，px）"),
            )
            detection_groups = (
                (
                    "cap偏移",
                    (
                        ("capTopOffsetThreshPix", "capTopOffsetThreshPix（相对CMOS上边界的上偏移阈值，px）"),
                        ("padTopToCmosEdgePix", "padTopToCmosEdgePix（pad到CMOS上边界，px）"),
                        ("capBottomOffsetThreshPix", "capBottomOffsetThreshPix（下偏移阈值，px）"),
                        ("capLeftOffsetThreshPix", "capLeftOffsetThreshPix（左偏移阈值，px）"),
                        ("capRightOffsetThreshPix", "capRightOffsetThreshPix（右偏移阈值，px）"),
                    ),
                ),
                (
                    "cap翘曲",
                    (
                        ("capWarpDetectRange", "capWarpDetectRange（翘起范围，px）"),
                        ("capWarpDetectThresh", "capWarpDetectThresh（翘起阈值）"),
                    ),
                ),
                (
                    "成像区",
                    (
                        ("imagingLeft", "imagingLeft（成像区左边界，um）"),
                        ("imagingTop", "imagingTop（成像区上边界，um）"),
                        ("imagingRight", "imagingRight（成像区右边界，um）"),
                        ("imagingBottom", "imagingBottom（成像区下边界，um）"),
                        ("detectNonImaging", "detectNonImaging（检测非成像区）"),
                        ("imagingCannyThresh1", "imagingCannyThresh1（Canny 阈值1）"),
                        ("imagingCannyThresh2", "imagingCannyThresh2（Canny 阈值2）"),
                        ("capGrayThresh", "capGrayThresh（cap 灰度阈值）"),
                        ("imagingCannyK", "imagingCannyK（闭运算核）"),
                        ("imagingMinDefectArea", "imagingMinDefectArea（最小缺陷面积，px）"),
                    ),
                ),
                (
                    "cap崩边",
                    (
                        ("capChipMinLengthUm", "capChipMinLengthUm（cap 崩边最小尺寸，um）"),
                        ("capChipMaxLengthUm", "capChipMaxLengthUm（cap 崩边最大尺寸，um）"),
                        ("capChipShrinkPix", "capChipShrinkPix（收缩像素，px）"),
                    ),
                ),
                (
                    "cap点缺陷",
                    (
                        ("pointCircularity", "pointCircularity（点缺陷圆度）"),
                        ("capPointMinArea1", "capPointMinArea1（点缺陷面积1）"),
                        ("capPointMinArea2", "capPointMinArea2（点缺陷面积2）"),
                        ("capPointGrayThresh1", "capPointGrayThresh1（点缺陷灰度1）"),
                        ("capPointGrayThresh2", "capPointGrayThresh2（点缺陷灰度2）"),
                    ),
                ),
                (
                    "cap脏污",
                    (
                        ("capDirtyMinArea1", "capDirtyMinArea1（脏污面积1）"),
                        ("capDirtyMinArea2", "capDirtyMinArea2（脏污面积2）"),
                        ("capDirtyGrayThresh1", "capDirtyGrayThresh1（脏污灰度1）"),
                        ("capDirtyGrayThresh2", "capDirtyGrayThresh2（脏污灰度2）"),
                    ),
                ),
                (
                    "cap划伤",
                    (
                        ("scratchCircularity", "scratchCircularity（划伤圆度）"),
                        ("scratchAreaRatio", "scratchAreaRatio（划伤面积比）"),
                        ("capScratchMinLength", "capScratchMinLength（划伤最小长度，px）"),
                        ("capScratchMaxLength", "capScratchMaxLength（划伤最大长度，px）"),
                        ("capScratchCount", "capScratchCount（划伤数量）"),
                    ),
                ),
                (
                    "深浅共用",
                    (("darkSearchRatio", "darkSearchRatio（深浅搜索比例）"),),
                ),
            )
            model_fields = (
                ("onnxPath", "模型路径"),
                ("confThresh", "confThresh（模型置信度）"),
                ("nmsThresh", "nmsThresh（NMS 阈值）"),
            )
            used_detect = {
                key
                for _, fields in detection_groups
                for key, _ in fields
            } | {
                key for key, _ in model_fields
            } | {
                "binInfoVec", "cmosExcludeBoxes",
                "cmosExcludeX", "cmosExcludeY",
                "cmosExcludeW", "cmosExcludeH",
                "cmosSafeX", "cmosSafeY", "cmosSafeW", "cmosSafeH",
                "cmosAdhesiveDx0", "cmosAdhesiveDx1",
                "cmosAdhesiveDy0", "cmosAdhesiveDy1",
                "cmosDetectGrayThresh", "cmosChipMinLengthUm",
                "cmosChipMaxLengthUm", "cmosChipShrinkPix",
                "cmosDirtyMinWidthPix",
            }
            groups = [
                (
                    "裁剪参数",
                    self._select_labeled_parameters(split, split_fields, split_prefix),
                )
            ]
            groups.extend(
                (
                    title,
                    self._select_labeled_parameters(detect, fields, detect_prefix),
                )
                for title, fields in detection_groups
            )
            groups.extend(
                (
                    ("CMOS", self._twenty_um_cmos_parameters(detect, detect_prefix)),
                    (
                        "模型",
                        self._select_labeled_parameters(
                            detect,
                            model_fields,
                            detect_prefix,
                        ),
                    ),
                    (
                        "其他参数",
                        {
                            f"其他参数{index + 1}": _ParameterField(
                                value,
                                detect_prefix + (key,),
                            )
                            for index, (key, value) in enumerate(
                                (item for item in detect.items() if item[0] not in used_detect)
                            )
                        },
                    ),
                )
            )
            return groups

        split = params.get("SplitImageParams", {})
        split = split if isinstance(split, dict) else {}
        detection = params.get("DetectionParams", {})
        detection = detection if isinstance(detection, dict) else {}
        model_fields = (
            ("modelPath", "模型路径"),
            ("blindCircuitModelPath", "盲元/电路/外围区模型"),
            ("blindCircuitThreshold", "盲元/电路/外围区像素分割阈值"),
            ("blindCircuitScoreThreshold", "盲元/电路/外围区结果平均像素概率下限"),
            ("badwire_conf_thresh", "坏线置信度"),
            ("bad_conf_thresh", "坏簇置信度"),
            ("foreign_conf_thresh", "异物置信度"),
            ("scratch_conf_thresh", "划伤置信度"),
            ("nmsThreshold", "NMS 阈值"),
        )
        frosted_fields = (
            ("lf_rate", "磨砂低频"),
            ("mf_rate", "磨砂高频"),
            ("abrazineThresh", "磨砂阈值"),
        )
        edge_fields = (
            ("thresh_gray_chip", "崩边灰度阈值"),
            ("offset", "崩边检测偏移"),
            ("safeLine", "安全线偏移"),
            ("w0", "图像宽 (px)"),
            ("h0", "图像高 (px)"),
            ("damTh", "崩边阈值"),
        )
        pad_fields = (
            ("padGrayThresh", "Pad 灰度阈值"),
            ("padSearchRange", "搜索条带宽度"),
            ("padMinNum", "单侧最少 Pad 数"),
            ("padW", "Pad 宽度"),
            ("padH", "Pad 高度"),
            ("padRange", "尺寸容差"),
            ("stain_threshold", "脏污灰度阈值"),
            ("padTh", "脏污面积比例"),
            ("coreMinArea", "外围区缺陷最小面积 (px²)"),
            ("kernel", "形态学核大小"),
            ("mainExpand", "主体扩展像素"),
        )
        used_detection = {
            "blindDetectionZone", "circuitDetectionZone",
            "outerDetectionZones", "excludeZones",
            "regionCoords", "regionThresh", "core_gray_thresh",
            "interPadGrayThresh",
        } | {
            key
            for fields in (model_fields, frosted_fields, edge_fields, pad_fields)
            for key, _ in fields
        }
        remaining = {
            key: _ParameterField(value, ("DetectionParams", key))
            for key, value in detection.items()
            if key not in used_detection
        }
        return [
            (
                "裁剪参数",
                {
                    "模板路径": _ParameterField(
                        split["tempPath"],
                        ("SplitImageParams", "tempPath"),
                    )
                }
                if "tempPath" in split
                else {},
            ),
            (
                "模型参数",
                self._select_labeled_parameters(
                    detection,
                    model_fields,
                    ("DetectionParams",),
                ),
            ),
            (
                "磨砂检测",
                self._select_labeled_parameters(
                    detection,
                    frosted_fields,
                    ("DetectionParams",),
                ),
            ),
            (
                "崩边检测",
                self._select_labeled_parameters(
                    detection,
                    edge_fields,
                    ("DetectionParams",),
                ),
            ),
            (
                "Pad 定位 / 脏污检测",
                self._select_labeled_parameters(
                    detection,
                    pad_fields,
                    ("DetectionParams",),
                ),
            ),
            ("其他参数", remaining),
        ]

    def _update_auto_crop_button(self) -> None:
        has_input = (
            bool(self.folder_edit.text().strip())
            and os.path.isdir(self.folder_edit.text().strip())
            if self._resolution_mode == "20um"
            else self._source_image is not None
        )
        ready = (
            has_input
            and self._runtime_recipe_params is not None
            and 0 <= self.wafer_recipe_combo.currentIndex() < len(self._wafer_recipes)
            and not (
                self._crop_thread is not None
                and self._crop_thread.isRunning()
            )
            and not (
                self._region_crop_thread is not None
                and self._region_crop_thread.isRunning()
            )
        )
        self.auto_crop_button.setEnabled(ready)

    def _handle_region_crop_clicked(
        self, _checked: bool = False, configured_zones: bool = False
    ) -> None:
        if self._region_crop_thread is not None and self._region_crop_thread.isRunning():
            if self._region_crop_worker is not None:
                self._region_crop_stop_requested = True
                self._region_crop_worker.request_stop()
                self._set_status("正在停止区域裁剪")
                self._update_detection_button()
            return
        self.start_batch_region_crop(configured_zones=configured_zones)

    def _handle_configured_region_crop_clicked(self, _checked: bool = False) -> None:
        self._handle_region_crop_clicked(configured_zones=True)

    def start_batch_region_crop(self, configured_zones: bool = False) -> None:
        folder = self.crop_folder_edit.text().strip()
        if self._resolution_mode != "5um" or not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "提示", "请先选择有效的 5um 裁剪图文件夹。")
            return
        if (
            self._runtime_recipe_params is None
            or not 0 <= self.wafer_recipe_combo.currentIndex() < len(self._wafer_recipes)
        ):
            QMessageBox.warning(self, "提示", "请先选择有效的 5um 配方。")
            return
        default_output_folder = Path(folder).resolve().parent / "split"
        try:
            default_output_folder.mkdir(exist_ok=True)
        except OSError as exc:
            QMessageBox.critical(self, "批量裁剪失败", f"无法创建保存文件夹：{exc}")
            return
        output_folder = QFileDialog.getExistingDirectory(
            self,
            "选择配置框裁剪图保存文件夹" if configured_zones else "选择盲元电路外围区裁剪图保存文件夹",
            str(default_output_folder),
        )
        if not output_folder:
            return
        self._region_crop_stop_requested = False
        self._region_crop_configured_zones = configured_zones
        self._region_crop_thread = QThread(self)
        self._region_crop_worker = BatchRegionCropWorker(
            folder, output_folder, deepcopy(self._runtime_recipe_params),
            self._wafer_recipes[self.wafer_recipe_combo.currentIndex()].recipe_name,
            configured_zones,
        )
        self._region_crop_worker.moveToThread(self._region_crop_thread)
        self._region_crop_thread.started.connect(self._region_crop_worker.run)
        self._region_crop_worker.progress.connect(self._on_region_crop_progress)
        self._region_crop_worker.finished.connect(self._on_region_crop_finished)
        self._region_crop_worker.stopped.connect(self._on_region_crop_stopped)
        self._region_crop_worker.failed.connect(self._on_region_crop_failed)
        self._region_crop_worker.finished.connect(self._region_crop_thread.quit)
        self._region_crop_worker.stopped.connect(self._region_crop_thread.quit)
        self._region_crop_worker.failed.connect(self._region_crop_thread.quit)
        self._region_crop_thread.finished.connect(self._cleanup_region_crop_thread)
        self._region_crop_thread.start()
        mode_name = "配置框" if configured_zones else "盲元电路外围区"
        self._set_status(f"开始{mode_name}批量裁剪")
        self._append_log(f"{mode_name}批量裁剪：{folder}，保存至 {output_folder}")
        self._update_auto_crop_button()
        self._update_detection_button()

    def _on_region_crop_progress(
        self, image_index: int, image_count: int, saved_count: int
    ) -> None:
        if not self._region_crop_stop_requested:
            self._set_status(
                f"区域裁剪：{image_index}/{image_count} 张原图，已保存 {saved_count} 张"
            )

    def _on_region_crop_finished(self, image_count: int, saved_count: int) -> None:
        mode_name = "配置框" if self._region_crop_configured_zones else "盲元电路外围区"
        message = f"{mode_name}裁剪完成：{image_count} 张原图，共保存 {saved_count} 张"
        self._set_status(message)
        self._append_log(message)

    def _on_region_crop_stopped(self, image_count: int, saved_count: int) -> None:
        mode_name = "配置框" if self._region_crop_configured_zones else "盲元电路外围区"
        message = f"{mode_name}裁剪已停止：已处理 {image_count} 张原图，保存 {saved_count} 张"
        self._set_status(message)
        self._append_log(message)

    def _on_region_crop_failed(self, message: str) -> None:
        self._set_status(f"区域裁剪失败：{message}")
        QMessageBox.critical(self, "批量裁剪失败", message)

    def _cleanup_region_crop_thread(self) -> None:
        if self._region_crop_worker is not None:
            self._region_crop_worker.deleteLater()
            self._region_crop_worker = None
        if self._region_crop_thread is not None:
            self._region_crop_thread.deleteLater()
            self._region_crop_thread = None
        self._region_crop_stop_requested = False
        self._region_crop_configured_zones = False
        self._update_auto_crop_button()
        self._update_detection_button()

    def start_auto_crop(self) -> None:
        index = self.wafer_recipe_combo.currentIndex()
        if self._resolution_mode == "20um":
            folder = self.folder_edit.text().strip()
            if not folder or not os.path.isdir(folder):
                QMessageBox.warning(self, "提示", "请先选择有效的小图文件夹。")
                return
        elif self._source_image is None:
            QMessageBox.warning(self, "提示", "请先加载或拼接待裁剪图像。")
            return
        if not 0 <= index < len(self._wafer_recipes):
            QMessageBox.warning(self, "提示", "请先连接 MySQL 并选择配方。")
            return
        if self._runtime_recipe_params is None:
            QMessageBox.warning(self, "提示", "请重新选择配方。")
            return
        crop_image = None if self._resolution_mode == "20um" else self._source_image

        recipe = self._wafer_recipes[index]
        self.auto_crop_button.setEnabled(False)
        self.batch_region_crop_button.setEnabled(False)
        self.batch_configured_region_crop_button.setEnabled(False)
        self.run_detection_button.setEnabled(False)
        self.run_detection_all_button.setEnabled(False)
        self._set_status(
            f"开始运行 {self._resolution_mode} 自动裁剪，配方：{recipe.recipe_name}"
        )
        self._crop_thread = QThread(self)
        self._crop_worker = CropWorker(
            crop_image,
            self._resolution_mode,
            deepcopy(self._runtime_recipe_params),
            self._up_image if self._resolution_mode == "20um" else None,
            self._last_auto_crop_angle,
            self.folder_edit.text().strip(),
            self.machine_combo.currentText(),
        )
        self._crop_worker.moveToThread(self._crop_thread)
        self._crop_thread.started.connect(self._crop_worker.run)
        self._crop_worker.log.connect(self._append_log)
        self._crop_worker.finished.connect(self._on_auto_crop_finished)
        self._crop_worker.failed.connect(self._on_auto_crop_failed)
        self._crop_worker.finished.connect(self._crop_thread.quit)
        self._crop_worker.failed.connect(self._crop_thread.quit)
        self._crop_thread.finished.connect(self._cleanup_crop_thread)
        self._crop_thread.start()

    def _cleanup_crop_thread(self) -> None:
        if self._crop_worker is not None:
            self._crop_worker.deleteLater()
            self._crop_worker = None
        if self._crop_thread is not None:
            self._crop_thread.deleteLater()
            self._crop_thread = None
        self._update_auto_crop_button()
        self._update_detection_button()

    def _on_auto_crop_finished(self, output: object) -> None:
        crop_output = output  # type: ignore[assignment]
        if not isinstance(crop_output, AutoCropOutput):
            self._on_auto_crop_failed("自动裁剪返回结果类型错误。")
            return
        if crop_output.crop_rect is None:
            self._on_auto_crop_failed("自动裁剪未返回裁剪框。")
            return
        self.set_working_image(crop_output.image)
        self._auto_crop_output = crop_output
        self._up_image = crop_output.up_image
        self._down_image = crop_output.down_image
        self._last_auto_crop_angle = crop_output.angle
        self.set_display_mode("original", force=True)
        rect = self._clamp_roi_rect(crop_output.crop_rect)
        if rect is None:
            self._on_auto_crop_failed("自动裁剪框无效。")
            return
        self._roi_rect = rect
        self.original_view.set_roi_from_spin(*rect)
        self._mirror_roi_to_binary()
        self._update_offset_display()
        _x, _y, width, height = rect
        self._set_status(
            f"自动裁剪完成，已生成 {width}×{height} 裁剪框，可拖动或缩放微调"
        )

    def _on_auto_crop_failed(self, message: str) -> None:
        self._set_status(f"自动裁剪失败：{message}")
        QMessageBox.critical(self, "自动裁剪失败", message)

    def _update_detection_button(self) -> None:
        pair_selected = bool(
            self.cropped_up_edit.text().strip()
            or self.cropped_down_edit.text().strip()
        )
        has_single_input = (
            (
                (
                    not self.cropped_up_edit.text().strip()
                    or Path(self.cropped_up_edit.text().strip()).is_file()
                ) and (
                    not self.cropped_down_edit.text().strip()
                    or Path(self.cropped_down_edit.text().strip()).is_file()
                )
                if pair_selected
                else bool(self.folder_edit.text().strip())
                and os.path.isdir(self.folder_edit.text().strip())
            )
            if self._resolution_mode == "20um"
            else (
                bool(self.crop_folder_edit.text().strip())
                and os.path.isdir(self.crop_folder_edit.text().strip())
                if self.crop_folder_edit.text().strip()
                else self._source_image is not None
            )
        )
        has_batch_input = (
            bool(self.folder_edit.text().strip())
            and os.path.isdir(self.folder_edit.text().strip())
            and not pair_selected
            if self._resolution_mode == "20um"
            else bool(self.crop_folder_edit.text().strip())
            and os.path.isdir(self.crop_folder_edit.text().strip())
        )
        detection_running = (
            self._detection_thread is not None
            and self._detection_thread.isRunning()
        )
        region_crop_running = (
            self._region_crop_thread is not None
            and self._region_crop_thread.isRunning()
        )
        ready = (
            self._runtime_recipe_params is not None
            and 0 <= self.wafer_recipe_combo.currentIndex() < len(self._wafer_recipes)
            and not detection_running
            and not region_crop_running
            and not (
                self._crop_thread is not None
                and self._crop_thread.isRunning()
            )
        )
        self.run_detection_button.setEnabled(ready and has_single_input)
        if region_crop_running:
            active_button = (
                self.batch_configured_region_crop_button
                if self._region_crop_configured_zones
                else self.batch_region_crop_button
            )
            inactive_button = (
                self.batch_region_crop_button
                if self._region_crop_configured_zones
                else self.batch_configured_region_crop_button
            )
            active_button.setText(
                "停止中" if self._region_crop_stop_requested else "停止裁剪"
            )
            active_button.setEnabled(not self._region_crop_stop_requested)
            inactive_button.setEnabled(False)
        else:
            self.batch_region_crop_button.setText("批量裁剪盲元电路外围区")
            self.batch_configured_region_crop_button.setText("按配置框批量裁剪")
            self.batch_region_crop_button.setEnabled(
                ready and self._resolution_mode == "5um" and has_batch_input
            )
            self.batch_configured_region_crop_button.setEnabled(
                ready and self._resolution_mode == "5um" and has_batch_input
            )
        if detection_running and self._detection_all:
            self.run_detection_all_button.setText(
                "停止中" if self._detection_stop_requested else "停止检测"
            )
            self.run_detection_all_button.setEnabled(
                not self._detection_stop_requested
            )
        else:
            self.run_detection_all_button.setText(
                "批量检测" if self._resolution_mode == "5um" else "全部检测"
            )
            self.run_detection_all_button.setEnabled(ready and has_batch_input)

    def _handle_detect_all_clicked(self, _checked: bool = False) -> None:
        if (
            self._detection_thread is not None
            and self._detection_thread.isRunning()
            and self._detection_all
        ):
            if self._detection_worker is not None:
                self._detection_stop_requested = True
                self._detection_worker.request_stop()
                self._set_status("正在停止检测")
                self._update_detection_button()
            return
        self.run_traditional_detection(detect_all=True)

    def run_traditional_detection(self, *, detect_all: bool = False) -> None:
        index = self.wafer_recipe_combo.currentIndex()
        crop_pair: tuple[str | None, str | None] | None = None
        if self._resolution_mode == "20um":
            up_path = self.cropped_up_edit.text().strip()
            down_path = self.cropped_down_edit.text().strip()
            if up_path or down_path:
                if detect_all:
                    QMessageBox.warning(self, "提示", "已裁剪图只支持检测单张。")
                    return
                if (up_path and not Path(up_path).is_file()) or (
                    down_path and not Path(down_path).is_file()
                ):
                    QMessageBox.warning(self, "提示", "请选择有效的已裁剪图。")
                    return
                crop_pair = (up_path or None, down_path or None)
            else:
                folder = self.folder_edit.text().strip()
                if not folder or not os.path.isdir(folder):
                    QMessageBox.warning(self, "提示", "请先选择有效的小图文件夹。")
                    return
        elif self.crop_folder_edit.text().strip() or detect_all:
            crop_folder = self.crop_folder_edit.text().strip()
            if not crop_folder or not os.path.isdir(crop_folder):
                QMessageBox.warning(self, "提示", "请先选择有效的裁剪图文件夹。")
                return
        elif self._source_image is None:
            QMessageBox.warning(self, "提示", "请先加载或拼接待检测图像。")
            return
        if not 0 <= index < len(self._wafer_recipes):
            QMessageBox.warning(self, "提示", "请先连接 MySQL 并选择配方。")
            return
        recipe = self._wafer_recipes[index]
        if self._runtime_recipe_params is None:
            QMessageBox.warning(self, "提示", "请重新选择配方。")
            return
        model_parameter_path = self._model_parameter_path()
        model_path = (
            str(self._parameter_value(model_parameter_path)).strip()
            if model_parameter_path is not None
            else ""
        )
        expected_suffix = ".pt" if self._resolution_mode == "20um" else ".onnx"
        needs_model = not (
            self._resolution_mode == "20um"
            and crop_pair is not None
            and crop_pair[0] is None
        )
        if needs_model and (
            not model_path
            or Path(model_path).suffix.lower() != expected_suffix
            or not Path(model_path).is_file()
        ):
            QMessageBox.warning(
                self,
                "模型文件无效",
                f"请先选择有效的 {self._resolution_mode} "
                f"{expected_suffix.upper()} 模型文件。",
            )
            return
        if self._resolution_mode == "5um":
            detection = self._runtime_recipe_params.get("DetectionParams", {})
            blind_circuit_path = str(detection.get(
                "blindCircuitModelPath", DEFAULT_BLIND_CIRCUIT_MODEL_PATH
            )).strip()
            if (
                not blind_circuit_path
                or Path(blind_circuit_path).suffix.lower() not in {".engine", ".onnx"}
                or not Path(blind_circuit_path).is_file()
            ):
                QMessageBox.warning(
                    self, "模型文件无效", "请先选择有效的盲元/电路/外围区 Engine 或 ONNX 模型文件。"
                )
                return
        detection_name = (
            f"{self._resolution_mode} {'批量' if self._resolution_mode == '5um' else '全部'}检测"
            if detect_all
            else f"{self._resolution_mode} 检测"
        )
        if crop_pair is not None:
            detection_name = (
                "20um CAP 上图检测" if crop_pair[1] is None
                else "20um CMOS 下图检测" if crop_pair[0] is None
                else "20um 上下图检测"
            )
        self._detection_all = detect_all
        self._detection_stop_requested = False
        self.run_detection_button.setEnabled(False)
        self.run_detection_all_button.setEnabled(False)
        self.batch_region_crop_button.setEnabled(False)
        self.batch_configured_region_crop_button.setEnabled(False)
        self.load_wafer_recipes_button.setEnabled(False)
        self.import_wafer_recipes_button.setEnabled(False)
        self.detection_result_edit.setPlainText(f"{detection_name}运行中")
        self._clear_detection_outputs()
        self._set_status(
            f"开始运行 {detection_name}，配方：{recipe.recipe_name}"
        )
        self._detection_thread = QThread(self)
        self._single_detection_file_name = (
            self._crop_folder_paths[self._crop_folder_index].name
            if self._resolution_mode == "5um"
            and not detect_all
            and self.crop_folder_edit.text().strip()
            and 0 <= self._crop_folder_index < len(self._crop_folder_paths)
            else None
        )
        self._single_detection_layer_name = detection_name if crop_pair else None
        detection_image = (
            self._original_image
            if self._resolution_mode == "5um" and not detect_all
            else None
        )
        self._detection_worker = TraditionalDetectionWorker(
            detection_image,
            self._resolution_mode,
            deepcopy(self._runtime_recipe_params),
            None if self._resolution_mode == "20um" else self._up_image,
            None if self._resolution_mode == "20um" else self._auto_crop_output,
            (
                self.crop_folder_edit.text().strip()
                if self._resolution_mode == "5um"
                else self.folder_edit.text().strip()
            ),
            self.machine_combo.currentText(),
            detect_all,
            crop_pair,
        )
        self._detection_worker.moveToThread(self._detection_thread)
        self._detection_thread.started.connect(self._detection_worker.run)
        self._detection_worker.log.connect(self._append_log)
        self._detection_worker.finished.connect(self._on_detection_finished)
        self._detection_worker.batch_finished.connect(
            self._on_detection_batch_finished
        )
        self._detection_worker.image_finished.connect(
            self._on_detection_image_finished
        )
        self._detection_worker.all_finished.connect(
            self._on_all_detection_finished
        )
        self._detection_worker.stopped.connect(self._on_detection_stopped)
        self._detection_worker.failed.connect(self._on_detection_failed)
        self._detection_worker.finished.connect(self._detection_thread.quit)
        self._detection_worker.all_finished.connect(self._detection_thread.quit)
        self._detection_worker.stopped.connect(self._detection_thread.quit)
        self._detection_worker.failed.connect(self._detection_thread.quit)
        self._detection_thread.finished.connect(self._cleanup_detection_thread)
        self._detection_thread.start()
        self._update_detection_button()

    def _cleanup_detection_thread(self) -> None:
        if self._detection_worker is not None:
            self._detection_worker.deleteLater()
            self._detection_worker = None
        if self._detection_thread is not None:
            self._detection_thread.deleteLater()
            self._detection_thread = None
        self.load_wafer_recipes_button.setEnabled(True)
        self.import_wafer_recipes_button.setEnabled(True)
        self._detection_all = False
        self._detection_stop_requested = False
        self._update_detection_button()

    def _on_detection_finished(self, output: object) -> None:
        if not isinstance(output, DetectionOutput):
            self._on_detection_failed("检测返回结果类型错误。")
            return
        layer_name = self._single_detection_layer_name or self._resolution_mode
        if self._single_detection_file_name is not None:
            layer_name = f"5um 单张：{self._single_detection_file_name}"
        self._show_detection_output(output, layer_name)

    def _on_detection_batch_finished(
        self,
        output: object,
        batch_index: int,
        batch_count: int,
    ) -> None:
        if not isinstance(output, DetectionOutput):
            self._on_detection_failed("检测返回结果类型错误。")
            return
        self._show_detection_output(
            output,
            f"20um 第 {batch_index + 1}/{batch_count} 批",
            activate=batch_index == 0,
        )

    def _on_detection_image_finished(
        self,
        output: object,
        file_name: str,
        image_index: int,
        image_count: int,
    ) -> None:
        if not isinstance(output, DetectionOutput):
            self._on_detection_failed("检测返回结果类型错误。")
            return
        self._show_detection_output(
            output,
            f"5um 第 {image_index + 1}/{image_count} 张：{file_name}",
            activate=image_index == 0,
        )

    def _on_all_detection_finished(
        self,
        completed_count: int,
        ignored_count: int,
    ) -> None:
        if self._resolution_mode == "5um":
            message = f"5um 批量检测完成，共完成 {completed_count} 张"
        else:
            message = f"20um 全部检测完成，共完成 {completed_count} 批"
        if ignored_count:
            message += f"，末尾 {ignored_count} 张不足一批已跳过"
        self._set_status(message)

    def _on_detection_stopped(self, completed_count: int) -> None:
        unit = "张" if self._resolution_mode == "5um" else "批"
        message = f"{self._resolution_mode} 检测已停止，已完成 {completed_count} {unit}"
        if self._detection_output is None:
            self.detection_result_edit.setPlainText(message)
        self._set_status(message)

    def _show_detection_output(
        self,
        output: DetectionOutput,
        layer_name: str,
        *,
        activate: bool = True,
    ) -> None:
        cache_paths = None
        if self._detection_all and output.source_image is not None:
            if self._detection_cache_dir is None:
                self._detection_cache_dir = QTemporaryDir()
            if not self._detection_cache_dir.isValid():
                if self._detection_worker is not None:
                    self._detection_worker.request_stop()
                self._on_detection_failed("无法创建检测结果临时文件夹。")
                return
            cache_dir = Path(self._detection_cache_dir.path())
            cache_index = len(self._detection_outputs)
            source_path = cache_dir / f"{cache_index}_source.png"
            result_path = cache_dir / f"{cache_index}_result.png"
            try:
                saved = (
                    imwrite_unicode(str(source_path), output.source_image)
                    and imwrite_unicode(str(result_path), output.image)
                )
            except OSError:
                saved = False
            if not saved:
                if self._detection_worker is not None:
                    self._detection_worker.request_stop()
                self._on_detection_failed("无法保存检测结果临时文件。")
                return
            cache_paths = (source_path, result_path)
            output.source_image = None
            output.image = np.empty((0, 0), dtype=np.uint8)
        self._detection_outputs.append((output, layer_name))
        self._detection_cache_paths.append(cache_paths)
        self._show_detection_output_at(
            len(self._detection_outputs) - 1,
            activate=activate,
        )

    def _show_detection_output_at(
        self,
        output_index: int,
        *,
        activate: bool = True,
    ) -> None:
        if not 0 <= output_index < len(self._detection_outputs):
            return
        showing_detection = self._display_mode == "detection"
        output, layer_name = self._detection_outputs[output_index]
        cache_paths = self._detection_cache_paths[output_index]
        if cache_paths is not None and output.image.size == 0:
            try:
                source_image = imread_unicode(str(cache_paths[0]))
                result_image = imread_unicode(str(cache_paths[1]))
            except OSError:
                source_image = None
                result_image = None
            if source_image is None or result_image is None:
                self._on_detection_failed("无法读取检测结果临时文件。")
                return
            output.source_image = source_image
            output.image = result_image
        previous_index = self._detection_output_index
        if previous_index != output_index and 0 <= previous_index < len(self._detection_outputs):
            if self._detection_cache_paths[previous_index] is not None:
                previous_output = self._detection_outputs[previous_index][0]
                previous_output.source_image = None
                previous_output.image = np.empty((0, 0), dtype=np.uint8)
        self._detection_output_index = output_index
        self._detection_output = output
        if output.source_image is not None:
            self._source_image = output.source_image
            self._original_image = output.source_image
            if self._resolution_mode == "5um":
                self.rotation_spin.blockSignals(True)
                self.rotation_spin.setValue(0.0)
                self.rotation_spin.blockSignals(False)
            self._gray_image = (
                cv2.cvtColor(output.source_image, cv2.COLOR_BGR2GRAY)
                if output.source_image.ndim == 3
                else output.source_image.copy()
            )
            self._auto_crop_output = None
            self._roi_rect = None
            self._crop_image = None
            self._crop_gray = None
            self.original_view.clear_roi()
            self.image_view.clear_roi()
            self.update_binary_image()
            self._update_gray_mean_display()
        lines = [layer_name, f"缺陷数量：{len(output.items)}"]
        for index, item in enumerate(output.items, start=1):
            _x, _y, width, height = item.bbox
            confidence_name = (
                "平均像素概率"
                if self._resolution_mode == "5um" and item.algorithm in {
                    "外围区模型检测", "盲元区", "电路区"
                }
                else "置信度"
            )
            confidence = (
                f" / {confidence_name} {item.confidence:.2f}"
                if item.confidence is not None
                else ""
            )
            darkness = (
                f" / 深浅度 {item.darkness:.2f}"
                if item.darkness is not None
                else ""
            )
            lines.append(
                f"{index}. {item.algorithm} / {item.label} / "
                f"外接框宽高：{width} × {height} 像素 / 面积 {item.area:.1f}"
                f"{darkness}{confidence}"
            )
        self.detection_result_edit.setPlainText("\n".join(lines))
        self.mode_detection_btn.setEnabled(True)
        self._update_detection_navigation_buttons()
        if activate or showing_detection:
            self.set_display_mode("detection", force=True)
        self._set_status(
            f"{layer_name} 检测完成，发现 {len(output.items)} 个缺陷"
        )

    def _clear_detection_outputs(self) -> None:
        self._detection_outputs.clear()
        self._detection_cache_paths.clear()
        if self._detection_cache_dir is not None:
            self._detection_cache_dir.remove()
            self._detection_cache_dir = None
        self._detection_output_index = -1
        self._detection_output = None
        self.mode_detection_btn.setEnabled(False)
        self._update_detection_navigation_buttons()

    def _update_detection_navigation_buttons(self) -> None:
        if self._detection_outputs:
            count = len(self._detection_outputs)
            current = self._detection_output_index
        elif self._resolution_mode == "5um" and self.crop_folder_edit.text().strip():
            count = len(self._crop_folder_paths)
            current = self._crop_folder_index
        else:
            count = 0
            current = -1
        self.detection_page_label.setText(
            f"{current + 1}/{count}" if current >= 0 else "0/0"
        )
        self.previous_detection_button.setEnabled(current > 0)
        self.next_detection_button.setEnabled(0 <= current < count - 1)

    def switch_detection_output(self, step: int) -> None:
        if self._detection_outputs:
            target_index = self._detection_output_index + step
            self._show_detection_output_at(target_index)
        else:
            self.switch_crop_folder_image(step)

    def _on_detection_failed(self, message: str) -> None:
        self.detection_result_edit.setPlainText(f"运行失败：{message}")
        self._set_status(f"检测运行失败：{message}")
        QMessageBox.critical(self, "检测运行失败", message)

    def _build_shortcuts(self) -> None:
        # 保留快捷键入口，方向键不绑定主界面预览切换。
        return

    def set_roi_draw_mode(self, enabled: bool) -> None:
        if enabled:
            self.set_measure_mode(False)
        # 裁剪框只能画在原图上；二值图/裁剪图禁止绘制
        if self._display_mode in {"binary", "crop"}:
            self.original_view.set_draw_mode(False)
            self.image_view.set_draw_mode(False)
            return
        self.original_view.set_draw_mode(enabled)
        self.image_view.set_draw_mode(False)

    def set_measure_mode(self, enabled: bool) -> None:
        if enabled:
            self.original_view.set_draw_mode(False)
            self.image_view.set_draw_mode(False)
        self._measure_mode = enabled
        self.original_view.set_measure_mode(
            enabled and self._display_mode in {"original", "parallel", "crop"}
        )
        self.image_view.set_measure_mode(
            enabled and self._display_mode in {"binary", "parallel"}
        )

    def keyPressEvent(self, event) -> None:  # noqa: ANN001
        if event.key() == Qt.Key.Key_W and not event.isAutoRepeat():
            self.set_roi_draw_mode(True)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: ANN001
        if event.key() == Qt.Key.Key_W and not event.isAutoRepeat():
            self.set_roi_draw_mode(False)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def select_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "选择小图文件夹",
            self.folder_edit.text() or self._settings.value("last_folder", "", type=str),
        )
        if folder:
            self.folder_edit.setText(folder)
            self.cropped_up_edit.clear()
            self.cropped_down_edit.clear()
            self._settings.setValue("last_folder", folder)
            self._clear_stitch_results()
            self._append_log(f"已选择小图文件夹：{folder}")
            self._update_auto_crop_button()
            self._update_detection_button()

    def select_cropped_pair_image(self, layer: str) -> None:
        last_dir = self._settings.value("last_cropped_pair_dir", "", type=str)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择已裁剪上图" if layer == "up" else "选择已裁剪下图",
            last_dir if last_dir and os.path.isdir(last_dir) else self.folder_edit.text(),
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;所有文件 (*)",
        )
        if not file_path:
            return
        image = imread_unicode(file_path)
        if image is None:
            QMessageBox.critical(self, "加载失败", f"无法读取图片：\n{file_path}")
            return
        edit = self.cropped_up_edit if layer == "up" else self.cropped_down_edit
        edit.setText(file_path)
        self._settings.setValue("last_cropped_pair_dir", str(Path(file_path).parent))
        self.set_working_image(image)
        self._append_log(f"已选择 20um 已裁剪{'上图' if layer == 'up' else '下图'}：{file_path}")

    def clear_cropped_pair_image(self, layer: str) -> None:
        edit = self.cropped_up_edit if layer == "up" else self.cropped_down_edit
        edit.clear()
        self._update_detection_button()

    def select_crop_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "选择裁剪图文件夹",
            self.crop_folder_edit.text()
            or self._settings.value("last_crop_detection_folder", "", type=str),
        )
        if folder:
            paths = sorted(
                (
                    path for path in Path(folder).iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                ),
                key=natural_sort_key,
            )
            if not paths:
                QMessageBox.warning(self, "提示", "裁剪图文件夹内没有可预览图片。")
                return
            image = imread_unicode(str(paths[0]))
            if image is None:
                QMessageBox.critical(self, "加载失败", f"无法读取图片：\n{paths[0]}")
                return
            self.crop_folder_edit.setText(folder)
            self.path_edit.clear()
            self._settings.setValue("last_crop_detection_folder", folder)
            self._settings.remove("last_image_path")
            self._crop_folder_paths = paths
            self._crop_folder_index = 0
            self.set_working_image(image)
            self.set_display_mode("original", force=True)
            self._append_log(f"已选择裁剪图文件夹：{folder}")
            self._set_status(f"裁剪图 1/{len(paths)}：{paths[0].name}")

    def switch_crop_folder_image(self, step: int) -> None:
        target_index = self._crop_folder_index + step
        if not 0 <= target_index < len(self._crop_folder_paths):
            return
        path = self._crop_folder_paths[target_index]
        image = imread_unicode(str(path))
        if image is None:
            QMessageBox.critical(self, "加载失败", f"无法读取图片：\n{path}")
            return
        self._crop_folder_index = target_index
        self.set_working_image(image)
        self._set_status(
            f"裁剪图 {target_index + 1}/{len(self._crop_folder_paths)}：{path.name}"
        )

    def select_image(self) -> None:
        last_image_dir = self._settings.value("last_image_dir", "", type=str)
        initial_path = (
            last_image_dir
            if last_image_dir and os.path.isdir(last_image_dir)
            else self.path_edit.text()
        )
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择小图",
            initial_path,
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;所有文件 (*)",
        )
        if file_path:
            self._settings.setValue("last_image_dir", str(Path(file_path).parent))
            self.load_image(file_path)

    def _restore_last_paths(self) -> None:
        folder = self._settings.value("last_folder", "", type=str)
        if folder and os.path.isdir(folder):
            self.folder_edit.setText(folder)
        crop_folder = self._settings.value("last_crop_detection_folder", "", type=str)
        if crop_folder and os.path.isdir(crop_folder):
            self.crop_folder_edit.setText(crop_folder)
        machine = self._settings.value(
            f"last_machine_{self._resolution_mode}",
            get_default_machine(self._resolution_mode),
            type=str,
        )
        if machine in get_machine_presets(self._resolution_mode):
            self.machine_combo.setCurrentText(machine)

    def _wafer_pad_source(
        self,
    ) -> tuple[dict[str, Any], tuple[str, ...], dict[str, Any]]:
        params = self._runtime_recipe_params
        if not isinstance(params, dict):
            return {}, (), {}
        if self._resolution_mode == "5um":
            detection = params.get("DetectionParams", {})
            detection = detection if isinstance(detection, dict) else {}
            return detection, ("DetectionParams",), detection
        split_name = "splitParams" if "splitParams" in params else "SplitParams"
        detect_name = "detectParams" if "detectParams" in params else "DetectParams"
        split = params.get(split_name, {})
        detection = params.get(detect_name, {})
        split = split if isinstance(split, dict) else {}
        detection = detection if isinstance(detection, dict) else {}
        return split, (split_name,), detection

    def _set_pad_actions_enabled(self, enabled: bool) -> None:
        self.find_pad_button.setEnabled(enabled)
        self.find_all_pads_button.setEnabled(enabled)

    def _apply_wafer_pad_params(self) -> None:
        source, _prefix, offset_source = self._wafer_pad_source()
        required = (
            "padW",
            "padH",
            "padRange",
            "padSearchRange",
            "padMinNum",
            "padGrayThresh",
        )
        missing = [key for key in required if key not in source]
        if missing:
            self._pad_params.clear()
            self._pad_top_to_cmos_edge_pix = None
            self._set_pad_actions_enabled(False)
            self._first_pad = None
            self._pad_rects = []
            self._pad_band_mode = None
            self._clear_pad_overlay()
            self._set_status(
                "当前 wafer_cpp 配方缺少 Pad 参数：" + ", ".join(missing)
            )
            return
        try:
            self._pad_params = {key: int(source[key]) for key in required}
            top_offset = offset_source.get("padTopToCmosEdgePix")
            self._pad_top_to_cmos_edge_pix = (
                int(top_offset) if top_offset is not None else None
            )
        except (TypeError, ValueError) as exc:
            self._pad_params.clear()
            self._pad_top_to_cmos_edge_pix = None
            self._set_pad_actions_enabled(False)
            self._set_status(f"当前 wafer_cpp 配方的 Pad 参数无效：{exc}")
            return
        threshold = max(0, min(255, self._pad_params["padGrayThresh"]))
        self.threshold_slider.blockSignals(True)
        self.threshold_spinbox.blockSignals(True)
        self.threshold_slider.setValue(threshold)
        self.threshold_spinbox.setValue(threshold)
        self.threshold_slider.blockSignals(False)
        self.threshold_spinbox.blockSignals(False)
        self.update_binary_image()
        self._set_pad_actions_enabled(True)
        if (
            self._first_pad is not None
            or self._pad_rects
            or self._pad_band_mode is not None
        ):
            self._apply_pad_overlay()
            self._update_pad_status_text()

    def apply_threshold_to_current_recipe(self) -> None:
        source, prefix, _offset_source = self._wafer_pad_source()
        if not prefix or "padGrayThresh" not in source:
            QMessageBox.warning(
                self,
                "提示",
                "当前 wafer_cpp 配方没有 padGrayThresh 参数。",
            )
            return
        threshold = int(self.threshold_slider.value())
        path = prefix + ("padGrayThresh",)
        self._set_parameter_value(path, threshold)
        editor = self._parameter_editors.get(path)
        if editor is not None:
            text = self._parameter_text(threshold)
            editor.setText(text)
            editor.setToolTip(text)
        self._apply_wafer_pad_params()
        self._set_status(
            f"已将阈值 {threshold} 应用到当前 wafer_cpp 运行配方"
        )

    def _pad_size_kwargs(self) -> dict[str, int]:
        p = self._pad_params
        return {
            "pad_min_num": p["padMinNum"],
            "pad_w": p["padW"],
            "pad_h": p["padH"],
            "pad_range": p["padRange"],
        }

    def locate_first_pad(self) -> None:
        gray = self._pad_source_gray()
        if gray is None:
            self._set_status("定位 Pad：请先加载或拼接图像")
            QMessageBox.warning(self, "提示", "请先加载或拼接图像。")
            return
        if not self._pad_params:
            QMessageBox.warning(self, "提示", "当前 wafer_cpp 配方缺少 Pad 参数。")
            return
        thresh = self.threshold_slider.value()
        p = self._pad_params
        column = find_pad_column(
            gray,
            thresh,
            search_width=p["padSearchRange"],
            **self._pad_size_kwargs(),
        )
        if not column:
            self._first_pad = None
            self._pad_rects = []
            self._pad_band_mode = "left"
            self._apply_pad_overlay()
            scope = "裁剪图" if self._display_mode == "crop" else "整图"
            self._set_status(
                f"定位 Pad：未找到符合条件的 pad（{scope}，阈值={thresh}）"
            )
            return
        first = min(column, key=lambda r: r[1])
        # first 放第一个便于高亮
        ordered = [first] + [r for r in column if r != first]
        self._first_pad = first
        self._pad_rects = ordered
        self._pad_band_mode = "left"
        self._apply_pad_overlay()
        self._update_pad_status_text()

    def locate_all_pads(self) -> None:
        gray = self._pad_source_gray()
        if gray is None:
            self._set_status("定位所有 Pad：请先加载或拼接图像")
            QMessageBox.warning(self, "提示", "请先加载或拼接图像。")
            return
        if not self._pad_params:
            QMessageBox.warning(self, "提示", "当前 wafer_cpp 配方缺少 Pad 参数。")
            return
        thresh = self.threshold_slider.value()
        scope = "裁剪图" if self._display_mode == "crop" else "整图"
        p = self._pad_params
        _angle, left_pads, right_pads = estimate_pad_rotation(
            gray,
            thresh,
            search_width=p["padSearchRange"],
            **self._pad_size_kwargs(),
        )
        all_pads = list(left_pads) + list(right_pads)
        if not all_pads:
            self._first_pad = None
            self._pad_rects = []
            self._pad_band_mode = "both"
            self._apply_pad_overlay()
            self._set_status(
                f"定位所有 Pad：未找到 pad（{scope}，阈值={thresh}）"
            )
            return
        first = min(left_pads, key=lambda r: r[1]) if left_pads else None
        ordered = ([first] if first else []) + [r for r in all_pads if r != first]
        self._first_pad = first
        self._pad_rects = ordered
        self._pad_band_mode = "both"
        self._apply_pad_overlay(draw_line=first is not None)
        self._set_status(
            f"定位所有 Pad：左 {len(left_pads)} / 右 {len(right_pads)}，"
            f"合计 {len(all_pads)}，阈值={thresh}（{scope}）"
        )

    def _pad_source_gray(self) -> np.ndarray | None:
        if self._display_mode == "crop":
            return self._crop_gray
        return self._gray_image

    def _pad_line_y(self) -> int | None:
        if self._first_pad is None or self._pad_top_to_cmos_edge_pix is None:
            return None
        return int(self._first_pad[1] - self._pad_top_to_cmos_edge_pix)

    def _pad_line_out_of_bounds(self, line_y: int) -> bool:
        gray = self._pad_source_gray()
        if gray is None:
            return True
        height = int(gray.shape[0])
        return line_y < 0 or line_y >= height

    def _update_pad_status_text(self) -> None:
        if self._first_pad is None and not self._pad_rects:
            return
        thresh = self.threshold_slider.value()
        scope = "裁剪图" if self._display_mode == "crop" else "整图"
        n = len(self._pad_rects)
        if self._first_pad is None:
            self._set_status(
                f"定位 Pad：找到 {n} 个 pad，无 first pad，阈值={thresh}（{scope}）"
            )
            return
        x, y, w, h = self._first_pad
        line_y = self._pad_line_y()
        if line_y is None:
            self._set_status(
                f"定位 Pad：pads={n}，first=({x}, {y}, {w}, {h})，"
                f"阈值={thresh}（{scope}）"
            )
            return
        text = (
            f"定位 Pad：pads={n}，first=({x}, {y}, {w}, {h})，"
            f"白线 y={line_y}，阈值={thresh}（{scope}）"
        )
        if self._pad_line_out_of_bounds(line_y):
            gray = self._pad_source_gray()
            height = 0 if gray is None else int(gray.shape[0])
            text += f"；白线越界（图像高度 {height}px），请在 wafer_cpp 配方中减小 padTopToCmosEdgePix"
        self._set_status(text)

    def _pad_search_bands(
        self, mode: str | None
    ) -> list[tuple[int, int, int, int]]:
        if mode not in ("left", "both"):
            return []
        gray = self._pad_source_gray()
        if gray is None:
            return []
        height, width = int(gray.shape[0]), int(gray.shape[1])
        if width < 1 or height < 1:
            return []
        if not self._pad_params:
            return []
        search = int(self._pad_params["padSearchRange"])
        if mode == "left":
            sw = min(search, width)
            if sw <= 0:
                return []
            return [(0, 0, sw, height)]
        sw = min(search, max(1, width // 2))
        if sw <= 0:
            return []
        return [(0, 0, sw, height), (width - sw, 0, sw, height)]

    def _clear_pad_overlay(self) -> None:
        self._pad_band_mode = None
        self.original_view.clear_pad_overlay()
        self.image_view.clear_pad_overlay()

    def _apply_pad_overlay(self, *, draw_line: bool = True) -> None:
        bands = self._pad_search_bands(self._pad_band_mode)
        if not self._pad_rects and self._first_pad is None:
            if not bands:
                self._clear_pad_overlay()
                return
            self.original_view.set_pad_overlay(
                [], None, highlight_index=None, search_bands=bands
            )
            if self._display_mode != "crop":
                self.image_view.set_pad_overlay(
                    [], None, highlight_index=None, search_bands=bands
                )
            else:
                self.image_view.clear_pad_overlay()
            return
        rects = list(self._pad_rects)
        if not rects and self._first_pad is not None:
            rects = [self._first_pad]
        highlight = 0 if self._first_pad is not None and rects else None
        line_y = None
        if (
            draw_line
            and self._first_pad is not None
            and self._pad_top_to_cmos_edge_pix is not None
        ):
            line_y = float(
                self._first_pad[1] - self._pad_top_to_cmos_edge_pix
            )
        # 越界仍绘制，由状态栏提示；不钳制到图像内
        self.original_view.set_pad_overlay(
            rects, line_y, highlight_index=highlight, search_bands=bands
        )
        if self._display_mode != "crop":
            self.image_view.set_pad_overlay(
                rects, line_y, highlight_index=highlight, search_bands=bands
            )
        else:
            self.image_view.clear_pad_overlay()

    def _capture_roi_rect(self) -> tuple[int, int, int, int] | None:
        if self._display_mode != "crop":
            rect = self.original_view.roi_rect()
            if rect is not None and rect.width() >= 1 and rect.height() >= 1:
                return (
                    int(round(rect.x())),
                    int(round(rect.y())),
                    int(round(rect.width())),
                    int(round(rect.height())),
                )
        return self._roi_rect

    def _clamp_roi_rect(
        self, roi: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int] | None:
        if self._original_image is None:
            return None
        height, width = self._original_image.shape[:2]
        x, y, w, h = roi
        x0 = max(0, min(width - 1, x))
        y0 = max(0, min(height - 1, y))
        x1 = max(x0 + 1, min(width, x + w))
        y1 = max(y0 + 1, min(height, y + h))
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1 - x0, y1 - y0)

    def _prepare_crop_from_roi(self) -> bool:
        if self._original_image is None:
            self._crop_image = None
            self._crop_gray = None
            return False
        roi = self._capture_roi_rect()
        if roi is None:
            self._crop_image = None
            self._crop_gray = None
            return False
        clamped = self._clamp_roi_rect(roi)
        if clamped is None:
            self._crop_image = None
            self._crop_gray = None
            return False
        x0, y0, w, h = clamped
        self._roi_rect = clamped
        crop = self._original_image[y0:y0 + h, x0:x0 + w]
        if crop.size == 0:
            self._crop_image = None
            self._crop_gray = None
            return False
        self._crop_image = crop.copy()
        if crop.ndim == 2:
            self._crop_gray = crop.copy()
        else:
            self._crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return True

    def _show_crop_on_view(self, *, fit_view: bool) -> None:
        if self._crop_image is None:
            return
        self.original_view.set_image(
            numpy_to_qimage(self._crop_image),
            clear_roi=True,
            keep_view=not fit_view,
        )
        if fit_view:
            self.original_view.fit_image()

    def _restore_full_image_views(self, *, fit_view: bool) -> None:
        if self._original_image is None:
            return
        saved_roi = self._roi_rect
        self.original_view.set_image(
            numpy_to_qimage(self._original_image),
            clear_roi=True,
            keep_view=not fit_view,
        )
        self.update_binary_image()
        self._roi_rect = saved_roi
        if saved_roi is not None:
            self.original_view.set_roi_from_spin(*saved_roi)
            self._mirror_roi_to_binary()
        if fit_view:
            self.fit_images()

    def _sync_display_mode_buttons(self, mode: str) -> None:
        self.mode_original_btn.setChecked(mode == "original")
        self.mode_binary_btn.setChecked(mode == "binary")
        self.mode_parallel_btn.setChecked(mode == "parallel")
        self.mode_crop_btn.setChecked(mode == "crop")
        self.mode_detection_btn.setChecked(mode == "detection")

    def _sync_mode_switch(self) -> None:
        self.mode_switch.blockSignals(True)
        self.mode_switch.setValue(1 if self._resolution_mode == "5um" else 0)
        self.mode_switch.blockSignals(False)
        self.mode_20um_button.setChecked(self._resolution_mode == "20um")
        self.mode_5um_button.setChecked(self._resolution_mode == "5um")

    def set_resolution_mode(self, target_mode: str) -> None:
        if target_mode == self._resolution_mode:
            self._sync_mode_switch()
            return
        if self._region_crop_thread is not None and self._region_crop_thread.isRunning():
            self._sync_mode_switch()
            self._set_status("请先停止盲元电路区裁剪，再切换分辨率")
            return
        self._resolution_mode = target_mode
        self._auto_crop_output = None
        self._settings.setValue("last_resolution_mode", target_mode)
        self._sync_mode_switch()
        self.run_detection_all_button.setText(
            "批量检测" if target_mode == "5um" else "全部检测"
        )
        self.crop_folder_edit.setVisible(target_mode == "5um")
        self.browse_crop_folder_button.setVisible(target_mode == "5um")
        self.cropped_up_edit.setVisible(target_mode == "20um")
        self.cropped_down_edit.setVisible(target_mode == "20um")
        self.browse_cropped_up_button.setVisible(target_mode == "20um")
        self.browse_cropped_down_button.setVisible(target_mode == "20um")
        self.clear_cropped_up_button.setVisible(target_mode == "20um")
        self.clear_cropped_down_button.setVisible(target_mode == "20um")
        self.batch_region_crop_button.setVisible(target_mode == "5um")
        self.batch_configured_region_crop_button.setVisible(target_mode == "5um")
        self.recipe_region_group.setVisible(target_mode == "5um")
        self._clear_stitch_results()
        self.stitch_layer_widget.setVisible(target_mode == "20um")
        self._first_pad = None
        self._pad_rects = []
        self._pad_band_mode = None
        self._clear_pad_overlay()
        self._wafer_recipes.clear()
        self.wafer_recipe_combo.clear()
        self._clear_parameter_groups()
        self.detection_result_edit.clear()
        self._update_detection_button()
        try:
            machine_ids = list_machine_ids(target_mode)
            default_machine = get_default_machine(target_mode)
        except Exception as exc:  # noqa: BLE001
            self.machine_combo.clear()
            self.machine_combo.setEnabled(False)
            self.stitch_button.setEnabled(False)
            self.stitch_all_button.setEnabled(False)
            self.manage_recipe_button.setEnabled(True)
            self._set_status(f"{target_mode} 参数文件不可用")
            QMessageBox.critical(
                self,
                "参数加载失败",
                f"无法读取拼接参数：\n{params_json_path(target_mode)}\n\n{exc}",
            )
            return

        self.machine_combo.blockSignals(True)
        self.machine_combo.clear()
        self.machine_combo.addItems(machine_ids)
        remembered_machine = self._settings.value(
            f"last_machine_{target_mode}", default_machine, type=str
        )
        self.machine_combo.setCurrentText(
            remembered_machine if remembered_machine in machine_ids else default_machine
        )
        self.machine_combo.blockSignals(False)
        if remembered_machine in machine_ids:
            self._settings.setValue(
                f"last_machine_{target_mode}", remembered_machine
            )
        self.machine_combo.setEnabled(True)
        self.stitch_button.setEnabled(True)
        self.stitch_all_button.setEnabled(True)
        self._set_status(f"已切换到 {target_mode} 模式")
        self._update_offset_display()
        self.refresh_wafer_recipes()

    def open_recipe_manager(self) -> None:
        current_machine = self.machine_combo.currentText()
        dialog = RecipeManagerDialog(
            self._resolution_mode, self, selected_machine=current_machine
        )
        dialog.exec()
        try:
            machine_ids = list_machine_ids(self._resolution_mode)
        except FileNotFoundError:
            machine_ids = []
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "参数加载失败", str(exc))
            return
        self.machine_combo.blockSignals(True)
        self.machine_combo.clear()
        self.machine_combo.addItems(machine_ids)
        if current_machine in machine_ids:
            self.machine_combo.setCurrentText(current_machine)
        elif machine_ids:
            self.machine_combo.setCurrentIndex(0)
        self.machine_combo.blockSignals(False)
        enabled = bool(machine_ids)
        self.machine_combo.setEnabled(enabled)
        self.stitch_button.setEnabled(enabled)
        self.stitch_all_button.setEnabled(enabled)
        self._clear_stitch_results()
        self._set_status(
            f"{self._resolution_mode} 配方已刷新"
            if enabled
            else f"{self._resolution_mode} 暂无配方"
        )
        self._update_offset_display()

    def _remember_machine(self, machine_id: str) -> None:
        if machine_id:
            self._settings.setValue(
                f"last_machine_{self._resolution_mode}", machine_id
            )
            self._append_log(f"机台配方已切换：{machine_id}")

    def open_stitch_calibration(self) -> None:
        if self._calibration_dialog is not None:
            if self._calibration_dialog.isMinimized():
                self._calibration_dialog.setWindowState(
                    self._calibration_dialog.windowState()
                    & ~Qt.WindowState.WindowMinimized
                    | Qt.WindowState.WindowActive
                )
            self._calibration_dialog.show()
            self._calibration_dialog.raise_()
            self._calibration_dialog.activateWindow()
            return
        machine_id = self.machine_combo.currentText()
        if not machine_id:
            QMessageBox.warning(self, "提示", "请先选择机台配方。")
            return
        try:
            dialog = StitchCalibrationDialog(
                self._resolution_mode,
                machine_id,
                self.folder_edit.text().strip(),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "打开失败", str(exc))
            return
        self._calibration_dialog = dialog
        dialog.finished.connect(self._on_calibration_closed)
        dialog.show()
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            frame = dialog.frameGeometry()
            frame.moveCenter(screen.availableGeometry().center())
            dialog.move(frame.topLeft())
        dialog.raise_()
        dialog.activateWindow()

    def _on_calibration_closed(self, _result: int) -> None:
        self._calibration_dialog = None

    def _clear_stitch_results(self, _value: str = "") -> None:
        self._stitched_images.clear()
        self._stitched_up_images.clear()
        self._stitch_layer = "down"
        self._down_image = None
        self._up_image = None
        self._clear_detection_outputs()
        self._ignored_image_count = 0
        self.stitch_preview_combo.clear()
        self.stitch_preview_combo.setEnabled(False)
        self.save_stitch_button.setEnabled(False)
        self.previous_stitch_button.setEnabled(False)
        self.next_stitch_button.setEnabled(False)
        self.stitch_down_button.setChecked(True)
        self.stitch_down_button.setEnabled(False)
        self.stitch_up_button.setEnabled(False)

    def start_stitch(self, _checked: bool = False, stitch_all: bool = False) -> None:
        folder = self.folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "提示", "请先选择有效的小图文件夹。")
            return
        if self._stitch_thread is not None and self._stitch_thread.isRunning():
            QMessageBox.information(self, "提示", "正在拼接，请稍候。")
            return

        machine_id = self.machine_combo.currentText()
        self._settings.setValue(f"last_machine_{self._resolution_mode}", machine_id)
        self._clear_stitch_results()
        self._stitch_all = stitch_all
        self._stitch_finished = False
        self.stitch_button.setEnabled(False)
        self.stitch_all_button.setEnabled(False)
        self.mode_switch.setEnabled(False)
        self.mode_20um_button.setEnabled(False)
        self.mode_5um_button.setEnabled(False)
        self.manage_recipe_button.setEnabled(False)
        action = "全部拼接" if stitch_all else "拼接单张"
        self._set_status(f"正在使用 {machine_id} 配方{action}…")

        self._stitch_thread = QThread(self)
        self._stitch_worker = StitchWorker(
            folder,
            machine_id,
            stitch_all,
            self._resolution_mode,
        )
        self._stitch_worker.moveToThread(self._stitch_thread)
        self._stitch_thread.started.connect(self._stitch_worker.run)
        self._stitch_worker.batch_finished.connect(self._on_stitch_batch_finished)
        self._stitch_worker.finished.connect(self._on_stitch_finished)
        self._stitch_worker.failed.connect(self._on_stitch_failed)
        self._stitch_worker.finished.connect(self._stitch_thread.quit)
        self._stitch_worker.failed.connect(self._stitch_thread.quit)
        self._stitch_thread.finished.connect(self._cleanup_stitch_thread)
        self._stitch_thread.start()

    def _cleanup_stitch_thread(self) -> None:
        if self._stitch_worker is not None:
            self._stitch_worker.deleteLater()
            self._stitch_worker = None
        if self._stitch_thread is not None:
            self._stitch_thread.deleteLater()
            self._stitch_thread = None
        self.stitch_button.setEnabled(True)
        self.stitch_all_button.setEnabled(True)
        self.mode_switch.setEnabled(True)
        self.mode_20um_button.setEnabled(True)
        self.mode_5um_button.setEnabled(True)
        self.manage_recipe_button.setEnabled(True)

    def _on_stitch_batch_finished(
        self,
        up_image: object,
        stitched_image: object,
        batch_index: int,
    ) -> None:
        self._stitched_up_images.append(up_image)  # type: ignore[arg-type]
        self._stitched_images.append(stitched_image)  # type: ignore[arg-type]
        params = get_joint_params(
            self.machine_combo.currentText(), self._resolution_mode
        )
        start = batch_index * params.joint_count + 1
        end = start + params.joint_count - 1
        self.stitch_preview_combo.addItem(
            f"第 {batch_index + 1} 批（第 {start}–{end} 张）"
        )
        self.stitch_preview_combo.setEnabled(True)
        self.save_stitch_button.setEnabled(True)
        if batch_index == 0:
            self.stitch_preview_combo.setCurrentIndex(0)
            self.preview_stitched_image(0)
        else:
            self._update_stitch_navigation_buttons()

    def _on_stitch_finished(self, ignored_count: int) -> None:
        self._ignored_image_count = ignored_count
        self._stitch_finished = True
        self.preview_stitched_image(self.stitch_preview_combo.currentIndex())

    def _update_stitch_navigation_buttons(self) -> None:
        batch_index = self.stitch_preview_combo.currentIndex()
        self.previous_stitch_button.setEnabled(batch_index > 0)
        self.next_stitch_button.setEnabled(
            0 <= batch_index < len(self._stitched_images) - 1
        )

    def switch_stitched_image(self, step: int) -> None:
        if not self._stitched_images:
            return
        current_index = self.stitch_preview_combo.currentIndex()
        target_index = current_index + step
        if 0 <= target_index < len(self._stitched_images):
            self.stitch_preview_combo.setCurrentIndex(target_index)

    def preview_stitched_image(self, batch_index: int) -> None:
        if not 0 <= batch_index < len(self._stitched_images):
            return
        down_image = self._stitched_images[batch_index]
        self._down_image = down_image
        self._up_image = self._stitched_up_images[batch_index]
        if self._stitch_layer == "up" and self._up_image is None:
            self._stitch_layer = "down"
        current_image = (
            self._up_image
            if self._stitch_layer == "up"
            else self._down_image
        )
        if current_image is None:
            return
        self.stitch_down_button.setEnabled(True)
        self.stitch_up_button.setEnabled(self._up_image is not None)
        self.stitch_down_button.setChecked(self._stitch_layer == "down")
        self.stitch_up_button.setChecked(self._stitch_layer == "up")
        height, width = current_image.shape[:2]
        layer_name = "上图" if self._stitch_layer == "up" else "下图"
        status = (
            f"当前预览第 {batch_index + 1}/{len(self._stitched_images)} 批，"
            f"{layer_name}尺寸 {width}×{height}"
        )
        if self._ignored_image_count:
            status += f"，末尾 {self._ignored_image_count} 张不足一批已跳过"
        elif not self._stitch_finished and self._stitch_all:
            status += f"，已完成 {len(self._stitched_images)} 批，正在继续拼接"
        self._set_status(status)
        self.set_working_image(current_image, source_label=f"拼接{layer_name}")
        self._update_stitch_navigation_buttons()

    def show_stitched_layer(self, layer: str) -> None:
        if layer not in {"up", "down"}:
            return
        if layer == "up" and self._up_image is None:
            return
        self._stitch_layer = layer
        self.preview_stitched_image(self.stitch_preview_combo.currentIndex())

    def _on_stitch_failed(self, message: str) -> None:
        self._stitch_finished = True
        self._set_status("拼接失败")
        QMessageBox.critical(self, "拼接失败", message)

    def save_stitched_image(self) -> None:
        current_image = (
            self._up_image
            if self._stitch_layer == "up"
            else self._down_image
        )
        if current_image is None or not self._stitched_images:
            QMessageBox.warning(self, "提示", "当前没有可保存的拼接图。")
            return
        batch_index = self.stitch_preview_combo.currentIndex()
        default_dir = self._settings.value("last_stitch_save_dir", "", type=str)
        if not default_dir or not os.path.isdir(default_dir):
            folder = self.folder_edit.text().strip()
            default_dir = folder if os.path.isdir(folder) else str(Path.cwd())
        layer_name = "up" if self._stitch_layer == "up" else "down"
        default_path = str(
            Path(default_dir) / f"stitched_{layer_name}_{batch_index + 1}.jpg"
        )
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存拼接图",
            default_path,
            "JPEG (*.jpg);;PNG (*.png);;BMP (*.bmp)",
        )
        if not file_path:
            return
        if not imwrite_unicode(file_path, current_image):
            QMessageBox.critical(self, "保存失败", f"无法写入：\n{file_path}")
            return
        self._settings.setValue("last_stitch_save_dir", str(Path(file_path).parent))
        self._append_log(f"已保存拼接图：{file_path}")
        QMessageBox.information(self, "完成", f"已保存拼接图：\n{file_path}")

    def load_image(self, file_path: str) -> None:
        image = imread_unicode(file_path)
        if image is None:
            QMessageBox.critical(self, "加载失败", f"无法读取图片：\n{file_path}")
            return
        if self._resolution_mode == "5um":
            self.crop_folder_edit.clear()
            self._settings.remove("last_crop_detection_folder")
            self._crop_folder_paths.clear()
            self._crop_folder_index = -1
        self.path_edit.setText(file_path)
        self._settings.setValue("last_image_path", file_path)
        self._up_image = None
        self.set_working_image(image, source_label=file_path)
        self._append_log(f"已加载图片：{file_path}")

    def set_working_image(self, image: np.ndarray, source_label: str = "") -> None:
        if self._resolution_mode == "5um" and source_label.startswith("拼接"):
            self.crop_folder_edit.clear()
            self._settings.remove("last_crop_detection_folder")
            self._crop_folder_paths.clear()
            self._crop_folder_index = -1
        self._source_image = image
        self._auto_crop_output = None
        self._clear_detection_outputs()
        self._first_pad = None
        self._pad_rects = []
        self._roi_rect = None
        self._crop_image = None
        self._crop_gray = None
        if self._display_mode == "crop":
            self._display_mode = "original"
            self._sync_display_mode_buttons("original")
        if source_label and not source_label.startswith("拼接"):
            self.path_edit.setText(source_label)
        self.image_view.clear_roi()
        self.original_view.clear_roi()
        self._clear_pad_overlay()
        self.rotation_spin.blockSignals(True)
        self.rotation_spin.setValue(0.0)
        self.rotation_spin.blockSignals(False)
        self._apply_rotation(fit_view=True, keep_roi=False)
        self._update_auto_crop_button()
        self._update_detection_button()

    def _on_rotation_changed(self, _angle: float) -> None:
        # 旋转微调时保持当前缩放/平移，避免每次改角度都 fit
        self._apply_rotation(fit_view=False, keep_roi=True)

    def _apply_rotation(self, *, fit_view: bool, keep_roi: bool = True) -> None:
        if self._source_image is None:
            return

        self._first_pad = None
        self._pad_rects = []
        self._pad_band_mode = None
        saved_roi = self._capture_roi_rect() if keep_roi else None

        angle = float(self.rotation_spin.value())
        rotated = rotate_image(self._source_image, angle)
        self._original_image = rotated
        self._gray_image = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        self._update_gray_mean_display()

        restored_roi: tuple[int, int, int, int] | None = None
        if saved_roi is not None:
            restored_roi = self._clamp_roi_rect(saved_roi)
        self._roi_rect = restored_roi

        if self._display_mode == "crop":
            self.update_binary_image()
            if restored_roi is not None and self._prepare_crop_from_roi():
                self.original_view.setVisible(True)
                self.image_view.setVisible(False)
                self._show_crop_on_view(fit_view=fit_view)
                self._clear_pad_overlay()
                self._roi_rect = restored_roi
            else:
                self._display_mode = "original"
                self._sync_display_mode_buttons("original")
                self._crop_image = None
                self._crop_gray = None
                self._roi_rect = restored_roi
                self._restore_full_image_views(fit_view=fit_view)
                self._set_status("裁剪图：ROI 无效，已切回原图")
            self._update_recipe_overlay()
            return

        self.original_view.set_image(
            numpy_to_qimage(rotated),
            clear_roi=True,
            keep_view=not fit_view,
        )
        self.update_binary_image()
        # set_image(clear_roi) 会发出空 ROI 信号，需在其后写回
        self._roi_rect = restored_roi

        if restored_roi is not None:
            # 框坐标不随旋转变换，仅按新图尺寸钳制后还原
            self.original_view.set_roi_from_spin(*restored_roi)
            self._mirror_roi_to_binary()
            self._update_offset_display()

        if fit_view:
            self.set_display_mode(self._display_mode, force=True)
            self._roi_rect = restored_roi
            if restored_roi is not None:
                self.original_view.set_roi_from_spin(*restored_roi)
                self._mirror_roi_to_binary()
                self._update_offset_display()
        else:
            # 旋转后面板尺寸可能变化，同步并排另一侧视角但不重置缩放
            if self._display_mode == "parallel":
                self._sync_views_from(self.original_view)
        self._update_recipe_overlay()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        mime_data = event.mimeData()
        if mime_data.hasUrls() and any(url.isLocalFile() for url in mime_data.urls()):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            local_path = url.toLocalFile()
            if os.path.isdir(local_path):
                self.folder_edit.setText(local_path)
                self._settings.setValue("last_folder", local_path)
                self._clear_stitch_results()
                self._append_log(f"已选择小图文件夹：{local_path}")
                self._update_auto_crop_button()
                self._update_detection_button()
                event.acceptProposedAction()
                return
            if os.path.isfile(local_path):
                self.load_image(local_path)
                event.acceptProposedAction()
                return

    def update_binary_image(self) -> None:
        if self._gray_image is None:
            return
        threshold = self.threshold_slider.value()
        _, binary_data = cv2.threshold(
            self._gray_image,
            threshold,
            255,
            cv2.THRESH_BINARY,
        )
        self.image_view.set_image(
            numpy_to_qimage(binary_data),
            clear_roi=False,
            keep_view=True,
        )
        self._mirror_roi_to_binary()

    def _update_gray_mean_display(self) -> None:
        gray_image = self._crop_gray if self._display_mode == "crop" else self._gray_image
        if gray_image is None or gray_image.size == 0:
            self.gray_mean_label.setText("灰度均值：暂无图像")
            return
        gray_mean = float(np.mean(gray_image))
        self.gray_mean_label.setText(f"灰度均值：{gray_mean:.2f}")

    def set_display_mode(self, mode: str, force: bool = False) -> None:
        if mode not in {"original", "binary", "parallel", "crop", "detection"}:
            return
        if not force and mode == self._display_mode:
            return

        previous = self._display_mode
        if mode == "detection":
            if self._detection_output is None:
                self._sync_display_mode_buttons(previous)
                return
            self._display_mode = "detection"
            self._update_recipe_overlay()
            self.original_view.setVisible(True)
            self.image_view.setVisible(False)
            self.original_view.set_roi_interactive(False)
            self.original_view.set_draw_mode(False)
            self.original_view.set_image(
                numpy_to_qimage(self._detection_output.image),
                clear_roi=True,
            )
            self.original_view.fit_image()
            self._sync_display_mode_buttons("detection")
            self.set_measure_mode(False)
            return
        if mode == "crop":
            if not self._prepare_crop_from_roi():
                self._set_status("裁剪图：请先在原图上按住 W 画裁剪框")
                self._sync_display_mode_buttons(previous)
                return
            self._first_pad = None
            self._pad_rects = []
            self._clear_pad_overlay()
            self._display_mode = "crop"
            self._update_recipe_overlay()
            self.original_view.setVisible(True)
            self.image_view.setVisible(False)
            self._show_crop_on_view(fit_view=True)
            self._update_gray_mean_display()
            self._sync_display_mode_buttons("crop")
            self.set_roi_draw_mode(False)
            self.set_measure_mode(self._measure_mode)
            self._set_status(
                f"裁剪图：{self._roi_rect[2]} × {self._roi_rect[3]} 像素"
                if self._roi_rect is not None
                else "裁剪图"
            )
            return

        leaving_crop = previous == "crop"
        self._display_mode = mode
        if leaving_crop:
            self._first_pad = None
            self._pad_rects = []
            self._clear_pad_overlay()
            self._restore_full_image_views(fit_view=False)

        show_original = mode in {"original", "parallel"}
        show_binary = mode in {"binary", "parallel"}
        self.original_view.set_roi_interactive(show_original)
        if show_original:
            if previous == "binary":
                self._restore_full_image_views(fit_view=False)
            elif self._original_image is not None:
                self.original_view.set_image(
                    numpy_to_qimage(self._original_image),
                    clear_roi=False,
                    keep_view=True,
                )
        if show_binary:
            self.update_binary_image()
            self.image_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.original_view.setVisible(show_original)
        self.image_view.setVisible(show_binary)
        self.original_view.set_draw_mode(False)
        self.image_view.set_draw_mode(False)
        if show_original:
            self.original_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        if show_binary:
            self.image_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._sync_display_mode_buttons(mode)
        self.fit_images()
        self._update_recipe_overlay()
        if mode == "parallel":
            self._sync_views_from(self.original_view)
            self._mirror_roi_to_binary()
        if mode in {"binary", "crop"}:
            self.set_roi_draw_mode(False)
        self.set_measure_mode(self._measure_mode)

    def _sync_views_from(self, source: ImageView) -> None:
        if self._display_mode != "parallel" or self._syncing_views:
            return
        target = self.image_view if source is self.original_view else self.original_view
        if not source.isVisible() or not target.isVisible():
            return
        self._syncing_views = True
        target.apply_transform_from(source)
        self._syncing_views = False

    def fit_images(self) -> None:
        if self.original_view.isVisible():
            self.original_view.fit_image()
        if self.image_view.isVisible():
            self.image_view.fit_image()
        if self._display_mode == "parallel":
            self._sync_views_from(self.original_view)

    def _on_roi_changed(self, x: int, y: int, width: int, height: int) -> None:
        if self._display_mode == "crop":
            return
        if width <= 0 or height <= 0:
            self._roi_rect = None
            self.roi_size_label.setText("裁剪框宽高：尚未选择裁剪区域")
            self.roi_position_label.setText("左上角坐标：尚未选择裁剪区域")
        else:
            self._roi_rect = (x, y, width, height)
            self.roi_size_label.setText(f"裁剪框宽高：{width} × {height} 像素")
            self.roi_position_label.setText(f"左上角坐标：{x}，{y} 像素")
        self._mirror_roi_to_binary()
        self._update_offset_display()

    def _on_measurement_changed(
        self,
        source: ImageView,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> None:
        if self._syncing_measurement:
            return
        if width <= 0 or height <= 0:
            self.measurement_label.setText("测量框宽高：尚未选择测量区域")
        else:
            self.measurement_label.setText(
                f"测量框宽高：{width} × {height} 像素，面积：{width * height} 像素²"
            )
        if self._display_mode != "parallel" or self._syncing_measurement:
            return
        target = self.image_view if source is self.original_view else self.original_view
        rect = source.measurement_rect()
        self._syncing_measurement = True
        try:
            target.set_measurement_rect(rect)
        finally:
            self._syncing_measurement = False

    def _update_offset_display(self) -> None:
        if self._original_image is None:
            self.roi_size_label.setText("裁剪框宽高：尚未选择裁剪区域")
            self.roi_position_label.setText("左上角坐标：尚未选择裁剪区域")
            self.horizontal_offset_label.setText("水平偏移：尚未选择裁剪区域")
            self.vertical_offset_label.setText("垂直偏移：尚未选择裁剪区域")
            return

        roi = self._capture_roi_rect()
        if roi is None:
            self.roi_size_label.setText("裁剪框宽高：尚未选择裁剪区域")
            self.roi_position_label.setText("左上角坐标：尚未选择裁剪区域")
            self.horizontal_offset_label.setText("水平偏移：尚未选择裁剪区域")
            self.vertical_offset_label.setText("垂直偏移：尚未选择裁剪区域")
            return

        x, y, width, height = roi
        self.roi_size_label.setText(f"裁剪框宽高：{width} × {height} 像素")
        self.roi_position_label.setText(f"左上角坐标：{x}，{y} 像素")

        image_height, image_width = self._original_image.shape[:2]
        center_x = x + width / 2.0
        center_y = y + height / 2.0
        pixel_size_um = self._settings.value(
            f"pixel_size_um_{self._resolution_mode}",
            1.43 if self._resolution_mode == "5um" else 2.5,
            type=float,
        )
        horizontal_mm = (image_width / 2.0 - center_x) * pixel_size_um / 1000.0
        vertical_mm = (center_y - image_height / 2.0) * pixel_size_um / 1000.0
        horizontal_direction = "往右" if horizontal_mm > 0 else "往左"
        vertical_direction = "往上" if vertical_mm > 0 else "往下"
        if abs(horizontal_mm) < 0.0005:
            self.horizontal_offset_label.setText("水平偏移：0.000 mm")
        else:
            self.horizontal_offset_label.setText(
                f"水平偏移：{horizontal_direction}偏移 {abs(horizontal_mm):.3f} mm"
            )
        if abs(vertical_mm) < 0.0005:
            self.vertical_offset_label.setText("垂直偏移：0.000 mm")
        else:
            self.vertical_offset_label.setText(
                f"垂直偏移：{vertical_direction}偏移 {abs(vertical_mm):.3f} mm"
            )

    def _mirror_roi_to_binary(self) -> None:
        if self._display_mode != "parallel":
            return
        rect = self.original_view.roi_rect()
        if rect is None or rect.width() < 1 or rect.height() < 1:
            if self.image_view.roi_rect() is not None:
                self.image_view.blockSignals(True)
                self.image_view.clear_roi()
                self.image_view.blockSignals(False)
            return
        self.image_view.set_roi_from_spin(
            int(round(rect.x())),
            int(round(rect.y())),
            int(round(rect.width())),
            int(round(rect.height())),
        )

    def crop_and_save(self) -> None:
        if self._original_image is None:
            QMessageBox.warning(self, "提示", "当前没有可裁剪的图像。")
            return
        roi = self._capture_roi_rect()
        if roi is None:
            QMessageBox.warning(self, "提示", "请先在原图上按住 W 画框。")
            return
        clamped = self._clamp_roi_rect(roi)
        if clamped is None:
            QMessageBox.warning(self, "提示", "裁剪区域无效。")
            return

        x0, y0, w, h = clamped
        cropped = self._original_image[y0:y0 + h, x0:x0 + w]
        if cropped.size == 0:
            QMessageBox.warning(self, "提示", "裁剪区域无效。")
            return

        remembered_parent = self._settings.value("last_crop_parent_dir", "", type=str)
        last_name = self._settings.value("last_crop_filename", "crop.jpg", type=str) or "crop.jpg"
        if remembered_parent and os.path.isdir(remembered_parent):
            default_path = str(Path(remembered_parent) / last_name)
        elif self.folder_edit.text() and os.path.isdir(self.folder_edit.text()):
            default_path = str(Path(self.folder_edit.text()).parent / last_name)
        else:
            default_path = str(Path.cwd() / last_name)

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存裁剪图",
            default_path,
            "JPEG (*.jpg);;PNG (*.png);;BMP (*.bmp)",
        )
        if not file_path:
            return
        if not imwrite_unicode(file_path, cropped):
            QMessageBox.critical(self, "保存失败", f"无法写入：\n{file_path}")
            return

        saved = Path(file_path)
        save_dir = saved.parent
        parent_dir = save_dir.parent if save_dir.parent != save_dir else save_dir
        self._settings.setValue("last_crop_parent_dir", str(parent_dir))
        self._settings.setValue("last_crop_filename", saved.name)
        self._append_log(f"已保存裁剪图：{file_path}")
        QMessageBox.information(self, "完成", f"已保存裁剪图：\n{file_path}")
