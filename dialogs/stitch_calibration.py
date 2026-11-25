from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QPointF, Qt, Signal
from PySide6.QtGui import (
    QColor, QDragEnterEvent, QPainter, QPen, QPixmap, QWheelEvent,QDropEvent
)
from PySide6.QtWidgets import (
    QAbstractSpinBox, QDialog, QFormLayout, QGraphicsItem,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene,
    QGraphicsView, QHBoxLayout, QLabel, QMessageBox, QPushButton, QMenu,
    QFileDialog,
    QSlider, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from core.image_utils import (
    IMAGE_EXTENSIONS, imread_unicode, natural_sort_key, numpy_to_qimage,
)
from core.stitching.params import JointParams, get_joint_params, save_machine_preset


class _ClearSpinSelectionFilter(QObject):
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.FocusOut and hasattr(watched, "deselect"):
            watched.deselect()  # type: ignore[attr-defined]
        return super().eventFilter(watched, event)


class _BlockArrowKeysFilter(QObject):
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.KeyPress and event.key() in (
            Qt.Key.Key_Left, Qt.Key.Key_Right,
            Qt.Key.Key_Up, Qt.Key.Key_Down,
        ):
            return True
        return super().eventFilter(watched, event)


class CalibrationPixmapItem(QGraphicsPixmapItem):
    def __init__(
        self, pixmap: QPixmap, view: "CalibrationCanvasView", movable: bool
    ) -> None:
        super().__init__(pixmap)
        self._view = view
        self._movable = movable
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, movable)
        self.setZValue(2 if movable else 1)

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: ANN001
        painter.drawPixmap(self.offset(), self.pixmap())

    def itemChange(self, change, value):  # noqa: ANN001, N802
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self._movable:
            return self._view.snap_position(self, value)
        return super().itemChange(change, value)

    def set_movable(self, movable: bool) -> None:
        self._movable = movable
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, movable)
        self.setZValue(2 if movable else 1)


class CalibrationCanvasView(QGraphicsView):
    selection_changed = Signal()
    position_changed = Signal()
    ZOOM_FACTOR = 1.2

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(Qt.GlobalColor.darkGray)
        self._canvas_width = 1
        self._canvas_height = 1
        self._canvas_item = QGraphicsRectItem()
        canvas_pen = QPen(QColor(59, 130, 246), 2)
        canvas_pen.setCosmetic(True)
        self._canvas_item.setPen(canvas_pen)
        self._canvas_item.setBrush(QColor(255, 255, 255))
        self._canvas_item.setZValue(0)
        self._scene.addItem(self._canvas_item)
        self._reference_item: CalibrationPixmapItem | None = None
        self._moving_item: CalibrationPixmapItem | None = None
        self._scene.selectionChanged.connect(self.selection_changed)

    def rebuild_canvas(self, width: int, height: int) -> None:
        self._canvas_width = width
        self._canvas_height = height
        if self._reference_item is not None:
            self._scene.removeItem(self._reference_item)
        if self._moving_item is not None:
            self._scene.removeItem(self._moving_item)
        self._canvas_item.setRect(0, 0, width, height)
        self._scene.setSceneRect(-width, -height, width * 3, height * 3)
        self._reference_item = None
        self._moving_item = None
        self.resetTransform()
        self.fitInView(self._canvas_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.selection_changed.emit()
        self.position_changed.emit()

    def snap_position(self, item: CalibrationPixmapItem, value: QPointF) -> QPointF:
        pos = QPointF(round(value.x()), round(value.y()))
        threshold = 10
        width = item.pixmap().width()
        height = item.pixmap().height()
        if abs(pos.x()) <= threshold:
            pos.setX(0)
        elif abs(pos.x() + width - self._canvas_width) <= threshold:
            pos.setX(self._canvas_width - width)
        if abs(pos.y()) <= threshold:
            pos.setY(0)
        elif abs(pos.y() + height - self._canvas_height) <= threshold:
            pos.setY(self._canvas_height - height)
        return pos

    def _add_image(self, file_path: str) -> None:
        image = imread_unicode(file_path)
        if image is None:
            QMessageBox.warning(self, "加载失败", f"无法读取图片：\n{file_path}")
            return
        pixmap = QPixmap.fromImage(numpy_to_qimage(image))
        if self._reference_item is None:
            self._reference_item = CalibrationPixmapItem(pixmap, self, False)
            self._reference_item.setPos(0, 0)
            self._scene.addItem(self._reference_item)
            self._reference_item.setSelected(True)
        else:
            if self._moving_item is not None:
                QMessageBox.information(self, "提示", "画布已有两张图片，请先删除一张图片。")
                return
            self._moving_item = CalibrationPixmapItem(pixmap, self, True)
            self._moving_item.setPos(0, 0)
            self._scene.addItem(self._moving_item)
            self._moving_item.setSelected(True)
        self.position_changed.emit()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:  # noqa: ANN001, N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in IMAGE_EXTENSIONS:
                self._add_image(url.toLocalFile())
        event.acceptProposedAction()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        factor = self.ZOOM_FACTOR if event.angleDelta().y() > 0 else 1 / self.ZOOM_FACTOR
        self.scale(factor, factor)
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: ANN001, N802
        self.setFocus()
        if event.button() == Qt.MouseButton.RightButton:
            item = self.itemAt(event.position().toPoint())
            if isinstance(item, CalibrationPixmapItem):
                menu = QMenu(self)
                delete_action = menu.addAction("删除图片")
                swap_action = None
                if self._reference_item is not None and self._moving_item is not None:
                    swap_action = menu.addAction("调换基准图与移动图")
                chosen = menu.exec(event.globalPosition().toPoint())
                if chosen is delete_action:
                    self.remove_image(item)
                elif swap_action is not None and chosen is swap_action:
                    self.swap_images()
                event.accept()
                return
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.position().toPoint())
            if isinstance(item, CalibrationPixmapItem):
                self._scene.clearSelection()
                item.setSelected(True)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001, N802
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self.position_changed.emit()

    def keyPressEvent(self, event) -> None:  # noqa: ANN001, N802
        item = self._moving_item
        if item is not None and item.isSelected():
            offsets = {
                Qt.Key.Key_Left: QPointF(-1, 0),
                Qt.Key.Key_Right: QPointF(1, 0),
                Qt.Key.Key_Up: QPointF(0, -1),
                Qt.Key.Key_Down: QPointF(0, 1),
            }
            offset = offsets.get(event.key())
            if offset is not None:
                item.setPos(item.pos() + offset)
                self.position_changed.emit()
                event.accept()
                return
        super().keyPressEvent(event)

    def selected_item(self) -> CalibrationPixmapItem | None:
        selected = self._scene.selectedItems()
        return selected[0] if selected and isinstance(selected[0], CalibrationPixmapItem) else None

    def remove_image(self, item: CalibrationPixmapItem) -> None:
        if item is self._reference_item:
            self._scene.removeItem(item)
            self._reference_item = self._moving_item
            self._moving_item = None
            if self._reference_item is not None:
                self._reference_item.set_movable(False)
        elif item is self._moving_item:
            self._scene.removeItem(item)
            self._moving_item = None
        else:
            return
        self.selection_changed.emit()
        self.position_changed.emit()

    def swap_images(self) -> None:
        if self._reference_item is None or self._moving_item is None:
            return
        reference_pos = self._reference_item.pos()
        self._reference_item, self._moving_item = self._moving_item, self._reference_item
        self._reference_item.set_movable(False)
        self._moving_item.set_movable(True)
        self._reference_item.setPos(0, 0)
        self._moving_item.setPos(reference_pos)
        self._moving_item.setSelected(True)
        self.position_changed.emit()

    def moving_position(self) -> QPointF:
        return self._moving_item.pos() if self._moving_item is not None else QPointF()

    def has_moving_image(self) -> bool:
        return self._moving_item is not None

    def has_images(self) -> bool:
        return self._reference_item is not None or self._moving_item is not None

    def clear_images(self) -> None:
        if self._reference_item is not None:
            self._scene.removeItem(self._reference_item)
        if self._moving_item is not None:
            self._scene.removeItem(self._moving_item)
        self._reference_item = None
        self._moving_item = None
        self.selection_changed.emit()
        self.position_changed.emit()

    def set_moving_position(self, x: int, y: int) -> None:
        if self._moving_item is not None:
            self._moving_item.setPos(x, y)


class StitchCalibrationDialog(QDialog):
    def __init__(
        self,
        mode: str,
        machine_id: str,
        initial_folder: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._mode = mode
        self._machine_id = machine_id
        self._folder = Path(initial_folder) if initial_folder else None
        params = get_joint_params(machine_id, mode)
        self._initial_positions = (
            (params.h_dx, params.h_dy),
            (params.v_dx, params.v_dy),
        )
        self._syncing = False
        self._created_width: int | None = None
        self._created_height: int | None = None
        self.setWindowTitle(f"确认拼接参数 - {machine_id}")
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowSystemMenuHint
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setAcceptDrops(True)
        self.resize(1080, 760)

        layout = QVBoxLayout(self)
        size_form = QFormLayout()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 100000)
        self.width_spin.setValue(params.single_w)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 100000)
        self.height_spin.setValue(params.single_h)
        self.create_canvas_button = QPushButton("创建画布")
        self.create_canvas_button.clicked.connect(self._rebuild_canvases)
        self.width_spin.valueChanged.connect(self._update_create_button)
        self.height_spin.valueChanged.connect(self._update_create_button)
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("宽"))
        size_row.addWidget(self._spin_control(self.width_spin))
        size_row.addWidget(QLabel("高"))
        size_row.addWidget(self._spin_control(self.height_spin))
        size_row.addWidget(QLabel("px"))
        size_row.addWidget(self.create_canvas_button)
        size_row.addStretch()
        size_form.addRow("单张图尺寸", self._layout_widget(size_row))
        layout.addLayout(size_form)

        folder_row = QHBoxLayout()
        self.folder_label = QLabel(str(self._folder) if self._folder else "未选择小图文件夹")
        self.folder_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.folder_label.setStyleSheet("color: #64748b;")
        import_folder_button = QPushButton("导入小图文件夹")
        import_folder_button.clicked.connect(self._select_image_folder)
        folder_row.addWidget(self.folder_label, 1)
        folder_row.addWidget(import_folder_button)
        layout.addLayout(folder_row)

        self.tabs = QTabWidget()
        self.horizontal_canvas, horizontal_page = self._create_page("hDx", "hDy")
        self.vertical_canvas, vertical_page = self._create_page("vDx", "vDy")
        self.tabs.addTab(horizontal_page, "横向校准")
        self.tabs.addTab(vertical_page, "纵向校准")
        self.tabs.setTabEnabled(0, params.cols > 1)
        self.tabs.setTabEnabled(1, params.rows > 1)
        if params.cols == 1 and params.rows > 1:
            self.tabs.setCurrentIndex(1)
        layout.addWidget(self.tabs, 1)

        hint = QLabel("拖入两张图片：第一张为基准图，第二张可拖动。选中第二张后可用方向键每次微调 1 px；靠近画布边框自动吸附。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #64748b;")
        layout.addWidget(hint)
        button_row = QHBoxLayout()
        button_row.addStretch()
        save_button = QPushButton("写入当前配方")
        save_button.clicked.connect(self._save_params)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.close)
        button_row.addWidget(save_button)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)
        self.setStyleSheet(
            "QPushButton { min-height: 34px; border-radius: 7px; }"
            "QSpinBox, QDoubleSpinBox { min-height: 34px; background: #f8fafc; color: #1f2937; }"
            "QPushButton { border: 1px solid #3b82f6; color: #2563eb; background: #eff6ff; padding: 0 12px; }"
            "QPushButton:hover { background: #dbeafe; }"
            "QPushButton:disabled { color: #94a3b8; border-color: #cbd5e1; background: #f1f5f9; }"
            "QPushButton#spinArrowButton { min-width: 28px; max-width: 28px; min-height: 16px; max-height: 16px; border-radius: 4px; padding: 0; font-size: 9px; font-weight: 600; }"
        )
        self._rebuild_canvases()
        if self._folder is not None and self._folder.is_dir():
            self._load_folder_images()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: ANN001, N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        canvas = self.horizontal_canvas if self.tabs.currentIndex() == 0 else self.vertical_canvas
        for url in event.mimeData().urls():
            if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in IMAGE_EXTENSIONS:
                canvas._add_image(url.toLocalFile())
        event.acceptProposedAction()

    def _select_image_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "选择小图文件夹",
            str(self._folder) if self._folder and self._folder.is_dir() else "",
        )
        if not folder:
            return
        self._folder = Path(folder)
        self.folder_label.setText(str(self._folder))
        self._load_folder_images()

    def _load_folder_images(self) -> None:
        if self._folder is None or not self._folder.is_dir():
            return
        paths = sorted(
            (
                path for path in self._folder.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=natural_sort_key,
        )
        params = get_joint_params(self._machine_id, self._mode)
        if params.layers == 1:
            image_indexes = (
                (0, 1) if params.cols > 1 else None,
                (0, 2 * params.cols - 1) if params.rows > 1 else None,
            )
        else:
            image_indexes = (
                (1, 2) if params.cols > 1 else None,
                (1, 4 * params.cols - 2) if params.rows > 1 else None,
            )
        required_count = max(
            (index for pair in image_indexes if pair is not None for index in pair),
            default=-1,
        ) + 1
        if len(paths) < required_count:
            QMessageBox.warning(
                self,
                "图片数量不足",
                f"当前分辨率需要至少 {required_count} 张小图，实际找到 {len(paths)} 张。",
            )
            return
        self.horizontal_canvas.clear_images()
        self.vertical_canvas.clear_images()
        for canvas, indexes, position in zip(
            (self.horizontal_canvas, self.vertical_canvas),
            image_indexes,
            self._initial_positions,
        ):
            if indexes is None:
                continue
            for index in indexes:
                canvas._add_image(str(paths[index]))
            canvas.set_moving_position(*position)

    @staticmethod
    def _layout_widget(layout: QHBoxLayout) -> QWidget:
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    @staticmethod
    def _spin_control(spin: QSpinBox) -> QWidget:
        spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin.lineEdit().installEventFilter(_ClearSpinSelectionFilter(spin.lineEdit()))
        arrows = QVBoxLayout()
        arrows.setContentsMargins(0, 0, 0, 0)
        arrows.setSpacing(2)
        up_button = QPushButton("▲")
        up_button.setObjectName("spinArrowButton")
        up_button.clicked.connect(lambda: StitchCalibrationDialog._step_spin(spin, 1))
        down_button = QPushButton("▼")
        down_button.setObjectName("spinArrowButton")
        down_button.clicked.connect(lambda: StitchCalibrationDialog._step_spin(spin, -1))
        arrows.addWidget(up_button)
        arrows.addWidget(down_button)
        control = QWidget()
        layout = QHBoxLayout(control)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(spin, 1)
        layout.addLayout(arrows)
        return control

    def _create_page(self, x_name: str, y_name: str) -> tuple[CalibrationCanvasView, QWidget]:
        page = QWidget()
        layout = QVBoxLayout(page)
        canvas = CalibrationCanvasView()
        layout.addWidget(canvas, 1)
        controls = QHBoxLayout()
        x_spin = QSpinBox()
        y_spin = QSpinBox()
        for spin in (x_spin, y_spin):
            spin.setRange(-100000, 100000)
            spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            spin.setReadOnly(True)
        opacity = QSlider(Qt.Orientation.Horizontal)
        opacity.setRange(10, 100)
        opacity.setValue(60)
        opacity.setFixedWidth(140)
        opacity_spin = QSpinBox()
        opacity_spin.setRange(10, 100)
        opacity_spin.setValue(60)
        opacity_spin.setSuffix(" %")
        opacity_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        opacity_spin.lineEdit().installEventFilter(
            _ClearSpinSelectionFilter(opacity_spin.lineEdit())
        )
        arrow_filter = _BlockArrowKeysFilter(opacity_spin)
        opacity_spin.installEventFilter(arrow_filter)
        opacity_spin.lineEdit().installEventFilter(arrow_filter)
        opacity.installEventFilter(arrow_filter)
        controls.addWidget(QLabel(f"{x_name}:"))
        controls.addWidget(x_spin)
        controls.addWidget(QLabel(f"{y_name}:"))
        controls.addWidget(y_spin)
        controls.addWidget(QLabel("选中图透明度："))
        controls.addWidget(opacity)
        controls.addWidget(opacity_spin)
        clear_button = QPushButton("清除图片")
        clear_button.clicked.connect(canvas.clear_images)
        controls.addWidget(clear_button)
        page.setProperty("x_spin", x_spin)
        page.setProperty("y_spin", y_spin)
        page.setProperty("opacity", opacity)
        page.setProperty("opacity_spin", opacity_spin)
        layout.addLayout(controls)
        canvas.position_changed.connect(lambda: self._sync_page(page, canvas))
        canvas.selection_changed.connect(lambda: self._sync_page(page, canvas))
        opacity.valueChanged.connect(lambda value: self._set_selected_opacity(canvas, value))
        opacity.valueChanged.connect(opacity_spin.setValue)
        opacity_spin.valueChanged.connect(opacity.setValue)
        return canvas, page

    @staticmethod
    def _step_spin(spin: QSpinBox, step: int) -> None:
        if step > 0:
            spin.stepUp()
        else:
            spin.stepDown()
        spin.lineEdit().deselect()

    def _rebuild_canvases(self) -> None:
        width = self.width_spin.value()
        height = self.height_spin.value()
        size_changed = (
            self._created_width is not None
            and (width != self._created_width or height != self._created_height)
        )
        if size_changed and (
            self.horizontal_canvas.has_images() or self.vertical_canvas.has_images()
        ):
            answer = QMessageBox.question(
                self,
                "确认重建画布",
                "修改画布尺寸会清除当前已放入的图片，是否继续？",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.horizontal_canvas.rebuild_canvas(width * 2, height)
        self.vertical_canvas.rebuild_canvas(width, height * 2)
        self._created_width = width
        self._created_height = height
        self._update_create_button()

    def _update_create_button(self, *_args: object) -> None:
        if self._created_width is None or self._created_height is None:
            self.create_canvas_button.setEnabled(False)
            return
        self.create_canvas_button.setEnabled(
            self.width_spin.value() != self._created_width
            or self.height_spin.value() != self._created_height
        )

    def _sync_page(self, page: QWidget, canvas: CalibrationCanvasView) -> None:
        if self._syncing:
            return
        self._syncing = True
        pos = canvas.moving_position()
        page.property("x_spin").setValue(round(pos.x()))
        page.property("y_spin").setValue(round(pos.y()))
        selected = canvas.selected_item()
        if selected is not None:
            page.property("opacity").blockSignals(True)
            page.property("opacity").setValue(round(selected.opacity() * 100))
            page.property("opacity").blockSignals(False)
            page.property("opacity_spin").blockSignals(True)
            page.property("opacity_spin").setValue(round(selected.opacity() * 100))
            page.property("opacity_spin").blockSignals(False)
        else:
            page.property("opacity").blockSignals(True)
            page.property("opacity").setValue(60)
            page.property("opacity").blockSignals(False)
            page.property("opacity_spin").blockSignals(True)
            page.property("opacity_spin").setValue(60)
            page.property("opacity_spin").blockSignals(False)
        self._syncing = False

    @staticmethod
    def _set_selected_opacity(canvas: CalibrationCanvasView, value: int) -> None:
        item = canvas.selected_item()
        if item is not None:
            item.setOpacity(value / 100.0)

    def _save_params(self) -> None:
        params = get_joint_params(self._machine_id, self._mode)
        h_pos = self.horizontal_canvas.moving_position()
        v_pos = self.vertical_canvas.moving_position()
        updated = JointParams(
            h_dx=round(h_pos.x()) if self.horizontal_canvas.has_moving_image() else params.h_dx,
            h_dy=round(h_pos.y()) if self.horizontal_canvas.has_moving_image() else params.h_dy,
            v_dx=round(v_pos.x()) if self.vertical_canvas.has_moving_image() else params.v_dx,
            v_dy=round(v_pos.y()) if self.vertical_canvas.has_moving_image() else params.v_dy,
            rows=params.rows, cols=params.cols, layers=params.layers,
            single_w=self.width_spin.value(), single_h=self.height_spin.value(),
            channel=params.channel,
        )
        answer = QMessageBox.question(
            self, "确认写入", f"确定将校准参数写入配方“{self._machine_id}”吗？"
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            save_machine_preset(self._mode, self._machine_id, updated, self._machine_id)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "写入失败", str(exc))
            return
        QMessageBox.information(self, "完成", "拼接参数已写入当前配方。")
