from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QSettings, Qt
from PySide6.QtWidgets import (
    QAbstractSpinBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QMessageBox, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

from core.stitching.params import (
    JointParams, delete_machine_preset, get_joint_params,
    get_machine_presets, save_machine_preset,
)

class RecipeManagerDialog(QDialog):
    def __init__(
        self, mode: str, parent: QWidget | None = None, selected_machine: str = ""
    ) -> None:
        super().__init__(parent)
        self._mode = mode
        self._settings = QSettings("ImageBinarizationTool", "ImageBinarizationTool")
        self._original_name: str | None = None
        self._saved_snapshot: tuple[object, ...] | None = None
        self._changing_selection = False
        self.setWindowTitle(f"{mode} 配方管理")
        self.resize(820, 460)

        root_layout = QHBoxLayout(self)
        left_layout = QVBoxLayout()
        pixel_size_form = QFormLayout()
        self.pixel_size_spin = QDoubleSpinBox()
        self.pixel_size_spin.setDecimals(4)
        self.pixel_size_spin.setRange(0.0001, 1000.0)
        self.pixel_size_spin.setSuffix(" μm/pix")
        self.pixel_size_spin.setValue(
            self._settings.value(
                f"pixel_size_um_{mode}",
                1.43 if mode == "5um" else 2.5,
                type=float,
            )
        )
        self.pixel_size_spin.installEventFilter(self)
        pixel_size_form.addRow("像素尺寸", self._spin_control(self.pixel_size_spin))
        left_layout.addLayout(pixel_size_form)
        self.recipe_list = QListWidget()
        self.recipe_list.setMinimumWidth(260)
        self.recipe_list.setStyleSheet(
            "QListWidget { border: 1px solid #dbe3ee; border-radius: 8px; "
            "background: #f8fafc; padding: 4px; }"
            "QListWidget::item { min-height: 32px; padding: 4px 10px; "
            "border-radius: 5px; }"
            "QListWidget::item:selected { background: #dbeafe; color: #1d4ed8; "
            "font-weight: 600; }"
        )
        self.recipe_list.currentItemChanged.connect(self._on_selection_changed)
        left_layout.addWidget(QLabel("配方列表"))
        left_layout.addWidget(self.recipe_list)
        refresh_button = QPushButton("刷新")
        refresh_button.clicked.connect(self.refresh)
        left_layout.addWidget(refresh_button)
        root_layout.addLayout(left_layout, 2)

        right_layout = QVBoxLayout()
        form = QFormLayout()
        self.name_edit = QLineEdit()
        form.addRow("配方名称", self.name_edit)
        self.param_spins: dict[str, QSpinBox] = {}
        fields = (
            ("h_dx", "hDx", -100000, 100000),
            ("h_dy", "hDy", -100000, 100000),
            ("v_dx", "vDx", -100000, 100000),
            ("v_dy", "vDy", -100000, 100000),
            ("rows", "rows", 1, 100),
            ("cols", "cols", 1, 100),
            ("layers", "layers", 1, 2),
            ("single_w", "singleW", 1, 100000),
            ("single_h", "singleH", 1, 100000),
            ("channel", "channel", 1, 4),
        )
        for key, label, minimum, maximum in fields:
            spin = QSpinBox()
            spin.setRange(minimum, maximum)
            spin.installEventFilter(self)
            form.addRow(label, self._spin_control(spin))
            self.param_spins[key] = spin
        right_layout.addLayout(form)

        button_row = QHBoxLayout()
        new_button = QPushButton("新增")
        new_button.clicked.connect(self._new_recipe)
        save_button = QPushButton("保存")
        save_button.clicked.connect(self._save_recipe)
        delete_button = QPushButton("删除")
        delete_button.clicked.connect(self._delete_recipe)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.close)
        for button in (new_button, save_button, delete_button, close_button):
            button_row.addWidget(button)
        right_layout.addLayout(button_row)
        root_layout.addLayout(right_layout, 3)
        self.setStyleSheet(
            "QLineEdit, QPushButton { min-height: 34px; border-radius: 7px; }"
            "QLineEdit { padding: 0 8px; border: 1px solid #cbd5e1; background: #f8fafc; }"
            "QSpinBox, QDoubleSpinBox { min-height: 34px; background: #f8fafc; color: #1f2937; }"
            "QPushButton { border: 1px solid #3b82f6; color: #2563eb; background: #eff6ff; padding: 0 12px; }"
            "QPushButton:hover { background: #dbeafe; }"
            "QPushButton#spinArrowButton { min-width: 28px; max-width: 28px; min-height: 16px; max-height: 16px; border-radius: 4px; padding: 0; font-size: 9px; font-weight: 600; }"
        )
        self.refresh(selected_machine)

    @staticmethod
    def _spin_control(spin: QSpinBox | QDoubleSpinBox) -> QWidget:
        spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        arrows = QVBoxLayout()
        arrows.setContentsMargins(0, 0, 0, 0)
        arrows.setSpacing(2)
        up_button = QPushButton("▲")
        up_button.setObjectName("spinArrowButton")
        up_button.clicked.connect(spin.stepUp)
        down_button = QPushButton("▼")
        down_button.setObjectName("spinArrowButton")
        down_button.clicked.connect(spin.stepDown)
        arrows.addWidget(up_button)
        arrows.addWidget(down_button)
        control = QWidget()
        layout = QHBoxLayout(control)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(spin, 1)
        layout.addLayout(arrows)
        return control

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if (
            watched is self.pixel_size_spin or watched in self.param_spins.values()
        ) and event.type() == QEvent.Type.Wheel:
            return True
        return super().eventFilter(watched, event)

    def _snapshot(self) -> tuple[object, ...]:
        return (
            self.pixel_size_spin.value(),
            self.name_edit.text(),
            *(spin.value() for spin in self.param_spins.values()),
        )

    def _restore_snapshot(self) -> None:
        if self._saved_snapshot is None:
            return
        self.pixel_size_spin.setValue(float(self._saved_snapshot[0]))
        self.name_edit.setText(str(self._saved_snapshot[1]))
        for spin, value in zip(self.param_spins.values(), self._saved_snapshot[2:]):
            spin.setValue(int(value))

    def _has_unsaved_changes(self) -> bool:
        return self._saved_snapshot is not None and self._snapshot() != self._saved_snapshot

    def _confirm_leave(self) -> bool:
        if not self._has_unsaved_changes():
            return True
        message_box = QMessageBox(self)
        message_box.setWindowTitle("未保存的改动")
        message_box.setText("当前配方有未保存的改动，是否保存？")
        save_button = message_box.addButton(
            "保存", QMessageBox.ButtonRole.AcceptRole
        )
        discard_button = message_box.addButton(
            "放弃", QMessageBox.ButtonRole.DestructiveRole
        )
        cancel_button = message_box.addButton(
            "取消", QMessageBox.ButtonRole.RejectRole
        )
        message_box.setDefaultButton(save_button)
        message_box.exec()
        clicked_button = message_box.clickedButton()
        if clicked_button is cancel_button:
            return False
        if clicked_button is save_button:
            return self._save_recipe()
        if clicked_button is discard_button:
            self._restore_snapshot()
            return True
        return False

    def refresh(self, selected: str = "") -> None:
        if not self._confirm_leave():
            return
        self._refresh_list(selected)

    def _refresh_list(self, selected: str = "") -> None:
        if not selected and self.recipe_list.currentItem():
            selected = self.recipe_list.currentItem().text()
        try:
            presets = get_machine_presets(self._mode)
        except FileNotFoundError:
            presets = {}
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "读取失败", str(exc))
            return
        self._changing_selection = True
        self.recipe_list.clear()
        self.recipe_list.addItems(presets.keys())
        matches = self.recipe_list.findItems(selected, Qt.MatchFlag.MatchExactly)
        if matches:
            self.recipe_list.setCurrentItem(matches[0])
        elif self.recipe_list.count():
            self.recipe_list.setCurrentRow(0)
        else:
            self._set_new_recipe()
        self._changing_selection = False
        current = self.recipe_list.currentItem()
        if current is not None:
            self._load_selected(current.text())

    def _new_recipe(self) -> None:
        if not self._confirm_leave():
            return
        self._set_new_recipe()

    def _set_new_recipe(self) -> None:
        self._changing_selection = True
        self.recipe_list.clearSelection()
        self._changing_selection = False
        self._original_name = None
        self.name_edit.clear()
        defaults = JointParams(0, 0, 0, 0)
        values = {
            "h_dx": defaults.h_dx,
            "h_dy": defaults.h_dy,
            "v_dx": defaults.v_dx,
            "v_dy": defaults.v_dy,
            "rows": 3 if self._mode == "5um" else defaults.rows,
            "cols": 3 if self._mode == "5um" else defaults.cols,
            "layers": 1 if self._mode == "5um" else defaults.layers,
            "single_w": defaults.single_w,
            "single_h": defaults.single_h,
            "channel": defaults.channel,
        }
        for key, value in values.items():
            self.param_spins[key].setValue(value)
        self._saved_snapshot = self._snapshot()

    def _on_selection_changed(self, current, previous) -> None:  # noqa: ANN001
        if self._changing_selection or current is None:
            return
        if not self._confirm_leave():
            self._changing_selection = True
            self.recipe_list.setCurrentItem(previous)
            self._changing_selection = False
            return
        self._load_selected(current.text())

    def _load_selected(self, machine_id: str) -> None:
        if not machine_id:
            return
        try:
            params = get_joint_params(machine_id, self._mode)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "读取失败", str(exc))
            return
        self._original_name = machine_id
        self.name_edit.setText(machine_id)
        for key, spin in self.param_spins.items():
            spin.setValue(getattr(params, key))
        self._saved_snapshot = self._snapshot()

    def _form_params(self) -> JointParams | None:
        values = {key: spin.value() for key, spin in self.param_spins.items()}
        if self._mode == "5um" and values["layers"] != 1:
            QMessageBox.warning(self, "参数无效", "5um 必须为 layers=1。")
            return None
        return JointParams(**values)

    def _save_recipe(self) -> bool:
        machine_id = self.name_edit.text().strip()
        if not machine_id:
            QMessageBox.warning(self, "提示", "配方名称不能为空。")
            return False
        params = self._form_params()
        if params is None:
            return False
        try:
            save_machine_preset(
                self._mode,
                machine_id,
                params,
                self._original_name,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "保存失败", str(exc))
            return False
        self._settings.setValue(
            f"pixel_size_um_{self._mode}", self.pixel_size_spin.value()
        )
        self._original_name = machine_id
        self._saved_snapshot = self._snapshot()
        self._refresh_list(machine_id)
        matches = self.recipe_list.findItems(machine_id, Qt.MatchFlag.MatchExactly)
        if matches:
            self.recipe_list.setCurrentItem(matches[0])
        return True

    def _delete_recipe(self) -> None:
        if not self._confirm_leave():
            return
        if self._original_name is None:
            QMessageBox.warning(self, "提示", "请先选择要删除的配方。")
            return
        answer = QMessageBox.question(
            self,
            "确认删除",
            f"确定删除配方“{self._original_name}”吗？",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            delete_machine_preset(self._mode, self._original_name)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "删除失败", str(exc))
            return
        self._original_name = None
        self._saved_snapshot = None
        self._refresh_list()

    def closeEvent(self, event) -> None:  # noqa: ANN001, N802
        if self._confirm_leave():
            event.accept()
        else:
            event.ignore()

    def reject(self) -> None:
        if self._confirm_leave():
            super().reject()
