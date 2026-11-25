# -*- coding: utf-8 -*-
"""芯片 Pad 配方管理（增删改查，对齐机台配方管理交互）。"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.pad_recipes import (
    PadParams,
    default_pad_params,
    delete_pad_recipe,
    get_pad_params,
    list_chip_ids,
    load_pad_recipes,
    save_pad_recipe,
)


class PadRecipeManagerDialog(QDialog):
    def __init__(
        self,
        resolution_mode: str,
        parent: QWidget | None = None,
        selected_chip: str = "",
    ) -> None:
        super().__init__(parent)
        self._resolution_mode = resolution_mode
        self._original_name: str | None = None
        self._saved_snapshot: tuple[object, ...] | None = None
        self._changing_selection = False
        self.setWindowTitle(f"{resolution_mode} 芯片 Pad 配方管理")
        self.resize(760, 460)

        root_layout = QHBoxLayout(self)
        left_layout = QVBoxLayout()
        self.recipe_list = QListWidget()
        self.recipe_list.setMinimumWidth(220)
        self.recipe_list.setStyleSheet(
            "QListWidget { border: 1px solid #dbe3ee; border-radius: 8px; "
            "background: #f8fafc; padding: 4px; }"
            "QListWidget::item { min-height: 32px; padding: 4px 10px; "
            "border-radius: 5px; }"
            "QListWidget::item:selected { background: #dbeafe; color: #1d4ed8; "
            "font-weight: 600; }"
        )
        self.recipe_list.currentItemChanged.connect(self._on_selection_changed)
        left_layout.addWidget(QLabel("芯片配方列表"))
        left_layout.addWidget(self.recipe_list)
        refresh_button = QPushButton("刷新")
        refresh_button.clicked.connect(self.refresh)
        left_layout.addWidget(refresh_button)
        root_layout.addLayout(left_layout, 2)

        right_layout = QVBoxLayout()
        tip = QLabel(
            "对应 wafer_cpp 配方 pad 参数；含 padGrayThresh。\n"
            "主界面阈值可随时微调，点「写入当前配方」保存。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #64748b;")
        right_layout.addWidget(tip)

        form = QFormLayout()
        self.name_edit = QLineEdit()
        form.addRow("配方名称", self.name_edit)
        self.param_spins: dict[str, QSpinBox] = {}
        fields = (
            ("pad_w", "padW（期望宽度，px）", 1, 500),
            ("pad_h", "padH（期望高度，px）", 1, 500),
            ("pad_range", "padRange（尺寸容差）", 0, 200),
            ("pad_search_range", "padSearchRange（搜索条带）", 10, 2000),
            ("pad_min_num", "padMinNum（列内最少点数）", 1, 100),
            (
                "pad_top_to_cmos_edge_pix",
                "padTopToCmosEdgePix（到 CMOS 上边界）",
                0,
                100000,
            ),
            (
                "pad_gray_thresh",
                "padGrayThresh（二值化阈值）",
                0,
                255,
            ),
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
            "QSpinBox { min-height: 34px; background: #f8fafc; color: #1f2937; }"
            "QPushButton { border: 1px solid #3b82f6; color: #2563eb; background: #eff6ff; padding: 0 12px; }"
            "QPushButton:hover { background: #dbeafe; }"
            "QPushButton#spinArrowButton { min-width: 28px; max-width: 28px; "
            "min-height: 16px; max-height: 16px; border-radius: 4px; padding: 0; "
            "font-size: 9px; font-weight: 600; }"
        )
        self.refresh(selected_chip)

    @staticmethod
    def _spin_control(spin: QSpinBox) -> QWidget:
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
        if watched in self.param_spins.values() and event.type() == QEvent.Type.Wheel:
            return True
        return super().eventFilter(watched, event)

    def _snapshot(self) -> tuple[object, ...]:
        return (
            self.name_edit.text(),
            *(spin.value() for spin in self.param_spins.values()),
        )

    def _restore_snapshot(self) -> None:
        if self._saved_snapshot is None:
            return
        self.name_edit.setText(str(self._saved_snapshot[0]))
        for spin, value in zip(self.param_spins.values(), self._saved_snapshot[1:]):
            spin.setValue(int(value))

    def _has_unsaved_changes(self) -> bool:
        return self._saved_snapshot is not None and self._snapshot() != self._saved_snapshot

    def _confirm_leave(self) -> bool:
        if not self._has_unsaved_changes():
            return True
        message_box = QMessageBox(self)
        message_box.setWindowTitle("未保存的改动")
        message_box.setText("当前芯片配方有未保存的改动，是否保存？")
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
        clicked = message_box.clickedButton()
        if clicked is cancel_button:
            return False
        if clicked is save_button:
            return self._save_recipe()
        if clicked is discard_button:
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
            load_pad_recipes(self._resolution_mode, force=True)
            chip_ids = list_chip_ids(self._resolution_mode)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "读取失败", str(exc))
            return
        self._changing_selection = True
        self.recipe_list.clear()
        self.recipe_list.addItems(chip_ids)
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
        defaults = default_pad_params(self._resolution_mode)
        for key, spin in self.param_spins.items():
            spin.setValue(getattr(defaults, key))
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

    def _load_selected(self, chip_id: str) -> None:
        if not chip_id:
            return
        try:
            params = get_pad_params(chip_id, self._resolution_mode)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "读取失败", str(exc))
            return
        self._original_name = chip_id
        self.name_edit.setText(chip_id)
        for key, spin in self.param_spins.items():
            spin.setValue(getattr(params, key))
        self._saved_snapshot = self._snapshot()

    def _form_params(self) -> PadParams:
        return PadParams(
            **{key: spin.value() for key, spin in self.param_spins.items()}
        )

    def _save_recipe(self) -> bool:
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请填写配方名称。")
            return False
        params = self._form_params()
        try:
            save_pad_recipe(
                name,
                params,
                self._resolution_mode,
                self._original_name,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "保存失败", str(exc))
            return False
        self._original_name = name
        self._saved_snapshot = self._snapshot()
        self._refresh_list(name)
        return True

    def _delete_recipe(self) -> None:
        name = self._original_name or (
            self.recipe_list.currentItem().text()
            if self.recipe_list.currentItem()
            else ""
        )
        if not name:
            QMessageBox.warning(self, "提示", "请先选择要删除的配方。")
            return
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定删除芯片配方「{name}」？",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            delete_pad_recipe(name, self._resolution_mode)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "删除失败", str(exc))
            return
        self._saved_snapshot = None
        self._refresh_list()

    def selected_chip_id(self) -> str:
        if self._original_name:
            return self._original_name
        item = self.recipe_list.currentItem()
        return item.text() if item is not None else ""
