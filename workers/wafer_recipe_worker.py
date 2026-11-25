from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from core.wafer_recipe_db import DatabaseConfig, load_wafer_recipes


class WaferRecipeWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, config: DatabaseConfig, resolution_mode: str) -> None:
        super().__init__()
        self._config = config
        self._resolution_mode = resolution_mode

    def run(self) -> None:
        try:
            self.finished.emit(
                load_wafer_recipes(self._config, self._resolution_mode)
            )
        except Exception as exc:  # noqa: BLE001 - 传到 UI 提示
            self.failed.emit(str(exc))
