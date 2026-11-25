from __future__ import annotations
import sys
from PySide6.QtCore import  QSharedMemory
from PySide6.QtWidgets import (QApplication,QMainWindow,QMessageBox)

from ui.main_window import MainWindow
def place_window(window: QMainWindow) -> None:
    screen = window.screen() or QApplication.primaryScreen()
    if screen is None:
        return
    available = screen.availableGeometry()
    max_w = int(available.width() * 0.9)
    max_h = int(available.height() * 0.9)
    width = min(window.width(), max_w)
    height = min(window.height(), max_h)
    window.resize(width, height)
    frame = window.frameGeometry()
    frame.moveCenter(available.center())
    x = max(available.left(), min(frame.x(), available.right() - frame.width()))
    y = max(available.top(), min(frame.y(), available.bottom() - frame.height()))
    window.move(x, y)


def main() -> int:
    app = QApplication(sys.argv)
    shared_memory = QSharedMemory("ImageBinarizationTool-SingleInstance")
    if not shared_memory.create(1):
        QMessageBox.information(None, "提示", "程序已在运行。")
        return 0

    window = MainWindow()
    window.show()
    place_window(window)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
