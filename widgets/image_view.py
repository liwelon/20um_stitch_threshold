from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPen, QPixmap, QTransform, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsLineItem, QGraphicsPixmapItem, QGraphicsRectItem,
    QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView, QWidget,
)

class ImageView(QGraphicsView):
    MIN_SCALE = 0.05
    MAX_SCALE = 50.0
    ZOOM_FACTOR = 1.2
    HANDLE_NAMES = ("tl", "tr", "bl", "br", "l", "r", "t", "b")

    roi_changed = Signal(int, int, int, int)
    recipe_region_selected = Signal(int)
    recipe_region_changed = Signal(int, int, int, int, int)
    measurement_changed = Signal(int, int, int, int)
    transform_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self.setScene(self._scene)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setBackgroundBrush(Qt.GlobalColor.darkGray)
        self.setMouseTracking(True)
        self._draw_mode = False
        self._drawing = False
        self._origin = QPointF()
        self._roi_item: QGraphicsRectItem | None = None
        self._handle_items: dict[str, QGraphicsRectItem] = {}
        self._updating_roi = False
        self._edit_mode: str | None = None  # draw / move / handle name
        self._press_scene = QPointF()
        self._press_rect = QRectF()
        self._roi_interactive = True
        self._measure_mode = False
        self._measuring = False
        self._measurement_item: QGraphicsRectItem | None = None
        self._measurement_handles: dict[str, QGraphicsRectItem] = {}
        self._measurement_edit: str | None = None
        self._block_transform_signal = False
        self._pad_rect_items: list[QGraphicsRectItem] = []
        self._pad_band_items: list[QGraphicsRectItem] = []
        self._pad_line_item: QGraphicsLineItem | None = None
        self._recipe_overlay_items: list[QGraphicsItem] = []
        self._recipe_rect_items: list[QGraphicsRectItem | None] = []
        self._recipe_label_items: list[QGraphicsSimpleTextItem | None] = []
        self._recipe_handles: dict[str, QGraphicsRectItem] = {}
        self._recipe_selected_index = -1
        self._recipe_edit: str | None = None
        self._recipe_press_scene = QPointF()
        self._recipe_press_rect = QRectF()
        self.horizontalScrollBar().valueChanged.connect(self._notify_transform)
        self.verticalScrollBar().valueChanged.connect(self._notify_transform)

    def set_roi_interactive(self, enabled: bool) -> None:
        self._roi_interactive = enabled
        if not enabled:
            self._draw_mode = False
            self._drawing = False
            self._edit_mode = None
            if self.dragMode() != QGraphicsView.DragMode.ScrollHandDrag:
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                self.unsetCursor()
        self._sync_handles()

    def set_measure_mode(self, enabled: bool) -> None:
        self._measure_mode = enabled
        if not enabled:
            if (
                not self._measuring and self._measurement_edit is None
                and not self._draw_mode and self._edit_mode is None
            ):
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                self.unsetCursor()
            return
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def clear_measurement(self) -> None:
        if self._measurement_item is not None:
            self._scene.removeItem(self._measurement_item)
            self._measurement_item = None
            for handle in self._measurement_handles.values():
                self._scene.removeItem(handle)
            self._measurement_handles.clear()
            self._measuring = False
            self._measurement_edit = None
            if not self._measure_mode and not self._draw_mode and self._edit_mode is None:
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                self.unsetCursor()
            self.measurement_changed.emit(0, 0, 0, 0)

    def measurement_rect(self) -> QRectF | None:
        if self._measurement_item is None:
            return None
        return self._measurement_item.rect().normalized()

    def set_measurement_rect(self, rect: QRectF | None) -> None:
        if rect is None:
            self.clear_measurement()
            return
        rect = self._clamp_rect(rect)
        if self._measurement_item is None:
            self._measurement_item = QGraphicsRectItem(rect)
            pen = QPen(QColor(234, 88, 12), 2)
            pen.setCosmetic(True)
            self._measurement_item.setPen(pen)
            self._measurement_item.setBrush(QColor(234, 88, 12, 35))
            self._measurement_item.setZValue(25)
            self._scene.addItem(self._measurement_item)
            for name in self.HANDLE_NAMES:
                handle = QGraphicsRectItem()
                handle_pen = QPen(QColor(255, 255, 255), 1)
                handle_pen.setCosmetic(True)
                handle.setPen(handle_pen)
                handle.setBrush(QColor(234, 88, 12))
                handle.setZValue(26)
                self._scene.addItem(handle)
                self._measurement_handles[name] = handle
        else:
            self._measurement_item.setRect(rect)
        self._sync_handles()

    def _emit_measurement(self) -> None:
        rect = self.measurement_rect()
        if rect is None or rect.width() <= 0 or rect.height() <= 0:
            return
        self.measurement_changed.emit(
            int(round(rect.x())),
            int(round(rect.y())),
            int(round(rect.width())),
            int(round(rect.height())),
        )

    def apply_transform_from(self, other: "ImageView") -> None:
        self._block_transform_signal = True
        self.setTransform(other.transform())
        self.horizontalScrollBar().setValue(other.horizontalScrollBar().value())
        self.verticalScrollBar().setValue(other.verticalScrollBar().value())
        self._sync_handles()
        self._block_transform_signal = False

    def _notify_transform(self, *_args: object) -> None:
        if not self._block_transform_signal:
            self.transform_changed.emit()

    def set_image(self, image: QImage, *, clear_roi: bool = True, keep_view: bool = False) -> None:
        saved_transform = None
        saved_h = 0
        saved_v = 0
        if keep_view:
            saved_transform = QTransform(self.transform())
            saved_h = self.horizontalScrollBar().value()
            saved_v = self.verticalScrollBar().value()
            self._block_transform_signal = True

        if clear_roi:
            self.clear_roi()
            self.clear_pad_overlay()
        self.clear_measurement()
        self._pixmap_item.setPixmap(QPixmap.fromImage(image))
        self._scene.setSceneRect(self._pixmap_item.boundingRect())

        if keep_view and saved_transform is not None:
            self.setTransform(saved_transform)
            self.horizontalScrollBar().setValue(saved_h)
            self.verticalScrollBar().setValue(saved_v)
            self._sync_handles()
            self._block_transform_signal = False

    def fit_image(self) -> None:
        if not self._pixmap_item.pixmap().isNull():
            self.resetTransform()
            self.fitInView(
                self._pixmap_item,
                Qt.AspectRatioMode.KeepAspectRatio,
            )
            self._sync_handles()
            self._notify_transform()

    def set_draw_mode(self, enabled: bool) -> None:
        if not self._roi_interactive:
            return
        self._draw_mode = enabled
        if enabled:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.CrossCursor)
            return
        if (
            not self._drawing and self._edit_mode is None
            and not self._measure_mode and not self._measuring
            and self._measurement_edit is None
        ):
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.unsetCursor()

    def clear_pad_overlay(self) -> None:
        for item in self._pad_rect_items:
            self._scene.removeItem(item)
        self._pad_rect_items.clear()
        for item in self._pad_band_items:
            self._scene.removeItem(item)
        self._pad_band_items.clear()
        if self._pad_line_item is not None:
            self._scene.removeItem(self._pad_line_item)
            self._pad_line_item = None

    def clear_recipe_overlay(self) -> None:
        for item in self._recipe_overlay_items:
            self._scene.removeItem(item)
        self._recipe_overlay_items.clear()
        self._recipe_rect_items.clear()
        self._recipe_label_items.clear()
        self._recipe_handles.clear()
        self._recipe_selected_index = -1
        self._recipe_edit = None

    def set_recipe_overlay(
        self,
        regions: list[tuple[str, tuple[int, int, int, int], str]],
        selected_index: int,
    ) -> None:
        self.clear_recipe_overlay()
        self._recipe_selected_index = selected_index
        for index, (name, (x, y, width, height), color_name) in enumerate(regions):
            if width <= 0 or height <= 0:
                self._recipe_rect_items.append(None)
                self._recipe_label_items.append(None)
                continue
            selected = index == selected_index
            color = QColor(255, 255, 255) if selected else QColor(color_name)
            rect_item = QGraphicsRectItem(QRectF(x, y, width, height))
            pen = QPen(color)
            pen.setCosmetic(True)
            pen.setWidth(4 if selected else 2)
            rect_item.setPen(pen)
            rect_item.setZValue(30 if selected else 25)
            rect_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            rect_item.setToolTip(name)
            self._scene.addItem(rect_item)
            self._recipe_overlay_items.append(rect_item)
            self._recipe_rect_items.append(rect_item)

            label_item = QGraphicsSimpleTextItem(name)
            label_item.setBrush(color)
            label_item.setPos(x, y)
            label_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            label_item.setZValue(31 if selected else 26)
            label_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self._scene.addItem(label_item)
            self._recipe_overlay_items.append(label_item)
            self._recipe_label_items.append(label_item)

            if selected:
                for handle_name in self.HANDLE_NAMES:
                    handle = QGraphicsRectItem()
                    handle.setPen(QPen(QColor(255, 255, 255), 1))
                    handle.setBrush(color)
                    handle.setZValue(32)
                    handle.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                    self._scene.addItem(handle)
                    self._recipe_overlay_items.append(handle)
                    self._recipe_handles[handle_name] = handle
        self._sync_handles()

    def _recipe_hit(self, scene_pos: QPointF) -> tuple[int, str] | None:
        if not self._recipe_rect_items:
            return None
        view_pos = self.mapFromScene(scene_pos)
        best_handle: str | None = None
        best_dist = 12.0
        for name, item in self._recipe_handles.items():
            center = self.mapFromScene(item.rect().center())
            dist = ((center.x() - view_pos.x()) ** 2 + (center.y() - view_pos.y()) ** 2) ** 0.5
            if dist <= best_dist:
                best_dist = dist
                best_handle = name
        if best_handle is not None:
            return self._recipe_selected_index, best_handle
        order = [self._recipe_selected_index] + [
            index for index in range(len(self._recipe_rect_items))
            if index != self._recipe_selected_index
        ]
        for index in order:
            if 0 <= index < len(self._recipe_rect_items):
                item = self._recipe_rect_items[index]
                if item is not None and item.rect().contains(scene_pos):
                    return index, "move"
        return None

    def _recipe_resized_rect(self, handle: str, pos: QPointF) -> QRectF:
        bounds = self._image_bounds()
        rect = QRectF(self._recipe_press_rect)
        x = max(bounds.left(), min(pos.x(), bounds.right()))
        y = max(bounds.top(), min(pos.y(), bounds.bottom()))
        if "l" in handle:
            rect.setLeft(min(x, rect.right() - 1))
        if "r" in handle:
            rect.setRight(max(x, rect.left() + 1))
        if "t" in handle:
            rect.setTop(min(y, rect.bottom() - 1))
        if "b" in handle:
            rect.setBottom(max(y, rect.top() + 1))
        return rect

    def set_pad_overlay(
        self,
        rects: list[tuple[int, int, int, int]] | tuple[int, int, int, int],
        line_y: float | None = None,
        *,
        highlight_index: int | None = 0,
        search_bands: list[tuple[int, int, int, int]] | None = None,
    ) -> None:
        """绘制 pad 矩形（蓝色粗线）；highlight_index 对应稍深高亮。

        line_y 非空时绘制整图宽白色水平线。
        search_bands 为搜索条带矩形列表 (x, y, w, h)。
        """
        self.clear_pad_overlay()
        if isinstance(rects, tuple) and len(rects) == 4 and isinstance(rects[0], int):
            pad_list = [rects]  # type: ignore[list-item]
        else:
            pad_list = list(rects)  # type: ignore[arg-type]

        band_color = QColor(20, 184, 166)
        for x, y, w, h in search_bands or []:
            band_item = QGraphicsRectItem(QRectF(x, y, w, h))
            band_pen = QPen(band_color)
            band_pen.setCosmetic(True)
            band_pen.setWidth(2)
            band_item.setPen(band_pen)
            band_item.setBrush(QColor(20, 184, 166, 35))
            band_item.setZValue(15)
            self._scene.addItem(band_item)
            self._pad_band_items.append(band_item)

        for idx, (x, y, w, h) in enumerate(pad_list):
            is_hi = highlight_index is not None and idx == highlight_index
            color = QColor(37, 99, 235) if is_hi else QColor(59, 130, 246)
            pad_item = QGraphicsRectItem(QRectF(x, y, w, h))
            pad_pen = QPen(color)
            pad_pen.setCosmetic(True)
            pad_pen.setWidth(4 if is_hi else 3)
            pad_item.setPen(pad_pen)
            pad_item.setBrush(QColor(color.red(), color.green(), color.blue(), 40))
            pad_item.setZValue(20)
            self._scene.addItem(pad_item)
            self._pad_rect_items.append(pad_item)

        if line_y is None:
            return
        pixmap = self._pixmap_item.pixmap()
        if pixmap.isNull():
            return
        width = float(pixmap.width())
        line_item = QGraphicsLineItem(0.0, float(line_y), width, float(line_y))
        line_pen = QPen(QColor(255, 255, 255))
        line_pen.setCosmetic(True)
        line_pen.setWidth(2)
        line_item.setPen(line_pen)
        line_item.setZValue(21)
        self._scene.addItem(line_item)
        self._pad_line_item = line_item

    def clear_roi(self) -> None:
        if self._roi_item is not None:
            self._scene.removeItem(self._roi_item)
            self._roi_item = None
        for item in self._handle_items.values():
            self._scene.removeItem(item)
        self._handle_items.clear()
        self._edit_mode = None
        self._drawing = False
        self.roi_changed.emit(0, 0, 0, 0)

    def roi_rect(self) -> QRectF | None:
        if self._roi_item is None:
            return None
        return self._roi_item.rect().normalized()

    def _image_bounds(self) -> QRectF:
        pixmap = self._pixmap_item.pixmap()
        return QRectF(0, 0, pixmap.width(), pixmap.height())

    def _clamp_rect(self, rect: QRectF) -> QRectF:
        bounds = self._image_bounds()
        rect = rect.normalized()
        width = max(1.0, min(rect.width(), bounds.width()))
        height = max(1.0, min(rect.height(), bounds.height()))
        x = max(bounds.left(), min(rect.x(), bounds.right() - width))
        y = max(bounds.top(), min(rect.y(), bounds.bottom() - height))
        return QRectF(x, y, width, height)

    def _ensure_roi_item(self, rect: QRectF) -> None:
        if self._roi_item is None:
            self._roi_item = QGraphicsRectItem(rect)
            pen = QPen(QColor(37, 99, 235), 2)
            pen.setCosmetic(True)
            self._roi_item.setPen(pen)
            self._roi_item.setBrush(QColor(37, 99, 235, 40))
            self._roi_item.setZValue(10)
            self._scene.addItem(self._roi_item)
            for name in self.HANDLE_NAMES:
                handle = QGraphicsRectItem()
                handle_pen = QPen(QColor(255, 255, 255), 1)
                handle_pen.setCosmetic(True)
                handle.setPen(handle_pen)
                handle.setBrush(QColor(37, 99, 235))
                handle.setZValue(11)
                self._scene.addItem(handle)
                self._handle_items[name] = handle
        else:
            self._roi_item.setRect(rect)
        self._sync_handles()

    def _handle_size_scene(self) -> float:
        scale = max(self.transform().m11(), 1e-6)
        return 10.0 / scale

    def _sync_handles(self) -> None:
        size = self._handle_size_scene()
        half = size / 2.0
        for item, handles, visible in (
            (self._roi_item, self._handle_items, self._roi_interactive),
            (self._measurement_item, self._measurement_handles, True),
            (
                self._recipe_rect_items[self._recipe_selected_index]
                if 0 <= self._recipe_selected_index < len(self._recipe_rect_items)
                else None,
                self._recipe_handles,
                True,
            ),
        ):
            if item is None:
                continue
            rect = item.rect().normalized()
            centers = {
                "tl": QPointF(rect.left(), rect.top()),
                "tr": QPointF(rect.right(), rect.top()),
                "bl": QPointF(rect.left(), rect.bottom()),
                "br": QPointF(rect.right(), rect.bottom()),
                "l": QPointF(rect.left(), rect.center().y()),
                "r": QPointF(rect.right(), rect.center().y()),
                "t": QPointF(rect.center().x(), rect.top()),
                "b": QPointF(rect.center().x(), rect.bottom()),
            }
            for name, center in centers.items():
                handle = handles[name]
                handle.setRect(QRectF(center.x() - half, center.y() - half, size, size))
                handle.setVisible(visible)

    def set_roi_from_spin(self, x: int, y: int, width: int, height: int) -> None:
        if self._pixmap_item.pixmap().isNull() or width <= 0 or height <= 0:
            return
        self._updating_roi = True
        rect = self._clamp_rect(QRectF(x, y, width, height))
        self._ensure_roi_item(rect)
        self._updating_roi = False
        self.viewport().update()

    def _emit_roi(self) -> None:
        rect = self.roi_rect()
        if rect is None or self._updating_roi:
            return
        self.roi_changed.emit(
            int(round(rect.x())),
            int(round(rect.y())),
            int(round(rect.width())),
            int(round(rect.height())),
        )

    def _hit_handle(self, scene_pos: QPointF, *, measurement: bool = False) -> str | None:
        if not measurement and not self._roi_interactive:
            return None
        view_pos = self.mapFromScene(scene_pos)
        best_name: str | None = None
        best_dist = 12.0
        handles = self._measurement_handles if measurement else self._handle_items
        for name, item in handles.items():
            center = self.mapFromScene(item.rect().center())
            dist = ((center.x() - view_pos.x()) ** 2 + (center.y() - view_pos.y()) ** 2) ** 0.5
            if dist <= best_dist:
                best_dist = dist
                best_name = name
        return best_name

    def _hit_roi_body(self, scene_pos: QPointF) -> bool:
        if not self._roi_interactive:
            return False
        rect = self.roi_rect()
        return rect is not None and rect.contains(scene_pos)

    def _cursor_for_handle(self, handle: str | None) -> Qt.CursorShape:
        mapping = {
            "tl": Qt.CursorShape.SizeFDiagCursor,
            "br": Qt.CursorShape.SizeFDiagCursor,
            "tr": Qt.CursorShape.SizeBDiagCursor,
            "bl": Qt.CursorShape.SizeBDiagCursor,
            "l": Qt.CursorShape.SizeHorCursor,
            "r": Qt.CursorShape.SizeHorCursor,
            "t": Qt.CursorShape.SizeVerCursor,
            "b": Qt.CursorShape.SizeVerCursor,
            "move": Qt.CursorShape.SizeAllCursor,
        }
        return mapping.get(handle or "", Qt.CursorShape.ArrowCursor)

    def _resized_rect(self, handle: str, pos: QPointF) -> QRectF:
        rect = QRectF(self._press_rect)
        if "l" in handle:
            rect.setLeft(pos.x())
        if "r" in handle:
            rect.setRight(pos.x())
        if "t" in handle:
            rect.setTop(pos.y())
        if "b" in handle:
            rect.setBottom(pos.y())
        return self._clamp_rect(rect)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._pixmap_item.pixmap().isNull():
            return
        factor = self.ZOOM_FACTOR if event.angleDelta().y() > 0 else 1 / self.ZOOM_FACTOR
        current_scale = self.transform().m11()
        next_scale = current_scale * factor
        if factor > 1 and current_scale < self.MIN_SCALE:
            allowed = next_scale <= self.MAX_SCALE
        else:
            allowed = self.MIN_SCALE <= next_scale <= self.MAX_SCALE
        if allowed:
            self.scale(factor, factor)
            self._sync_handles()
            self._notify_transform()
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if event.button() == Qt.MouseButton.RightButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            measurement_rect = self.measurement_rect()
            if measurement_rect is not None and (
                self._hit_handle(scene_pos, measurement=True) is not None
                or measurement_rect.contains(scene_pos)
            ):
                self.clear_measurement()
                event.accept()
                return
            if self._roi_interactive:
                scene_pos = self.mapToScene(event.position().toPoint())
                if self._hit_handle(scene_pos) is not None or self._hit_roi_body(scene_pos):
                    self.clear_roi()
                    event.accept()
                    return
            super().mousePressEvent(event)
            return

        if event.button() != Qt.MouseButton.LeftButton or self._pixmap_item.pixmap().isNull():
            super().mousePressEvent(event)
            return

        scene_pos = self.mapToScene(event.position().toPoint())

        if self._recipe_rect_items:
            hit = self._recipe_hit(scene_pos)
            if hit is not None:
                index, edit = hit
                if index != self._recipe_selected_index:
                    self.recipe_region_selected.emit(index)
                    hit = self._recipe_hit(scene_pos)
                    if hit is None:
                        event.accept()
                        return
                    index, edit = hit
                item = self._recipe_rect_items[index]
                if item is not None:
                    self._recipe_edit = edit
                    self._recipe_press_scene = scene_pos
                    self._recipe_press_rect = item.rect().normalized()
                    self.setDragMode(QGraphicsView.DragMode.NoDrag)
                    self.setCursor(self._cursor_for_handle(edit))
                    event.accept()
                    return

        if self._measure_mode:
            self._measuring = True
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self._origin = scene_pos
            self.set_measurement_rect(QRectF(scene_pos, scene_pos))
            event.accept()
            return

        if self._roi_interactive and self._draw_mode:
            self._drawing = True
            self._edit_mode = "draw"
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self._origin = scene_pos
            self._ensure_roi_item(QRectF(self._origin, self._origin))
            event.accept()
            return

        measurement_rect = self.measurement_rect()
        measurement_handle = self._hit_handle(scene_pos, measurement=True)
        if measurement_rect is not None and (
            measurement_handle is not None or measurement_rect.contains(scene_pos)
        ):
            self._measurement_edit = measurement_handle or "move"
            self._press_scene = scene_pos
            self._press_rect = measurement_rect
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(self._cursor_for_handle(self._measurement_edit))
            event.accept()
            return

        handle = self._hit_handle(scene_pos)
        if handle is not None and self._roi_item is not None:
            self._edit_mode = handle
            self._press_scene = scene_pos
            self._press_rect = self._roi_item.rect().normalized()
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(self._cursor_for_handle(handle))
            event.accept()
            return

        if self._hit_roi_body(scene_pos) and self._roi_item is not None:
            self._edit_mode = "move"
            self._press_scene = scene_pos
            self._press_rect = self._roi_item.rect().normalized()
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.SizeAllCursor)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        scene_pos = self.mapToScene(event.position().toPoint())

        if self._recipe_edit is not None:
            index = self._recipe_selected_index
            item = self._recipe_rect_items[index]
            if item is not None:
                if self._recipe_edit == "move":
                    rect = self._clamp_rect(
                        self._recipe_press_rect.translated(scene_pos - self._recipe_press_scene)
                    )
                else:
                    rect = self._recipe_resized_rect(self._recipe_edit, scene_pos)
                item.setRect(rect)
                label = self._recipe_label_items[index]
                if label is not None:
                    label.setPos(rect.topLeft())
                self._sync_handles()
            event.accept()
            return

        if self._recipe_rect_items:
            hit = self._recipe_hit(scene_pos)
            if hit is not None:
                self.setCursor(self._cursor_for_handle(hit[1]))
                event.accept()
                return
            if self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag:
                self.unsetCursor()

        if self._measuring and self._measurement_item is not None:
            self.set_measurement_rect(QRectF(self._origin, scene_pos))
            self._emit_measurement()
            event.accept()
            return

        if self._measurement_edit is not None and self._measurement_item is not None:
            if self._measurement_edit == "move":
                rect = self._press_rect.translated(scene_pos - self._press_scene)
            else:
                rect = self._resized_rect(self._measurement_edit, scene_pos)
            self.set_measurement_rect(rect)
            self._emit_measurement()
            event.accept()
            return

        if self._drawing and self._edit_mode == "draw" and self._roi_item is not None:
            rect = self._clamp_rect(QRectF(self._origin, scene_pos))
            self._roi_item.setRect(rect)
            self._sync_handles()
            self._emit_roi()
            event.accept()
            return

        if self._edit_mode == "move" and self._roi_item is not None:
            delta = scene_pos - self._press_scene
            moved = self._press_rect.translated(delta)
            rect = self._clamp_rect(moved)
            self._roi_item.setRect(rect)
            self._sync_handles()
            self._emit_roi()
            event.accept()
            return

        if self._edit_mode in self.HANDLE_NAMES and self._roi_item is not None:
            rect = self._resized_rect(self._edit_mode, scene_pos)
            self._roi_item.setRect(rect)
            self._sync_handles()
            self._emit_roi()
            event.accept()
            return

        if not self._measure_mode and not self._draw_mode and self._edit_mode is None:
            measurement_rect = self.measurement_rect()
            measurement_handle = self._hit_handle(scene_pos, measurement=True)
            if measurement_handle is not None:
                self.setCursor(self._cursor_for_handle(measurement_handle))
                super().mouseMoveEvent(event)
                return
            if measurement_rect is not None and measurement_rect.contains(scene_pos):
                self.setCursor(Qt.CursorShape.SizeAllCursor)
                super().mouseMoveEvent(event)
                return
            if self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag:
                self.unsetCursor()

        if self._roi_interactive and not self._draw_mode and not self._measure_mode and self._edit_mode is None:
            handle = self._hit_handle(scene_pos)
            if handle is not None:
                self.setCursor(self._cursor_for_handle(handle))
            elif self._hit_roi_body(scene_pos):
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            elif self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag:
                self.unsetCursor()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        if event.button() == Qt.MouseButton.LeftButton and self._recipe_edit is not None:
            index = self._recipe_selected_index
            item = self._recipe_rect_items[index]
            self._recipe_edit = None
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.unsetCursor()
            if item is not None:
                rect = item.rect().normalized()
                self.recipe_region_changed.emit(
                    index, int(round(rect.x())), int(round(rect.y())),
                    int(round(rect.width())), int(round(rect.height())),
                )
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and (
            self._measuring or self._measurement_edit is not None
        ):
            self._measuring = False
            self._measurement_edit = None
            self._emit_measurement()
            if not self._measure_mode and not self._draw_mode:
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                self.unsetCursor()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and (
            self._drawing or self._edit_mode is not None
        ):
            self._drawing = False
            self._edit_mode = None
            self._emit_roi()
            if not self._draw_mode:
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                self.unsetCursor()
            self._notify_transform()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._notify_transform()
        super().mouseReleaseEvent(event)

