from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Callable

import cv2
import numpy as np

from core.first_pad import estimate_pad_rotation, find_pad_column, find_first_pad
from core.image_utils import imread_unicode
from core.tensorrt_inference import BlindCircuitTensorRt
from core.window_segment import segment_window

Rect = tuple[int, int, int, int]
LogCallback = Callable[[str], None]
CancelCallback = Callable[[], bool]
MAX_CORE_AREA = 10000
EXPAND = 6
MERGE_DIST = 10
DEFAULT_BLIND_CIRCUIT_MODEL_PATH = r"D:\WaferStudio\model\circuit_fp16.engine"
_20UM_PT_MODEL_CACHE: tuple[Path, int, Any] | None = None
_20UM_PT_MODEL_LOCK = Lock()


class DetectionCancelled(RuntimeError):
    pass


def _check_cancelled(callback: CancelCallback | None) -> None:
    if callback is not None and callback():
        raise DetectionCancelled("检测已停止")


@dataclass(frozen=True)
class DetectionItem:
    label: str
    bbox: Rect
    area: float
    algorithm: str
    confidence: float | None = None
    darkness: float | None = None


@dataclass
class DetectionOutput:
    image: np.ndarray
    items: list[DetectionItem] = field(default_factory=list)
    regions: list[tuple[str, Rect]] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    source_image: np.ndarray | None = None
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class AutoCropOutput:
    image: np.ndarray
    angle: float = 0.0
    messages: list[str] = field(default_factory=list)
    crop_rect: Rect | None = None
    cap_image: np.ndarray | None = None
    cmos_image: np.ndarray | None = None
    cap_box: Rect | None = None
    pad_centers: tuple[int, int] | None = None
    up_image: np.ndarray | None = None
    down_image: np.ndarray | None = None
    cap_mask: np.ndarray | None = None
    cap_corners: np.ndarray | None = None


@dataclass(frozen=True)
class _YoloDetection:
    class_id: int
    confidence: float
    bbox: Rect


def _log(messages: list[str], callback: LogCallback | None, message: str) -> None:
    messages.append(message)
    if callback is not None:
        callback(message)


def _section(params: dict[str, Any], *names: str) -> dict[str, Any]:
    for name in names:
        value = params.get(name)
        if isinstance(value, dict):
            return value
    return {}


def _rect(value: Any) -> Rect:
    if isinstance(value, dict):
        x1 = int(value.get("x1", value.get("x", 0)))
        y1 = int(value.get("y1", value.get("y", 0)))
        if "x2" in value or "y2" in value:
            return x1, y1, int(value.get("x2", x1)) - x1, int(value.get("y2", y1)) - y1
        return x1, y1, int(value.get("w", 0)), int(value.get("h", 0))
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        x1, y1, x2, y2 = (int(v) for v in value[:4])
        return x1, y1, x2 - x1, y2 - y1
    return 0, 0, 0, 0


def _clamp(rect: Rect, shape: tuple[int, ...]) -> Rect:
    x, y, w, h = rect
    height, width = shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + w)
    y1 = min(height, y + h)
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def five_um_blind_circuit_regions(
    shape: tuple[int, ...], detection: dict[str, Any]
) -> list[tuple[str, Rect]]:
    coords = detection.get("regionCoords", [])
    parsed = [_rect(value) for value in coords] if isinstance(coords, list) else []
    main = _clamp(
        parsed[3] if len(parsed) > 3 else (0, 0, shape[1], shape[0]), shape
    )
    extended = _clamp(parsed[4] if len(parsed) > 4 else main, shape)
    main_expand = int(detection.get("mainExpand", 10))
    fallback_blind = (
        extended[0] + main_expand,
        extended[1] + main_expand,
        extended[2] - main_expand * 2,
        main[1] - extended[1] - main_expand * 2,
    )
    fallback_circuit = (
        extended[0] + main_expand,
        main[1] + main[3] + main_expand,
        extended[2] - main_expand * 2,
        extended[1] + extended[3] - main[1] - main[3] - main_expand * 2,
    )
    regions: list[tuple[str, Rect]] = []
    for name, list_key, old_key, fallback in (
        ("盲元区", "blindDetectionZones", "blindDetectionZone", fallback_blind),
        ("电路区", "circuitDetectionZones", "circuitDetectionZone", fallback_circuit),
    ):
        configured_zones = detection.get(list_key)
        if isinstance(configured_zones, list):
            rois = [_rect(value) for value in configured_zones]
        else:
            rois = [_rect(detection.get(old_key, {}))]
        valid_rois = [roi for roi in rois if roi[2] > 0 and roi[3] > 0]
        for roi in valid_rois or [fallback]:
            clamped = _clamp(roi, shape)
            if clamped[2] > 0 and clamped[3] > 0:
                regions.append((name, clamped))
    return regions


def five_um_tile_starts(length: int) -> list[int]:
    patch = 512
    stride = 384
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def _intersection(a: Rect, b: Rect) -> Rect:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def _union(a: Rect, b: Rect) -> Rect:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = min(ax, bx), min(ay, by)
    x1, y1 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
    return x0, y0, x1 - x0, y1 - y0


def _near(a: Rect, b: Rect, distance: int = MERGE_DIST) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = max(ax - (bx + bw), bx - (ax + aw), 0)
    dy = max(ay - (by + bh), by - (ay + ah), 0)
    return max(dx, dy) <= distance


def _merge_boxes(boxes: list[Rect], distance: int = MERGE_DIST) -> list[Rect]:
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if _near(boxes[i], boxes[j], distance):
                    boxes[i] = _union(boxes[i], boxes[j])
                    boxes.pop(j)
                    changed = True
                    break
            if changed:
                break
    return boxes


def _connected_items(
    gray: np.ndarray,
    roi: Rect,
    threshold: int,
    min_area: int,
    label: str,
    algorithm: str,
    excludes: list[Rect],
) -> list[DetectionItem]:
    safe = _clamp(roi, gray.shape)
    x, y, w, h = safe
    if w <= 0 or h <= 0:
        return []
    _, binary = cv2.threshold(gray[y:y + h, x:x + w], threshold, 255, cv2.THRESH_BINARY_INV)
    for excluded in excludes:
        ex, ey, ew, eh = _intersection(excluded, safe)
        if ew > 0 and eh > 0:
            binary[ey - y:ey - y + eh, ex - x:ex - x + ew] = 0
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)
    boxes: list[Rect] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_area or area > MAX_CORE_AREA:
            continue
        bx = x + int(stats[index, cv2.CC_STAT_LEFT]) - EXPAND
        by = y + int(stats[index, cv2.CC_STAT_TOP]) - EXPAND
        bw = int(stats[index, cv2.CC_STAT_WIDTH]) + EXPAND * 2
        bh = int(stats[index, cv2.CC_STAT_HEIGHT]) + EXPAND * 2
        boxes.append(_clamp((bx, by, bw, bh), gray.shape))
    return [DetectionItem(label, box, float(box[2] * box[3]), algorithm) for box in _merge_boxes(boxes)]


def _segmented_5um_items(
    image: np.ndarray,
    roi: Rect,
    net: cv2.dnn.Net | BlindCircuitTensorRt,
    probability_threshold: float,
    min_area: int,
    algorithm: str,
    excludes: list[Rect],
) -> list[DetectionItem]:
    safe = _clamp(roi, image.shape)
    x, y, w, h = safe
    if w <= 0 or h <= 0:
        return []
    tile_size = 512
    source = image[y:y + h, x:x + w]
    rgb = (
        cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
        if source.ndim == 2
        else cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    )
    rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
    rotated_h, rotated_w = rgb.shape[:2]
    padded = cv2.copyMakeBorder(
        rgb, 0, max(0, tile_size - rotated_h), 0, max(0, tile_size - rotated_w),
        cv2.BORDER_REFLECT_101,
    )
    padded_h, padded_w = padded.shape[:2]
    y_positions = five_um_tile_starts(padded_h)
    x_positions = five_um_tile_starts(padded_w)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    axis = np.maximum(np.hanning(tile_size), 0.05).astype(np.float32)
    window = axis[:, None] * axis[None, :]
    accumulated = np.zeros((padded_h, padded_w), dtype=np.float32)
    weights = np.zeros((padded_h, padded_w), dtype=np.float32)
    for tile_y in y_positions:
        for tile_x in x_positions:
            tile = padded[
                tile_y:tile_y + tile_size, tile_x:tile_x + tile_size
            ]
            blob = np.ascontiguousarray(
                ((tile.astype(np.float32) / np.float32(255) - mean) / std)
                .transpose(2, 0, 1)[np.newaxis]
            )
            if isinstance(net, BlindCircuitTensorRt):
                probability = net.infer(blob)
            else:
                net.setInput(blob)
                probability = np.asarray(net.forward("probability"))
            if probability.shape != (1, 1, tile_size, tile_size):
                raise ValueError(
                    f"盲元电路区模型输出格式不匹配：{probability.shape}"
                )
            accumulated[
                tile_y:tile_y + tile_size, tile_x:tile_x + tile_size
            ] += probability[0, 0] * window
            weights[
                tile_y:tile_y + tile_size, tile_x:tile_x + tile_size
            ] += window
    probability_map = (accumulated / weights)[:rotated_h, :rotated_w]
    if not np.isfinite(probability_map).all():
        raise ValueError("盲元电路区模型返回非有限概率值。")
    probability_map = cv2.rotate(probability_map, cv2.ROTATE_90_COUNTERCLOCKWISE)
    binary = (probability_map >= probability_threshold).astype(np.uint8) * 255
    for excluded in excludes:
        ex, ey, ew, eh = _intersection(excluded, safe)
        if ew > 0 and eh > 0:
            binary[ey - y:ey - y + eh, ex - x:ex - x + ew] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)
    boxes: list[Rect] = []
    confidences: list[float] = []
    pixel_areas: list[int] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        left = int(stats[index, cv2.CC_STAT_LEFT])
        top = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        component = labels[top:top + height, left:left + width] == index
        confidences.append(float(np.mean(
            probability_map[top:top + height, left:left + width][component],
            dtype=np.float64,
        )))
        pixel_areas.append(area)
        bx = x + int(stats[index, cv2.CC_STAT_LEFT]) - EXPAND
        by = y + int(stats[index, cv2.CC_STAT_TOP]) - EXPAND
        bw = int(stats[index, cv2.CC_STAT_WIDTH]) + EXPAND * 2
        bh = int(stats[index, cv2.CC_STAT_HEIGHT]) + EXPAND * 2
        boxes.append(_clamp((bx, by, bw, bh), image.shape))
    items: list[DetectionItem] = []
    for merged in _merge_boxes(boxes):
        members = [
            index for index, box in enumerate(boxes)
            if _intersection(box, merged) == box
        ]
        confidence = sum(confidences[index] * pixel_areas[index] for index in members) / sum(
            pixel_areas[index] for index in members
        )
        items.append(DetectionItem(
            "other", merged, float(merged[2] * merged[3]), algorithm, confidence,
        ))
    return items


def _draw_output(image: np.ndarray, items: list[DetectionItem], regions: list[tuple[str, Rect]]) -> np.ndarray:
    output = image.copy()
    for name, (x, y, w, h) in regions:
        if w <= 0 or h <= 0:
            continue
        color = (255, 0, 0) if name == "CMOS 排除框" else (255, 0, 255) if name == "CAP 成像区" else (0, 0, 255) if name == "CAP 允许范围" else (0, 200, 255)
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 3 if name == "CAP 允许范围" else 2)
    for index, item in enumerate(items, start=1):
        if item.label == "cap_offset":
            continue
        x, y, w, h = item.bbox
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
        label = str(index)
        if item.darkness is not None:
            label = f"{label}:{item.darkness:.2f}"
        if item.confidence is not None:
            label = f"{label} {item.confidence:.2f}"
        cv2.putText(output, label, (x, max(24, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    return output


def _letterbox(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height))
    horizontal = size - resized_width
    vertical = size - resized_height
    left = int(round(horizontal / 2 - 0.1))
    right = int(round(horizontal / 2 + 0.1))
    top = int(round(vertical / 2 - 0.1))
    bottom = int(round(vertical / 2 + 0.1))
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, scale, left, top


def _detect_yolo_image(
    net: cv2.dnn.Net,
    image: np.ndarray,
    class_threshold: float,
    nms_threshold: float,
) -> list[_YoloDetection]:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    input_image, scale, left, top = _letterbox(image)
    blob = cv2.dnn.blobFromImage(
        input_image,
        1.0 / 255.0,
        (640, 640),
        (0, 0, 0),
        swapRB=True,
        crop=False,
    )
    net.setInput(blob)
    raw_outputs = net.forward(net.getUnconnectedOutLayersNames())
    outputs = list(raw_outputs) if isinstance(raw_outputs, (list, tuple)) else [raw_outputs]
    if not outputs:
        return []
    output = max((np.asarray(value) for value in outputs), key=lambda value: value.size)
    if output.ndim == 3 and output.shape[0] == 1:
        output = output[0]
    if output.ndim != 2:
        raise ValueError(f"模型输出维度不支持：{output.shape}")
    if output.shape[1] != 9 and output.shape[0] == 9:
        output = output.T
    if output.shape[1] != 9:
        raise ValueError(
            f"模型输出格式不匹配：期望四分类 YOLO 输出 [N, 9]，实际 {output.shape}"
        )

    class_ids: list[int] = []
    confidences: list[float] = []
    boxes: list[list[int]] = []
    for row in output:
        object_score = float(row[4])
        if object_score < class_threshold:
            continue
        class_id = int(np.argmax(row[5:9]))
        confidence = object_score * float(row[5 + class_id])
        if confidence < class_threshold:
            continue
        center_x = (float(row[0]) - left) / scale
        center_y = (float(row[1]) - top) / scale
        width = float(row[2]) / scale
        height = float(row[3]) / scale
        x = max(int(round(center_x - width * 0.5)), 0)
        y = max(int(round(center_y - height * 0.5)), 0)
        class_ids.append(class_id)
        confidences.append(confidence)
        boxes.append([x, y, max(1, int(round(width))), max(1, int(round(height)))])

    if not boxes:
        return []
    indices = cv2.dnn.NMSBoxes(boxes, confidences, class_threshold, nms_threshold)
    return [
        _YoloDetection(class_ids[index], confidences[index], tuple(boxes[index]))
        for index in np.asarray(indices).reshape(-1)
    ]


def _merge_yolo_detections(detections: list[_YoloDetection]) -> list[_YoloDetection]:
    ordered = sorted(detections, key=lambda value: value.confidence, reverse=True)
    consumed = [False] * len(ordered)
    merged: list[_YoloDetection] = []
    for index, detection in enumerate(ordered):
        if consumed[index]:
            continue
        cluster = [index]
        consumed[index] = True
        cluster_index = 0
        while cluster_index < len(cluster):
            first = ordered[cluster[cluster_index]].bbox
            ax, ay, aw, ah = first
            for other_index in range(index + 1, len(ordered)):
                if consumed[other_index]:
                    continue
                bx, by, bw, bh = ordered[other_index].bbox
                dx = max(0, ax - (bx + bw), bx - (ax + aw))
                dy = max(0, ay - (by + bh), by - (ay + ah))
                if dx <= 20 and dy <= 20:
                    cluster.append(other_index)
                    consumed[other_index] = True
            cluster_index += 1

        selected = ordered[cluster[0]]
        x0 = min(ordered[value].bbox[0] for value in cluster)
        y0 = min(ordered[value].bbox[1] for value in cluster)
        x1 = max(ordered[value].bbox[0] + ordered[value].bbox[2] for value in cluster)
        y1 = max(ordered[value].bbox[1] + ordered[value].bbox[3] for value in cluster)
        for value in cluster[1:]:
            candidate = ordered[value]
            if candidate.bbox[2] * candidate.bbox[3] > selected.bbox[2] * selected.bbox[3]:
                selected = candidate
        merged.append(
            _YoloDetection(
                selected.class_id,
                selected.confidence,
                (x0, y0, x1 - x0, y1 - y0),
            )
        )
    return merged


def _filter_yolo_regions(
    detections: list[_YoloDetection],
    region_coords: Any,
    region_thresholds: Any,
) -> list[_YoloDetection]:
    if not isinstance(region_coords, list) or len(region_coords) < 4:
        return detections
    if not isinstance(region_thresholds, list) or len(region_thresholds) < 5:
        return detections
    regions = [_rect(value) for value in region_coords[:4]]
    threshold_rows: list[list[float]] = []
    for row in region_thresholds[:5]:
        if not isinstance(row, (list, tuple)) or len(row) < 4:
            return detections
        threshold_rows.append([float(value) for value in row[:4]])

    micrometers_per_pixel = 1.43
    square_micrometers_per_pixel = micrometers_per_pixel * micrometers_per_pixel
    filtered: list[_YoloDetection] = []
    for detection in detections:
        x, y, width, height = detection.bbox
        center = (x + width // 2, y + height // 2)
        region_index = next(
            (
                index
                for index, (rx, ry, rw, rh) in enumerate(regions)
                if rx <= center[0] < rx + rw and ry <= center[1] < ry + rh
            ),
            None,
        )
        if region_index is None:
            continue
        if detection.class_id in (2, 3):
            threshold = threshold_rows[1][region_index] / micrometers_per_pixel
            value = max(width, height)
        elif detection.class_id == 0:
            threshold = threshold_rows[0][region_index] / square_micrometers_per_pixel
            value = width * height
        else:
            threshold = threshold_rows[4][region_index] / square_micrometers_per_pixel
            value = width * height
        if value > threshold:
            filtered.append(detection)
    return filtered


def _run_5um_deep_learning(
    image: np.ndarray,
    detection: dict[str, Any],
    main: Rect,
    messages: list[str],
    callback: LogCallback | None,
) -> list[DetectionItem]:
    model_path = Path(str(detection.get("modelPath", "")).strip())
    if not model_path.is_file():
        raise FileNotFoundError(f"5um 模型文件不存在：{model_path}")

    class_names = ("bad", "foreign", "badwire", "scratch")
    class_thresholds = (
        float(detection.get("bad_conf_thresh", 0.35)),
        float(detection.get("foreign_conf_thresh", 0.35)),
        float(detection.get("badwire_conf_thresh", 0.65)),
        float(detection.get("scratch_conf_thresh", 0.35)),
    )
    nms_threshold = float(detection.get("nmsThreshold", 0.5))
    minimum_threshold = min(class_thresholds)
    _log(messages, callback, f"5um：加载深度学习模型 {model_path.name}")
    net = cv2.dnn.readNet(str(model_path))

    x, y, width, height = main
    roi = image[y:y + height, x:x + width]
    if roi.size == 0:
        raise ValueError("5um 主体检测区域为空。")
    tile_size = 640
    columns = (width + tile_size - 1) // tile_size
    rows = (height + tile_size - 1) // tile_size
    tile_positions: list[tuple[int, int]] = []
    for row in range(rows):
        for column in range(columns):
            tile_x = max(0, min(column * tile_size, width - tile_size))
            tile_y = max(0, min(row * tile_size, height - tile_size))
            position = (tile_x, tile_y)
            if position not in tile_positions:
                tile_positions.append(position)

    candidates: list[_YoloDetection] = []
    _log(messages, callback, f"5um：执行深度学习分块检测，共 {len(tile_positions)} 块")
    for tile_x, tile_y in tile_positions:
        tile_width = min(tile_size, width - tile_x)
        tile_height = min(tile_size, height - tile_y)
        tile = roi[tile_y:tile_y + tile_height, tile_x:tile_x + tile_width]
        if tile_width < tile_size or tile_height < tile_size:
            padded = np.zeros((tile_size, tile_size, 3), dtype=tile.dtype)
            if tile.ndim == 2:
                padded = np.zeros((tile_size, tile_size), dtype=tile.dtype)
            padded[:tile_height, :tile_width] = tile
            tile = padded
        for result in _detect_yolo_image(net, tile, minimum_threshold, nms_threshold):
            bx, by, bw, bh = result.bbox
            candidates.append(
                _YoloDetection(
                    result.class_id,
                    result.confidence,
                    _clamp((x + tile_x + bx, y + tile_y + by, bw, bh), image.shape),
                )
            )

    _log(messages, callback, "5um：执行主体区域整体缩放检测")
    compressed = cv2.resize(roi, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
    scale_x = width / tile_size
    scale_y = height / tile_size
    for result in _detect_yolo_image(net, compressed, minimum_threshold, nms_threshold):
        bx, by, bw, bh = result.bbox
        candidates.append(
            _YoloDetection(
                result.class_id,
                result.confidence,
                _clamp(
                    (
                        x + int(bx * scale_x),
                        y + int(by * scale_y),
                        max(1, int(bw * scale_x)),
                        max(1, int(bh * scale_y)),
                    ),
                    image.shape,
                ),
            )
        )

    candidates = [
        value
        for value in candidates
        if value.confidence >= class_thresholds[value.class_id]
        and value.bbox[2] > 0
        and value.bbox[3] > 0
    ]
    candidates = _merge_yolo_detections(candidates)
    candidates = _filter_yolo_regions(
        candidates,
        detection.get("regionCoords"),
        detection.get("regionThresh"),
    )
    _log(messages, callback, f"5um：深度学习检测完成，缺陷 {len(candidates)} 个")
    return [
        DetectionItem(
            class_names[value.class_id],
            value.bbox,
            float(value.bbox[2] * value.bbox[3]),
            "深度学习",
            value.confidence,
        )
        for value in candidates
    ]


def _merge_5um_chipping_boxes(boxes: list[Rect]) -> list[Rect]:
    checked: set[int] = set()
    merged: list[Rect] = []
    for index, box in enumerate(boxes):
        if index in checked:
            continue
        checked.add(index)
        combined: Rect | None = None
        for other_index in range(index + 1, len(boxes)):
            if other_index in checked:
                continue
            intersection = _intersection(box, boxes[other_index])
            if intersection[2] > 0 and intersection[3] > 0:
                checked.add(other_index)
                combined = _union(box, boxes[other_index])
                break
        merged.append(combined if combined is not None else box)
    return merged


def _detect_5um_chipping(
    gray: np.ndarray,
    detection: dict[str, Any],
) -> list[DetectionItem]:
    image_height, image_width = gray.shape[:2]
    offset = max(0, int(detection.get("offset", 10)))
    safe_line = max(
        1,
        int(detection.get("safeLine", 55)) + offset,
    )
    safe_line = min(safe_line, image_height, image_width)
    gray_threshold = int(detection.get("thresh_gray_chip", 50))
    exclude_x0 = offset
    exclude_x1 = image_width - offset
    side_images = (
        ("top", gray[:safe_line, :]),
        ("bottom", gray[image_height - safe_line:, :]),
        ("left", gray[:, :safe_line]),
        ("right", gray[:, image_width - safe_line:]),
    )
    boxes: list[Rect] = []

    for direction, side_gray in side_images:
        if direction in {"left", "right"}:
            side_gray = cv2.rotate(side_gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
        binary = cv2.threshold(
            side_gray,
            gray_threshold,
            1,
            cv2.THRESH_BINARY_INV,
        )[1]
        column_sums = np.sum(binary, axis=0)
        binary[:, column_sums < offset] = 0
        corner_width = min(offset, binary.shape[1])
        if corner_width > 0:
            binary[:, :corner_width] = 0
            binary[:, binary.shape[1] - corner_width:] = 0

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            width_threshold = offset
            if x == exclude_x0 or x + width == exclude_x1:
                width_threshold = safe_line
            if height != safe_line or width <= width_threshold:
                continue
            if direction == "top":
                box = (x, y, width, height)
            elif direction == "bottom":
                box = (
                    x,
                    y + image_height - side_gray.shape[0],
                    width,
                    height,
                )
            elif direction == "left":
                box = (0, x, height, width)
            else:
                box = (
                    image_width - side_gray.shape[0],
                    x,
                    height,
                    width,
                )
            boxes.append(box)

    return [
        DetectionItem(
            "chipping",
            box,
            float(box[2] * box[3]),
            "崩边检测",
        )
        for box in _merge_5um_chipping_boxes(boxes)
    ]


def _run_5um(image: np.ndarray, params: dict[str, Any], callback: LogCallback | None) -> DetectionOutput:
    detection = _section(params, "DetectionParams") or params
    messages: list[str] = []
    items: list[DetectionItem] = []
    regions: list[tuple[str, Rect]] = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    coords = detection.get("regionCoords", [])
    parsed = [_rect(value) for value in coords] if isinstance(coords, list) else []
    region_names = ("A", "B", "C", "Dummy", "Detection")
    for index, region in enumerate(parsed):
        name = region_names[index] if index < len(region_names) else f"Region {index + 1}"
        regions.append((name, _clamp(region, image.shape)))
    main = _clamp(parsed[3] if len(parsed) > 3 else (0, 0, image.shape[1], image.shape[0]), image.shape)
    excludes = [_rect(value) for value in detection.get("excludeZones", []) if isinstance(value, dict)]

    _log(messages, callback, "5um：执行磨砂检测")
    x, y, w, h = main
    if w > 1 and h > 1:
        cy, cx = h // 2, w // 2
        base = min(cx, cy)
        low_radius = base * float(detection.get("lf_rate", 0.03))
        middle_radius = base * float(detection.get("mf_rate", 0.4))
        spectrum = cv2.dft(
            gray[y:y + h, x:x + w].astype(np.float32),
            flags=cv2.DFT_COMPLEX_OUTPUT,
        )
        columns = np.arange(w)
        column_distance = np.minimum(columns, w - columns) ** 2
        low = 0.0
        middle = 0.0
        for row in range(h):
            radius_squared = column_distance + min(row, h - row) ** 2
            magnitude = cv2.magnitude(spectrum[row, :, 0], spectrum[row, :, 1]).reshape(-1)
            low_mask = radius_squared <= low_radius ** 2
            middle_mask = (radius_squared > low_radius ** 2) & (radius_squared <= middle_radius ** 2)
            low += float(magnitude[low_mask].sum(dtype=np.float64))
            middle += float(magnitude[middle_mask].sum(dtype=np.float64))
        if low > 1e-12 and middle / low > float(detection.get("abrazineThresh", 30.0)):
            items.append(DetectionItem("abrazine", main, float(w * h), "磨砂检测"))
            _log(messages, callback, "5um：检出磨砂，结束后续检测")
            return DetectionOutput(_draw_output(image, items, regions), items, regions, messages)

    items.extend(_run_5um_deep_learning(image, detection, main, messages, callback))

    pad_args = {
        "thresh": int(detection.get("padGrayThresh", 220)),
        "pad_min_num": int(detection.get("padMinNum", 6)),
        "pad_w": int(detection.get("padW", 66)),
        "pad_h": int(detection.get("padH", 66)),
        "pad_range": int(detection.get("padRange", 6)),
    }
    search = min(int(detection.get("padSearchRange", 210)), max(1, gray.shape[1] // 2))
    left = find_pad_column(gray[:, :search], **pad_args)
    right_local = find_pad_column(gray[:, gray.shape[1] - search:], **pad_args)
    right = [(px + gray.shape[1] - search, py, pw, ph) for px, py, pw, ph in right_local]
    pads = left + right
    _log(messages, callback, f"5um：Pad 定位完成，共 {len(pads)} 个")

    kernel_size = max(1, int(detection.get("kernel", 5)))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    pad_ratio = float(detection.get("padTh", 0.33))
    average_area = sum(pw * ph for _, _, pw, ph in pads) / len(pads) if pads else 0.0
    for box in pads:
        px, py, pw, ph = _clamp(box, gray.shape)
        if pw <= 0 or ph <= 0:
            continue
        _, binary = cv2.threshold(gray[py:py + ph, px:px + pw], int(detection.get("stain_threshold", 100)), 255, cv2.THRESH_BINARY_INV)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        if cv2.countNonZero(binary) > average_area * pad_ratio:
            items.append(DetectionItem("dirty", (px, py, pw, ph), float(pw * ph), "Pad 脏污"))

    outer_values = detection.get("outerDetectionZones", [])
    outer = [_clamp(_rect(value), gray.shape) for value in outer_values if isinstance(value, dict)]
    safe_line = int(detection.get("safeLine", 55)) + int(detection.get("offset", 10))
    if not outer and left and right:
        left_edge = int(np.mean([px + pw for px, _, pw, _ in left]))
        right_edge = int(np.mean([px for px, _, _, _ in right]))
        outer = [_clamp((left_edge, safe_line, right_edge - left_edge, gray.shape[0] - safe_line * 2), gray.shape)]

    chipping_offset = max(0, int(detection.get("offset", 10)))
    chipping_safe_line = max(
        1,
        int(detection.get("safeLine", 55)) + chipping_offset,
    )
    chipping_safe_line = min(chipping_safe_line, gray.shape[0], gray.shape[1])
    chipping_gray_threshold = int(detection.get("thresh_gray_chip", 50))
    _log(
        messages,
        callback,
        "5um：执行崩边检测，"
        f"灰度阈值 {chipping_gray_threshold}，"
        f"安全线 {chipping_safe_line}，偏移 {chipping_offset}，"
        f"图像 {gray.shape[1]}×{gray.shape[0]}",
    )
    chipping_items = _detect_5um_chipping(gray, detection)

    segmentation_path = Path(str(detection.get(
        "blindCircuitModelPath", DEFAULT_BLIND_CIRCUIT_MODEL_PATH
    )).strip())
    if not segmentation_path.is_file():
        raise FileNotFoundError(f"盲元电路区模型文件不存在：{segmentation_path}")
    if segmentation_path.suffix.lower() not in {".engine", ".onnx"}:
        raise ValueError("盲元电路区模型必须是 .engine 或 .onnx 文件。")
    probability_threshold = float(detection.get("blindCircuitThreshold", 0.35))
    if not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("盲元电路区概率阈值必须在 0 到 1 之间。")
    result_threshold = float(detection.get("blindCircuitScoreThreshold", 0.0))
    if not 0.0 <= result_threshold <= 1.0:
        raise ValueError("盲元/电路/外围区结果平均像素概率下限必须在 0 到 1 之间。")
    _log(messages, callback, f"5um：加载盲元电路区模型 {segmentation_path.name}")
    segmentation_model = (
        BlindCircuitTensorRt(segmentation_path)
        if segmentation_path.suffix.lower() == ".engine"
        else nullcontext(cv2.dnn.readNet(str(segmentation_path)))
    )
    with segmentation_model as segmentation_net:
        outer_results: list[DetectionItem] = []
        for roi in outer:
            regions.append(("外围区", roi))
            outer_results.extend(_segmented_5um_items(
                image, roi, segmentation_net, probability_threshold,
                int(detection.get("coreMinArea", 300)), "外围区模型检测",
                excludes + [main],
            ))
        changed = True
        while changed:
            changed = False
            for index, result in enumerate(outer_results):
                for other_index in range(index + 1, len(outer_results)):
                    other = outer_results[other_index]
                    overlap = _intersection(result.bbox, other.bbox)
                    if overlap[2] <= 0 or overlap[3] <= 0:
                        continue
                    box = _union(result.bbox, other.bbox)
                    outer_results[index] = DetectionItem(
                        "other", box, float(box[2] * box[3]),
                        result.algorithm,
                        max(
                            (score for score in (result.confidence, other.confidence) if score is not None),
                            default=None,
                        ),
                    )
                    outer_results.pop(other_index)
                    changed = True
                    break
                if changed:
                    break
        for result in outer_results:
            if result_threshold > 0.0 and (
                result.confidence is None or result.confidence < result_threshold
            ):
                continue
            _, _, width, height = result.bbox
            ratio = max(width, height) / max(1, min(width, height))
            items.append(DetectionItem(
                "外围检测划伤" if ratio >= 4.0 else "外围检测异物",
                result.bbox, result.area, result.algorithm, result.confidence,
            ))
        items.extend(chipping_items)
        for name, roi in five_um_blind_circuit_regions(gray.shape, detection):
            regions.append((name, roi))
            for result in _segmented_5um_items(
                image, roi, segmentation_net, probability_threshold,
                30, name, excludes,
            ):
                if result_threshold > 0.0 and (
                    result.confidence is None or result.confidence < result_threshold
                ):
                    continue
                items.append(DetectionItem(
                    "盲元破损" if name == "盲元区" else "电路区异物",
                    result.bbox, result.area, result.algorithm, result.confidence,
                ))
    _log(messages, callback, f"5um：检测完成，缺陷 {len(items)} 个")
    return DetectionOutput(_draw_output(image, items, regions), items, regions, messages)


def _template_match(image: np.ndarray, template: np.ndarray, split: dict[str, Any]) -> tuple[float, Rect]:
    ratio = float(split.get("ratio", 0.1))
    kernel = int(split.get("ksize", 5))
    kernel = kernel if kernel > 0 and kernel % 2 == 1 else 5
    sigma = float(split.get("sigmax", 0.0))

    def preprocess(source: np.ndarray) -> np.ndarray:
        resized = cv2.resize(source, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (kernel, kernel), sigma)

    source = preprocess(image)
    target = preprocess(template)
    if source.shape[0] < target.shape[0] or source.shape[1] < target.shape[1]:
        raise ValueError("模板尺寸大于待检测图像")
    mask = np.full(target.shape, 255, np.uint8)
    inset_x, inset_y = int(target.shape[1] * 0.2), int(target.shape[0] * 0.2)
    if target.shape[1] > inset_x * 2 and target.shape[0] > inset_y * 2:
        mask[inset_y:target.shape[0] - inset_y, inset_x:target.shape[1] - inset_x] = 0
    result = cv2.matchTemplate(source, target, cv2.TM_CCORR_NORMED, mask=mask)
    _, score, _, location = cv2.minMaxLoc(result)
    return float(score), (int(location[0] / ratio), int(location[1] / ratio), template.shape[1], template.shape[0])


def _split_20um(
    up_image: np.ndarray,
    down_image: np.ndarray,
    split: dict[str, Any],
    callback: LogCallback | None,
    messages: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    Rect,
    np.ndarray,
    np.ndarray,
    tuple[int, int],
    float,
    np.ndarray,
    np.ndarray,
    Rect,
]:
    template_path = str(split.get("tempPath", ""))
    if not template_path or not Path(template_path).is_file():
        raise ValueError(f"20um 模板不存在：{template_path}")
    template = imread_unicode(template_path)
    if template is None:
        raise ValueError(f"无法读取 20um 模板：{template_path}")
    score, match = _template_match(down_image, template, split)
    threshold = float(split.get("matchedThresh", 0.65))
    _log(messages, callback, f"20um：第一次模板匹配 {score:.4f}")
    if score < threshold:
        raise ValueError(f"第一次模板匹配低于阈值：{score:.4f} < {threshold:.4f}")
    mx, my, mw, mh = _clamp(match, down_image.shape)
    angle, left_pads, right_pads = estimate_pad_rotation(
        down_image[my:my + mh, mx:mx + mw],
        thresh=int(split.get("padGrayThresh", 210)),
        search_width=int(split.get("padSearchRange", 200)),
        pad_min_num=int(split.get("padMinNum", 6)),
        pad_w=int(split.get("padW", 36)),
        pad_h=int(split.get("padH", 36)),
        pad_range=int(split.get("padRange", 10)),
    )
    if angle is None:
        raise ValueError("20um 模板区域内未找到左右 Pad 列")
    angle = angle if abs(angle) < float(split.get("maxRotateAngle", 1.0)) else 0.0
    matrix = cv2.getRotationMatrix2D((down_image.shape[1] / 2.0, down_image.shape[0] / 2.0), angle, 1.0)
    rotated_down = cv2.warpAffine(down_image, matrix, (down_image.shape[1], down_image.shape[0]))
    rotated_up = cv2.warpAffine(up_image, matrix, (up_image.shape[1], up_image.shape[0]))
    score, match = _template_match(rotated_down, template, split)
    _log(messages, callback, f"20um：第二次模板匹配 {score:.4f}，旋转 {angle:.4f}°")
    if score < threshold:
        raise ValueError(f"第二次模板匹配低于阈值：{score:.4f} < {threshold:.4f}")
    x, y, w, h = match
    top_offset = int(split.get("capTopOffset", 200))
    roi = _clamp((x, max(0, y - top_offset), w, h + top_offset), rotated_down.shape)
    x, y, w, h = roi
    up_crop = rotated_up[y:y + h, x:x + w].copy()
    down_crop = rotated_down[y:y + h, x:x + w].copy()
    window_result = segment_window(up_crop)
    if window_result is None:
        raise ValueError("20um 窗片分割失败")
    cap_corners = np.asarray(window_result["box"], dtype=np.int32)
    cap_mask = np.asarray(window_result["mask"], dtype=np.uint8)
    cap_rect = cv2.boundingRect(cap_corners)
    _log(messages, callback, "20um：窗片四边形分割完成")
    left_center = int(np.mean([px + pw / 2 for px, _, pw, _ in left_pads])) if left_pads else 0
    right_center = int(np.mean([px + pw / 2 for px, _, pw, _ in right_pads])) if right_pads else w
    return (
        up_crop,
        down_crop,
        cap_rect,
        cap_mask,
        cap_corners,
        (left_center, right_center),
        angle,
        rotated_up,
        rotated_down,
        roi,
    )


def _load_crop_template(value: Any) -> np.ndarray:
    template_path = Path(str(value or "").strip())
    if template_path.is_dir():
        preferred = template_path / "temp.jpg"
        candidates = [preferred] if preferred.is_file() else sorted(
            path
            for path in template_path.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not candidates:
            raise ValueError(f"模板目录中没有可用图片：{template_path}")
        template_path = candidates[0]
    if not template_path.is_file():
        raise ValueError(f"模板不存在：{template_path}")
    template = imread_unicode(str(template_path))
    if template is None:
        raise ValueError(f"无法读取模板：{template_path}")
    return template


def _crop_preprocess(
    image: np.ndarray,
    ratio: float,
    kernel_size: int,
    sigma_x: float,
) -> np.ndarray:
    if ratio <= 0:
        raise ValueError("裁剪参数 ratio 必须大于 0。")
    kernel_size = max(1, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    resized = cv2.resize(
        image,
        None,
        fx=ratio,
        fy=ratio,
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma_x)


def _match_crop_template(
    image: np.ndarray,
    prepared_template: np.ndarray,
    template_size: tuple[int, int],
    ratio: float,
    kernel_size: int,
    sigma_x: float,
) -> tuple[float, Rect]:
    source = _crop_preprocess(image, ratio, kernel_size, sigma_x)
    if source.shape[0] < prepared_template.shape[0] or source.shape[1] < prepared_template.shape[1]:
        raise ValueError("模板尺寸大于待裁剪图像。")
    result = cv2.matchTemplate(source, prepared_template, cv2.TM_CCORR_NORMED)
    _, score, _, location = cv2.minMaxLoc(result)
    width, height = template_size
    return float(score), (
        int(location[0] / ratio),
        int(location[1] / ratio),
        width,
        height,
    )


def _run_5um_auto_crop(
    image: np.ndarray,
    params: dict[str, Any],
    last_angle: float,
    callback: LogCallback | None,
) -> AutoCropOutput:
    split = _section(params, "SplitImageParams")
    detection = _section(params, "DetectionParams")
    if not split:
        raise ValueError("当前 5um 配方缺少 SplitImageParams。")
    template = _load_crop_template(split.get("tempPath"))
    ratio = float(split.get("ratio", 0.1))
    kernel_size = int(split.get("kSize", 5))
    sigma_x = float(split.get("sigmaX", 0.0))
    matched_threshold = float(split.get("matchedThresh", 0.65))
    max_angle = float(split.get("maxRotateAngle", 1.0))
    prepared_template = _crop_preprocess(template, ratio, kernel_size, sigma_x)
    template_size = (template.shape[1], template.shape[0])
    messages: list[str] = []

    score, rect = _match_crop_template(
        image,
        prepared_template,
        template_size,
        ratio,
        kernel_size,
        sigma_x,
    )
    _log(messages, callback, f"5um 自动裁剪：第一次模板匹配 {score:.4f}")
    if score <= matched_threshold:
        raise ValueError(
            f"首次模板匹配失败：{score:.4f} <= {matched_threshold:.4f}"
        )
    x, y, width, height = _clamp(rect, image.shape)
    if width <= 0 or height <= 0:
        raise ValueError("首次模板匹配区域无效。")
    roi = image[y:y + height, x:x + width]

    pad_threshold = int(detection.get("padGrayThresh", 220))
    pad_width = int(detection.get("padW", 66))
    pad_height = int(detection.get("padH", 66))
    pad_range = int(detection.get("padRange", 6))
    pad_search = min(int(detection.get("padSearchRange", 210)), roi.shape[1])
    pad_minimum = int(detection.get("padMinNum", 6))
    left_pads = find_pad_column(
        roi[:, :pad_search],
        thresh=pad_threshold,
        pad_min_num=pad_minimum,
        pad_w=pad_width,
        pad_h=pad_height,
        pad_range=pad_range,
    )
    right_start = roi.shape[1] - pad_search
    right_local = find_pad_column(
        roi[:, right_start:],
        thresh=pad_threshold,
        pad_min_num=pad_minimum,
        pad_w=pad_width,
        pad_h=pad_height,
        pad_range=pad_range,
    )
    if len(left_pads) < pad_minimum or len(right_local) < pad_minimum:
        raise ValueError("未找到满足数量要求的左右 Pad 列。")

    def pad_angle(boxes: list[Rect], offset_x: int) -> float:
        centers = np.asarray(
            [
                (px + pw / 2 + offset_x, py + ph / 2)
                for px, py, pw, ph in boxes
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        line = cv2.fitLine(centers, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy = (float(value) for value in line.reshape(-1)[:2])
        angle = np.degrees(np.arctan2(vy, vx)) % 180.0
        return float(angle - 90.0)

    raw_angle = (
        pad_angle(left_pads, 0) + pad_angle(right_local, right_start)
    ) / 2.0
    if abs(raw_angle) < max_angle:
        angle = raw_angle
    elif abs(last_angle) > 1e-9:
        angle = last_angle
    else:
        angle = 0.0
    _log(
        messages,
        callback,
        f"5um 自动裁剪：Pad 角度 {raw_angle:.4f}°，采用 {angle:.4f}°",
    )

    image_height, image_width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(
        (image_width / 2.0, image_height / 2.0),
        angle,
        1.0,
    )
    rotated = cv2.warpAffine(image, matrix, (image_width, image_height))
    score, rect = _match_crop_template(
        rotated,
        prepared_template,
        template_size,
        ratio,
        kernel_size,
        sigma_x,
    )
    _log(messages, callback, f"5um 自动裁剪：第二次模板匹配 {score:.4f}")
    if score <= matched_threshold:
        raise ValueError(
            f"二次模板匹配失败：{score:.4f} <= {matched_threshold:.4f}"
        )
    x, y, width, height = _clamp(rect, rotated.shape)
    if width <= 0 or height <= 0:
        raise ValueError("二次模板裁剪区域无效。")
    return AutoCropOutput(
        image=rotated,
        angle=angle,
        messages=messages,
        crop_rect=(x, y, width, height),
    )


def run_wafer_crop(
    image: np.ndarray,
    resolution_mode: str,
    params: dict[str, Any],
    *,
    up_image: np.ndarray | None = None,
    last_angle: float = 0.0,
    log_callback: LogCallback | None = None,
) -> AutoCropOutput:
    if image is None or image.size == 0:
        raise ValueError("当前没有可裁剪图像。")
    if resolution_mode == "5um":
        return _run_5um_auto_crop(image, params, last_angle, log_callback)
    if resolution_mode == "20um":
        if up_image is None or up_image.size == 0:
            raise ValueError("20um 自动裁剪需要当前批次的上下层拼接图。")
        split = _section(params, "splitParams", "SplitParams")
        messages: list[str] = []
        (
            up_crop,
            down_crop,
            cap_box,
            cap_mask,
            cap_corners,
            pad_centers,
            angle,
            rotated_up,
            rotated_down,
            crop_rect,
        ) = _split_20um(
            up_image,
            image,
            split,
            log_callback,
            messages,
        )
        return AutoCropOutput(
            image=rotated_down,
            angle=angle,
            messages=messages,
            crop_rect=crop_rect,
            cap_image=up_crop,
            cmos_image=down_crop,
            cap_box=cap_box,
            pad_centers=pad_centers,
            up_image=rotated_up,
            down_image=rotated_down,
            cap_mask=cap_mask,
            cap_corners=cap_corners,
        )
    raise ValueError(f"不支持的分辨率：{resolution_mode}")


def _centered_crop_rect(rect: Rect, bounds: Rect, size: int = 128) -> Rect:
    bx, by, bw, bh = bounds
    x, y, width, height = rect
    crop_size = max(size, width, height)
    if crop_size > bw or crop_size > bh:
        return _intersection(rect, bounds)
    center_x = x + width // 2
    center_y = y + height // 2
    crop_x = max(bx, min(center_x - crop_size // 2, bx + bw - crop_size))
    crop_y = max(by, min(center_y - crop_size // 2, by + bh - crop_size))
    return crop_x, crop_y, crop_size, crop_size


def _calc_ratio_mean(
    gray: np.ndarray,
    bbox: Rect,
    ratio: float,
) -> float | None:
    if gray.size == 0 or ratio <= 0 or ratio > 1:
        return None
    x, y, width, height = _clamp(bbox, gray.shape)
    if width <= 0 or height <= 0:
        return None
    roi = gray[y:y + height, x:x + width]
    target_pixels = int(roi.size * ratio)
    if target_pixels <= 0:
        return None
    histogram = np.bincount(roi.ravel(), minlength=256)
    accumulated_count = 0
    gray_sum = 0.0
    for gray_value, pixel_count_value in enumerate(histogram):
        pixel_count = int(pixel_count_value)
        if accumulated_count + pixel_count > target_pixels:
            needed = target_pixels - accumulated_count
            gray_sum += gray_value * needed
            break
        gray_sum += gray_value * pixel_count
        accumulated_count += pixel_count
        if accumulated_count == target_pixels:
            break
    return 1.0 - gray_sum / target_pixels / 255.0


def _filter_20um_depth_items(
    items: list[DetectionItem],
    gray: np.ndarray,
    detection: dict[str, Any],
) -> list[DetectionItem]:
    ratio = float(detection.get("darkSearchRatio", 0.1))
    filtered: list[DetectionItem] = []
    for item in items:
        if item.label not in {"cap_point", "cap_dirty"}:
            filtered.append(item)
            continue
        darkness = _calc_ratio_mean(gray, item.bbox, ratio)
        if darkness is None:
            continue
        if item.label == "cap_point":
            minimum_area1 = float(detection.get("capPointMinArea1", 0))
            minimum_area2 = float(detection.get("capPointMinArea2", 0))
            gray_threshold1 = float(detection.get("capPointGrayThresh1", 0))
            gray_threshold2 = float(detection.get("capPointGrayThresh2", 0))
        else:
            minimum_area1 = float(detection.get("capDirtyMinArea1", 0))
            minimum_area2 = float(detection.get("capDirtyMinArea2", 0))
            gray_threshold1 = float(detection.get("capDirtyGrayThresh1", 0))
            gray_threshold2 = float(detection.get("capDirtyGrayThresh2", 0))
        if not (
            (item.area > minimum_area1 and darkness > gray_threshold1)
            or (item.area > minimum_area2 and darkness > gray_threshold2)
        ):
            continue
        filtered.append(
            DetectionItem(
                item.label,
                item.bbox,
                item.area,
                item.algorithm,
                item.confidence,
                darkness,
            )
        )
    return filtered


def _run_20um_deep_learning(
    image: np.ndarray,
    binary: np.ndarray,
    cap_box: Rect,
    cap_mask: np.ndarray,
    cap_corners: np.ndarray,
    imaging: Rect,
    detection: dict[str, Any],
    messages: list[str],
    callback: LogCallback | None,
    cancel_callback: CancelCallback | None,
) -> list[DetectionItem]:
    _check_cancelled(cancel_callback)
    model_path = Path(str(detection.get("onnxPath", "")).strip())
    if model_path.suffix.lower() != ".pt":
        raise ValueError(f"20um 模型必须是 PT 文件：{model_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"20um 模型文件不存在：{model_path}")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("20um PT 模型需要安装 PyTorch。") from exc

    shrink = int(detection.get("capChipShrinkPix", 0))
    inner_mask = np.zeros_like(binary)
    if len(cap_corners) == 4:
        shrink_corners = cap_corners + np.array(
            ((shrink, shrink), (-shrink, shrink),
             (-shrink, -shrink), (shrink, -shrink)),
            dtype=np.int32,
        )
        inner = _clamp(cv2.boundingRect(shrink_corners), image.shape)
        cv2.fillPoly(inner_mask, [shrink_corners], 255)
    else:
        inner = _clamp(
            (
                cap_box[0] + shrink,
                cap_box[1] + shrink,
                cap_box[2] - shrink * 2,
                cap_box[3] - shrink * 2,
            ),
            image.shape,
        )
        ix, iy, iw, ih = inner
        inner_mask[iy:iy + ih, ix:ix + iw] = 255
    ix, iy, iw, ih = inner
    if iw <= 0 or ih <= 0:
        raise ValueError("20um CAP 内缩检测区域为空。")
    candidate_binary = cv2.bitwise_and(binary, cap_mask)
    candidate_binary = cv2.bitwise_and(candidate_binary, inner_mask)
    if not bool(detection.get("detectNonImaging", True)):
        imaging_mask = np.zeros_like(binary)
        mx, my, mw, mh = _clamp(imaging, image.shape)
        imaging_mask[my:my + mh, mx:mx + mw] = 255
        candidate_binary = cv2.bitwise_and(candidate_binary, imaging_mask)

    contours, _ = cv2.findContours(
        candidate_binary[iy:iy + ih, ix:ix + iw],
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    minimum_area = int(detection.get("imagingMinDefectArea", 0))
    point_circularity = float(detection.get("pointCircularity", 0.85))
    items: list[DetectionItem] = []
    candidate_boxes: list[Rect] = []
    for contour in contours:
        _check_cancelled(cancel_callback)
        bx, by, width, height = cv2.boundingRect(contour)
        if width * height <= minimum_area:
            continue
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (
            4.0 * np.pi * area / (perimeter * perimeter)
            if perimeter > 1e-6
            else 0.0
        )
        box = (ix + bx, iy + by, width, height)
        if circularity > point_circularity:
            items.append(DetectionItem("cap_point", box, area, "CAP 点缺陷"))
        else:
            candidate_boxes.append(box)

    candidate_boxes = _merge_boxes(candidate_boxes, 10)
    if not candidate_boxes:
        return items

    _check_cancelled(cancel_callback)
    global _20UM_PT_MODEL_CACHE
    model_key = (model_path.resolve(), model_path.stat().st_mtime_ns)
    with _20UM_PT_MODEL_LOCK:
        if _20UM_PT_MODEL_CACHE is None or _20UM_PT_MODEL_CACHE[:2] != model_key:
            _log(messages, callback, f"20um：加载 TorchScript 模型 {model_path.name}")
            load_start = perf_counter()
            model = torch.jit.load(str(model_path), map_location="cpu")
            model.eval()
            _20UM_PT_MODEL_CACHE = (*model_key, model)
            _log(messages, callback, f"20um：PT 模型加载耗时 {perf_counter() - load_start:.3f} 秒")
        else:
            model = _20UM_PT_MODEL_CACHE[2]
            _log(messages, callback, f"20um：复用 TorchScript 模型 {model_path.name}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)
    class_names = ("cap_chip", "cap_dirty", "cap_scratch")
    chip_edge = int(float(detection.get("capChipMinLengthUm", 0.0)) / 2.5)
    inference_seconds = 0.0
    for box in candidate_boxes:
        _check_cancelled(cancel_callback)
        crop_x, crop_y, crop_width, crop_height = _centered_crop_rect(box, inner)
        if crop_width <= 0 or crop_height <= 0:
            continue
        crop = gray[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
        crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(dtype=torch.float32).div(255.0)
        tensor = tensor.sub(mean).div(std)
        with torch.no_grad():
            inference_start = perf_counter()
            probabilities = torch.softmax(model(tensor), dim=1)[0]
            inference_seconds += perf_counter() - inference_start
        confidence, class_index = torch.max(probabilities, dim=0)
        label = class_names[int(class_index.item())]
        result_box = box
        if label == "cap_chip":
            x, y, width, height = box
            near_edge = False
            if x - ix < chip_edge:
                x -= shrink
                width += shrink
                near_edge = True
            elif x + width > ix + iw - chip_edge:
                width += shrink
                near_edge = True
            if y - iy < chip_edge:
                y -= shrink
                height += shrink
                near_edge = True
            elif y + height > iy + ih - chip_edge:
                height += shrink
                near_edge = True
            if near_edge:
                result_box = _clamp((x, y, width, height), image.shape)
            else:
                label = "cap_dirty"
        items.append(
            DetectionItem(
                label,
                result_box,
                float(result_box[2] * result_box[3]),
                "TorchScript ResNet",
                float(confidence.item()),
            )
        )
    _log(messages, callback, f"20um：PT 候选推理 {len(candidate_boxes)} 个，耗时 {inference_seconds:.3f} 秒")
    _log(messages, callback, f"20um：PT 分类完成，缺陷 {len(items)} 个")
    return items


def _merge_20um_detection_items(
    traditional_items: list[DetectionItem],
    model_items: list[DetectionItem],
) -> list[DetectionItem]:
    remaining = list(traditional_items)
    merged_models: list[DetectionItem] = []
    cap_labels = {"cap_point", "cap_dirty", "cap_scratch"}
    for model_item in model_items:
        x0, y0, width, height = model_item.bbox
        x1 = x0 + width
        y1 = y0 + height
        kept: list[DetectionItem] = []
        for item in remaining:
            overlap = _intersection(model_item.bbox, item.bbox)
            if item.label not in cap_labels or overlap[2] <= 0 or overlap[3] <= 0:
                kept.append(item)
                continue
            item_x, item_y, item_width, item_height = item.bbox
            x0 = min(x0, item_x)
            y0 = min(y0, item_y)
            x1 = max(x1, item_x + item_width)
            y1 = max(y1, item_y + item_height)
        box = (x0, y0, x1 - x0, y1 - y0)
        merged_models.append(
            DetectionItem(
                model_item.label,
                box,
                float(box[2] * box[3]),
                model_item.algorithm,
                model_item.confidence,
            )
        )
        remaining = kept
    remaining.extend(merged_models)
    return remaining


def _run_20um(
    image: np.ndarray,
    up_image: np.ndarray | None,
    params: dict[str, Any],
    callback: LogCallback | None,
    crop_output: AutoCropOutput | None = None,
    cancel_callback: CancelCallback | None = None,
    cropped_pair: bool = False,
    cropped_layer: str | None = None,
) -> DetectionOutput:
    _check_cancelled(cancel_callback)
    if cropped_layer not in {None, "up", "down"} or (cropped_pair and cropped_layer):
        raise ValueError("20um 裁剪图检测模式无效。")
    split = _section(params, "splitParams", "SplitParams")
    detection = _section(params, "detectParams", "DetectParams")
    messages: list[str] = []
    debug_images: dict[str, np.ndarray] = {}
    if cropped_layer == "up":
        up_crop = image.copy()
        down_crop = image
        window_result = segment_window(up_crop)
        if window_result is None:
            cap_corners = np.empty((0, 2), dtype=np.int32)
            cap_mask = np.full(up_crop.shape[:2], 255, dtype=np.uint8)
            cap_box = (0, 0, up_crop.shape[1], up_crop.shape[0])
            _log(messages, callback, "20um：上图未定位窗片，按整张图作为 CAP 区域")
        else:
            cap_corners = np.asarray(window_result["box"], dtype=np.int32)
            cap_mask = np.asarray(window_result["mask"], dtype=np.uint8)
            cap_box = cv2.boundingRect(cap_corners)
        pad_centers = None
        _log(messages, callback, "20um：单独检测已裁剪上图")
    elif cropped_layer == "down":
        up_crop = image
        down_crop = image.copy()
        cap_corners = np.empty((0, 2), dtype=np.int32)
        cap_mask = np.zeros(down_crop.shape[:2], dtype=np.uint8)
        cap_box = (0, 0, 0, 0)
        pad_centers = None
        _log(messages, callback, "20um：单独检测已裁剪下图，缺少 CAP 遮罩，仅使用 CMOS 安全区和排除框")
    elif cropped_pair:
        if up_image is None or up_image.size == 0:
            raise ValueError("20um 成对裁剪图缺少上图。")
        if up_image.shape != image.shape:
            raise ValueError("20um 上图和下图的尺寸及通道数必须一致。")
        up_crop = up_image.copy()
        down_crop = image.copy()
        window_result = segment_window(up_crop)
        if window_result is None:
            raise ValueError("20um 上图窗片分割失败。")
        cap_corners = np.asarray(window_result["box"], dtype=np.int32)
        cap_mask = np.asarray(window_result["mask"], dtype=np.uint8)
        cap_box = cv2.boundingRect(cap_corners)
        pad_angle, left_pads, right_pads = estimate_pad_rotation(
            down_crop,
            thresh=int(split.get("padGrayThresh", 210)),
            search_width=int(split.get("padSearchRange", 200)),
            pad_min_num=int(split.get("padMinNum", 6)),
            pad_w=int(split.get("padW", 36)),
            pad_h=int(split.get("padH", 36)),
            pad_range=int(split.get("padRange", 10)),
        )
        if pad_angle is None:
            raise ValueError("20um 下图未找到左右 Pad 列，无法确定 CAP 允许范围。")
        pad_centers = (
            int(np.mean([x + width / 2 for x, _, width, _ in left_pads])),
            int(np.mean([x + width / 2 for x, _, width, _ in right_pads])),
        )
        _log(messages, callback, "20um：使用成对裁剪图计算窗片和 Pad 位置")
    elif (
        crop_output is not None
        and crop_output.cap_image is not None
        and crop_output.cmos_image is not None
        and crop_output.cap_box is not None
        and crop_output.cap_mask is not None
        and crop_output.cap_corners is not None
        and crop_output.pad_centers is not None
    ):
        up_crop = crop_output.cap_image.copy()
        down_crop = crop_output.cmos_image.copy()
        cap_box = crop_output.cap_box
        cap_mask = crop_output.cap_mask.copy()
        cap_corners = crop_output.cap_corners.copy()
        pad_centers = crop_output.pad_centers
        _log(messages, callback, "20um：使用操作页自动裁剪结果")
    else:
        if up_image is None:
            raise ValueError("20um 检测需要先完成上下层拼接或自动裁剪。")
        (
            up_crop,
            down_crop,
            cap_box,
            cap_mask,
            cap_corners,
            pad_centers,
            _,
            _,
            _,
            _,
        ) = _split_20um(
            up_image,
            image,
            split,
            callback,
            messages,
        )
    _check_cancelled(cancel_callback)
    run_cap = cropped_layer != "down"
    run_cmos = cropped_layer != "up"
    cap_image = cv2.bitwise_and(up_crop, up_crop, mask=cap_mask) if run_cap else None
    cmos_image = (
        cv2.bitwise_and(down_crop, down_crop, mask=cv2.bitwise_not(cap_mask))
        if run_cmos else None
    )
    cap_items: list[DetectionItem] = []
    cmos_items: list[DetectionItem] = []
    cap_regions: list[tuple[str, Rect]] = []
    cmos_regions: list[tuple[str, Rect]] = []
    offset_detected = False
    cmos_top_line: int | None = None
    if run_cmos:
        cap_top = int(split.get("capTopOffset", 200))
        cmos_top_line = cap_top
        first = find_first_pad(
            down_crop,
            thresh=int(split.get("padGrayThresh", 210)),
            pad_min_num=int(split.get("padMinNum", 6)),
            pad_w=int(split.get("padW", 36)),
            pad_h=int(split.get("padH", 36)),
            pad_range=int(split.get("padRange", 10)),
            search_width=int(split.get("padSearchRange", 200)),
        )
        if first is not None:
            estimated = first[1] - int(detection.get("padTopToCmosEdgePix", 190))
            if abs(estimated - cap_top) <= 120:
                cmos_top_line = estimated
    if run_cap:
        if pad_centers is None:
            _log(messages, callback, "20um：单上图缺少下图 Pad 位置，跳过 CAP 偏移检测")
        else:
            _log(messages, callback, "20um：执行 CAP 偏移检测")
            _check_cancelled(cancel_callback)
            left_pad, right_pad = pad_centers
            cmos_top = cmos_top_line
            allowed_x = left_pad + int(detection.get("capLeftOffsetThreshPix", 260))
            allowed_y = max(
                0,
                cmos_top - int(detection.get("capTopOffsetThreshPix", 260)),
            )
            allowed_right = right_pad - int(
                detection.get("capRightOffsetThreshPix", 260)
            )
            allowed_bottom = up_crop.shape[0] - int(
                detection.get("capBottomOffsetThreshPix", 260)
            )
            allowed = _clamp((
                allowed_x,
                allowed_y,
                allowed_right - allowed_x,
                allowed_bottom - allowed_y,
            ), up_crop.shape)
            cap_regions.append(("CAP 允许范围", allowed))
            _log(messages, callback, f"20um：CAP 偏移安全范围 {allowed}")
            if _intersection(cap_box, allowed) != cap_box:
                offset_detected = True
                cap_items.append(DetectionItem("cap_offset", (0, 0, up_crop.shape[1], up_crop.shape[0]), float(up_crop.shape[0] * up_crop.shape[1]), "CAP 偏移"))

    if run_cap and not offset_detected:
        _log(messages, callback, "20um：执行 CAP 翘曲检测")
        _check_cancelled(cancel_callback)
        gray_cap = cv2.cvtColor(cap_image, cv2.COLOR_BGR2GRAY)
        detect_range = int(detection.get("capWarpDetectRange", 100))
        center_x, center_y = gray_cap.shape[1] // 2, gray_cap.shape[0] // 2
        warp_roi = _clamp((center_x - detect_range, center_y - detect_range, detect_range * 2, detect_range * 2), gray_cap.shape)
        wx, wy, ww, wh = warp_roi
        if ww > 0 and wh > 0 and cv2.mean(gray_cap[wy:wy + wh, wx:wx + ww])[0] < int(detection.get("capWarpDetectThresh", 50)):
            cap_items.append(DetectionItem("cap_warp", cap_box, float(cap_box[2] * cap_box[3]), "CAP 翘曲"))

        _log(messages, callback, "20um：执行 CAP 成像区和非成像区检测")
        _check_cancelled(cancel_callback)
        canny1 = int(detection.get("imagingCannyThresh1", 20))
        canny2 = int(detection.get("imagingCannyThresh2", 100))
        binary = cv2.bitwise_or(cv2.Canny(gray_cap, canny1, canny2), cv2.threshold(gray_cap, int(detection.get("capGrayThresh", 100)), 255, cv2.THRESH_BINARY_INV)[1])
        debug_images["cap_union"] = binary.copy()
        kernel_size = max(1, int(detection.get("imagingCannyK", 5)))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8))
        debug_images["cap_closed"] = binary.copy()
        top_offset = int(split.get("capTopOffset", 200))
        imaging_left = int(int(detection.get("imagingLeft", 0)) / 2.5)
        imaging_top = int(int(detection.get("imagingTop", 0)) / 2.5) + top_offset
        imaging_right = up_crop.shape[1] - int(int(detection.get("imagingRight", 0)) / 2.5)
        imaging_bottom = up_crop.shape[0] - top_offset - int(int(detection.get("imagingBottom", 0)) / 2.5)
        imaging = _clamp((
            imaging_left, imaging_top,
            imaging_right - imaging_left, imaging_bottom - imaging_top,
        ), up_crop.shape)
        cap_regions.append(("CAP 成像区", imaging))
        model_items = _run_20um_deep_learning(
            up_crop,
            binary,
            cap_box,
            cap_mask,
            cap_corners,
            imaging,
            detection,
            messages,
            callback,
            cancel_callback,
        )
        cap_items.extend(model_items)
        cap_items = _filter_20um_depth_items(
            cap_items,
            gray_cap,
            detection,
        )

    if run_cmos and not offset_detected:
        _log(messages, callback, "20um：执行 CMOS 崩边、脏污和排除框检测")
        _check_cancelled(cancel_callback)
        cmos_offset = max(0, int(split.get("capTopOffset", 200)))
        if cmos_offset >= cmos_image.shape[0]:
            raise ValueError("20um CMOS 下图高度不足，无法去除顶部偏移。")
        cmos_crop = cmos_image[cmos_offset:]
        gray_cmos = cv2.cvtColor(cmos_crop, cv2.COLOR_BGR2GRAY)
        cmos_binary = cv2.threshold(gray_cmos, int(detection.get("cmosDetectGrayThresh", 100)), 255, cv2.THRESH_BINARY_INV)[1]
        safe = _clamp((int(detection.get("cmosSafeX", 0)), int(detection.get("cmosSafeY", 0)), int(detection.get("cmosSafeW", cmos_crop.shape[1])), int(detection.get("cmosSafeH", cmos_crop.shape[0]))), cmos_crop.shape)
        top = max(0, min(cmos_crop.shape[0], cmos_top_line - cmos_offset))
        safe = _intersection(safe, (0, top, cmos_crop.shape[1], cmos_crop.shape[0] - top))
        _log(messages, callback, f"20um：CMOS 上边界 y={cmos_top_line}，屏蔽上方区域")
        cmos_regions.append(("CMOS 安全区", (safe[0], safe[1] + cmos_offset, safe[2], safe[3])))
        mask = np.zeros_like(cmos_binary)
        sx, sy, sw, sh = safe
        if sw > 0 and sh > 0:
            mask[sy:sy + sh, sx:sx + sw] = 255
        if cap_box[2] > 0 and cap_box[3] > 0:
            adhesive = _clamp((cap_box[0] - int(detection.get("cmosAdhesiveDx0", 0)), cap_box[1] - cmos_offset - int(detection.get("cmosAdhesiveDy0", 0)), cap_box[2] + int(detection.get("cmosAdhesiveDx0", 0)) + int(detection.get("cmosAdhesiveDx1", 0)), cap_box[3] + int(detection.get("cmosAdhesiveDy0", 0)) + int(detection.get("cmosAdhesiveDy1", 0))), cmos_crop.shape)
            ax, ay, aw, ah = adhesive
            mask[ay:ay + ah, ax:ax + aw] = 0
        exclusions = detection.get("cmosExcludeBoxes", [])
        if not exclusions:
            exclusions = [{"x": detection.get("cmosExcludeX", 0), "y": detection.get("cmosExcludeY", 0), "w": detection.get("cmosExcludeW", 0), "h": detection.get("cmosExcludeH", 0)}]
        for value in exclusions:
            ex, ey, ew, eh = _rect(value)
            ex, ey, ew, eh = _clamp((ex, ey - cmos_offset, ew, eh), cmos_crop.shape)
            if ew > 0 and eh > 0:
                mask[ey:ey + eh, ex:ex + ew] = 0
                cmos_regions.append(("CMOS 排除框", (ex, ey + cmos_offset, ew, eh)))
        contours, _ = cv2.findContours(cv2.bitwise_and(cmos_binary, mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cmos_shrink = int(detection.get("cmosChipShrinkPix", 0))
        inner_safe = (sx + cmos_shrink, sy + cmos_shrink, sw - cmos_shrink * 2, sh - cmos_shrink * 2)
        chip_min = int(float(detection.get("cmosChipMinLengthUm", 0.0)) / 2.5)
        chip_max = int(float(detection.get("cmosChipMaxLengthUm", 0.0)) / 2.5)
        dirty_min = int(detection.get("cmosDirtyMinWidthPix", 0))
        for contour in contours:
            _check_cancelled(cancel_callback)
            box = cv2.boundingRect(contour)
            inter = _intersection(box, inner_safe)
            if inter[2] <= 0 or inter[3] <= 0:
                continue
            mapped_box = (box[0], box[1] + cmos_offset, box[2], box[3])
            if inter[2] * inter[3] != box[2] * box[3]:
                if max(box[2], box[3]) > chip_max and min(box[2], box[3]) > chip_min:
                    cmos_items.append(DetectionItem("cmos_chip", mapped_box, float(box[2] * box[3]), "CMOS 崩边"))
            elif min(box[2], box[3]) > dirty_min:
                cmos_items.append(DetectionItem("cmos_dirty", mapped_box, float(box[2] * box[3]), "CMOS 脏污"))
    _log(messages, callback, f"20um：上图 CAP 缺陷 {len(cap_items)} 个，下图 CMOS 缺陷 {len(cmos_items)} 个")
    if cropped_layer == "up":
        combined = up_crop.copy()
    elif cropped_layer == "down":
        combined = down_crop.copy()
    else:
        combined = cv2.add(
            cv2.bitwise_and(up_crop, up_crop, mask=cap_mask),
            cv2.bitwise_and(down_crop, down_crop, mask=cv2.bitwise_not(cap_mask)),
        )
    items = cap_items + cmos_items
    regions = cap_regions + cmos_regions
    source_image = combined.copy()
    if cmos_top_line is not None and 0 <= cmos_top_line < combined.shape[0]:
        cv2.line(
            combined,
            (0, cmos_top_line),
            (combined.shape[1] - 1, cmos_top_line),
            (255, 255, 255),
            2,
        )
    result_image = _draw_output(combined, items, regions)
    if len(cap_corners) >= 3:
        cv2.polylines(
            result_image,
            [cap_corners],
            True,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
    return DetectionOutput(
        result_image,
        items,
        regions,
        messages,
        source_image=source_image,
        debug_images=debug_images,
    )


def run_traditional_detection(
    image: np.ndarray,
    resolution_mode: str,
    params: dict[str, Any],
    *,
    up_image: np.ndarray | None = None,
    crop_output: AutoCropOutput | None = None,
    log_callback: LogCallback | None = None,
    cancel_callback: CancelCallback | None = None,
    cropped_pair: bool = False,
    cropped_layer: str | None = None,
) -> DetectionOutput:
    _check_cancelled(cancel_callback)
    if image is None or image.size == 0:
        raise ValueError("当前没有可检测图像。")
    if resolution_mode == "5um":
        output = _run_5um(image, params, log_callback)
    elif resolution_mode == "20um":
        output = _run_20um(
            image,
            up_image,
            params,
            log_callback,
            crop_output,
            cancel_callback,
            cropped_pair,
            cropped_layer,
        )
    else:
        raise ValueError(f"不支持的分辨率：{resolution_mode}")
    if output.source_image is None:
        output.source_image = image
    return output
