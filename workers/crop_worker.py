from __future__ import annotations

from threading import Event
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from core.first_pad import find_pad_column
from core.image_utils import IMAGE_EXTENSIONS, imread_unicode, imwrite_unicode, natural_sort_key
from core.traditional_detection import (
    AutoCropOutput, _clamp, _rect, five_um_blind_circuit_regions,
    run_wafer_crop,
)
from workers.stitch_worker import load_stitched_batch


class CropWorker(QObject):
    log = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        image: np.ndarray | None,
        resolution_mode: str,
        params: dict[str, Any],
        up_image: np.ndarray | None = None,
        last_angle: float = 0.0,
        folder: str = "",
        machine_id: str = "",
    ) -> None:
        super().__init__()
        self._image = None if image is None else image.copy()
        self._resolution_mode = resolution_mode
        self._params = params
        self._up_image = None if up_image is None else up_image.copy()
        self._last_angle = last_angle
        self._folder = folder
        self._machine_id = machine_id

    def run(self) -> None:
        try:
            image = self._image
            up_image = self._up_image
            if self._resolution_mode == "20um":
                up_image, image = load_stitched_batch(
                    self._folder,
                    self._machine_id,
                    self._resolution_mode,
                    require_exact=False,
                )
                if up_image is None:
                    raise ValueError("20um 机台配方必须生成上下两层图像。")
                self.log.emit("20um：已从小图文件夹生成上图和下图")
            if image is None:
                raise ValueError("当前没有可裁剪图像。")
            output: AutoCropOutput = run_wafer_crop(
                image,
                self._resolution_mode,
                self._params,
                up_image=up_image,
                last_angle=self._last_angle,
                log_callback=self.log.emit,
            )
            self.finished.emit(output)
        except Exception as exc:  # noqa: BLE001 - 传到 UI 提示
            self.failed.emit(str(exc))


class BatchRegionCropWorker(QObject):
    progress = Signal(int, int, int)
    finished = Signal(int, int)
    stopped = Signal(int, int)
    failed = Signal(str)

    def __init__(
        self,
        folder: str,
        output_folder: str,
        params: dict[str, Any],
        recipe_name: str,
        configured_zones: bool = False,
    ) -> None:
        super().__init__()
        self._folder = folder
        self._output_folder = output_folder
        self._params = params
        self._recipe_name = "".join(
            "_" if char in '<>:"/\\|?*' or ord(char) < 32 else char
            for char in recipe_name
        ).strip(" .")
        if not self._recipe_name:
            raise ValueError("配方名称不能为空。")
        self._configured_zones = configured_zones
        self._stop_event = Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            source = Path(self._folder)
            destination = Path(self._output_folder)
            if not source.is_dir() or not destination.is_dir():
                raise ValueError("输入或输出文件夹无效。")
            paths = sorted(
                (
                    path for path in source.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                ),
                key=natural_sort_key,
            )
            if not paths:
                raise ValueError("裁剪图文件夹内没有可处理图片。")
            detection = self._params.get("DetectionParams", {})
            if not isinstance(detection, dict):
                raise ValueError("当前 5um 配方缺少检测参数。")

            for region_name in ("blind", "circuit", "outer", "dummy"):
                (destination / region_name).mkdir(exist_ok=True)

            targets: set[str] = set()
            saved_count = 0
            for image_index, path in enumerate(paths, start=1):
                if self._stop_event.is_set():
                    self.stopped.emit(image_index - 1, saved_count)
                    return
                image = imread_unicode(str(path))
                if image is None:
                    raise ValueError(f"无法读取图片：{path}")
                regions = []
                coords = detection.get("regionCoords", [])
                dummy_region = None
                if isinstance(coords, list) and len(coords) > 3:
                    dummy_roi = _clamp(_rect(coords[3]), image.shape)
                    dx, dy, dw, dh = dummy_roi
                    if dw > 0 and dh > 0:
                        dummy_target = destination / "dummy" / (
                            f"{path.name}_{self._recipe_name}_dummy_"
                            f"x{dx}_y{dy}_w{dw}_h{dh}.png"
                        )
                        target_key = str(dummy_target).casefold()
                        if dummy_target.exists() or target_key in targets:
                            raise FileExistsError(f"输出文件已存在：{dummy_target}")
                        targets.add(target_key)
                        dummy_region = (dummy_roi, dummy_target)
                if self._configured_zones:
                    for region_name, list_key, old_key in (
                        ("blind", "blindDetectionZones", "blindDetectionZone"),
                        ("circuit", "circuitDetectionZones", "circuitDetectionZone"),
                        ("outer", "outerDetectionZones", None),
                    ):
                        configured = detection.get(list_key)
                        if isinstance(configured, list):
                            values = configured
                        elif old_key is not None and isinstance(detection.get(old_key), dict):
                            values = [detection[old_key]]
                        else:
                            values = []
                        for index, value in enumerate(values, start=1):
                            roi = _clamp(_rect(value), image.shape)
                            x, y, width, height = roi
                            if width <= 0 or height <= 0:
                                continue
                            target = destination / region_name / (
                                f"{path.name}_{self._recipe_name}_configured_{region_name}_{index}_"
                                f"x{x}_y{y}_w{width}_h{height}.png"
                            )
                            target_key = str(target).casefold()
                            if target.exists() or target_key in targets:
                                raise FileExistsError(f"输出文件已存在：{target}")
                            targets.add(target_key)
                            regions.append((roi, target))
                    if not regions:
                        raise ValueError(f"{path.name} 的配方没有有效的盲元、电路或外围区域框。")
                    if dummy_region is not None:
                        regions.append(dummy_region)
                    for (x, y, width, height), target in regions:
                        crop = image[y:y + height, x:x + width]
                        if target.exists():
                            raise FileExistsError(f"输出文件已存在：{target}")
                        if not imwrite_unicode(str(target), crop):
                            raise OSError(f"无法保存裁剪图：{target}")
                        saved_count += 1
                    self.progress.emit(image_index, len(paths), saved_count)
                    continue
                for name, roi in five_um_blind_circuit_regions(image.shape, detection):
                    x, y, width, height = roi
                    if width <= 0 or height <= 0:
                        raise ValueError(f"{path.name} 的{name}无效。")
                    region_name = "blind" if name == "盲元区" else "circuit"
                    target = destination / region_name / (
                        f"{path.name}_{self._recipe_name}_{region_name}_"
                        f"x{x}_y{y}_w{width}_h{height}.png"
                    )
                    target_key = str(target).casefold()
                    if target.exists() or target_key in targets:
                        raise FileExistsError(f"输出文件已存在：{target}")
                    targets.add(target_key)
                    regions.append((roi, target))

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
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
                if not left or not right_local:
                    raise ValueError(f"{path.name} 未找到左右两侧 Pad，无法裁剪外围区。")
                left_edge = int(np.mean([px + pw for px, _, pw, _ in left]))
                right_edge = int(np.mean([px + gray.shape[1] - search for px, _, _, _ in right_local]))
                safe_line = int(detection.get("safeLine", 55)) + int(detection.get("offset", 10))
                outer_roi = (left_edge, safe_line, right_edge - left_edge, gray.shape[0] - safe_line * 2)
                if (
                    left_edge < 0 or right_edge > gray.shape[1]
                    or outer_roi[2] <= 0 or safe_line < 0 or outer_roi[3] <= 0
                ):
                    raise ValueError(f"{path.name} 的外围区坐标无效。")
                if not isinstance(coords, list) or len(coords) <= 4:
                    raise ValueError(f"{path.name} 的配方缺少检测区域 Detection。")
                detection_roi = _rect(coords[4])
                x, y, width, height = outer_roi
                dx, dy, dw, dh = detection_roi
                if (
                    dw <= 0 or dh <= 0 or dx <= x or dy <= y
                    or dx + dw >= x + width or dy + dh >= y + height
                ):
                    raise ValueError(f"{path.name} 的 Detection 区域未完整位于外围区内部，无法切分四张图片。")
                outer_parts = (
                    ("top", (x, y, width, dy - y)),
                    ("bottom", (x, dy + dh, width, y + height - dy - dh)),
                    ("left", (x, dy, dx - x, dh)),
                    ("right", (dx + dw, dy, x + width - dx - dw, dh)),
                )
                for part_name, roi in outer_parts:
                    px, py, pw, ph = roi
                    target = destination / "outer" / (
                        f"{path.name}_{self._recipe_name}_outer_{part_name}_"
                        f"x{px}_y{py}_w{pw}_h{ph}.png"
                    )
                    target_key = str(target).casefold()
                    if target.exists() or target_key in targets:
                        raise FileExistsError(f"输出文件已存在：{target}")
                    targets.add(target_key)
                    regions.append((roi, target))
                if dummy_region is not None:
                    regions.append(dummy_region)
                for (x, y, width, height), target in regions:
                    crop = image[y:y + height, x:x + width]
                    if target.exists():
                        raise FileExistsError(f"输出文件已存在：{target}")
                    if not imwrite_unicode(str(target), crop):
                        raise OSError(f"无法保存裁剪图：{target}")
                    saved_count += 1
                self.progress.emit(image_index, len(paths), saved_count)
            self.finished.emit(len(paths), saved_count)
        except Exception as exc:  # noqa: BLE001 - 传到 UI 提示
            self.failed.emit(str(exc))
