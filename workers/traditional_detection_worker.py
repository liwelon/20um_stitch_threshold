from __future__ import annotations

from threading import Event
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, Signal

from core.image_utils import IMAGE_EXTENSIONS, imread_unicode, natural_sort_key
from core.traditional_detection import (
    AutoCropOutput,
    DetectionCancelled,
    DetectionOutput,
    run_traditional_detection,
)
from workers.stitch_worker import load_stitched_batch, stitched_batch_count


class TraditionalDetectionWorker(QObject):
    log = Signal(str)
    finished = Signal(object)
    batch_finished = Signal(object, int, int)
    image_finished = Signal(object, str, int, int)
    all_finished = Signal(int, int)
    stopped = Signal(int)
    failed = Signal(str)
    pair_started = Signal(int)
    pair_finished = Signal(object, int)
    pair_failed = Signal(int, str)

    def __init__(
        self,
        image: np.ndarray | None,
        resolution_mode: str,
        params: dict[str, Any],
        up_image: np.ndarray | None = None,
        crop_output: AutoCropOutput | None = None,
        folder: str = "",
        machine_id: str = "",
        detect_all: bool = False,
        crop_pair: tuple[str | None, str | None] | None = None,
        stitched_pairs: list[tuple[int, str, str]] | None = None,
    ) -> None:
        super().__init__()
        self._image = None if image is None else image.copy()
        self._resolution_mode = resolution_mode
        self._params = params
        self._up_image = None if up_image is None else up_image.copy()
        self._crop_output = crop_output
        self._folder = folder
        self._machine_id = machine_id
        self._detect_all = detect_all
        self._crop_pair = crop_pair
        self._stitched_pairs = list(stitched_pairs) if stitched_pairs is not None else None
        self._stop_event = Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        completed_count = 0
        try:
            image = self._image
            up_image = self._up_image
            if self._resolution_mode == "20um":
                if self._stitched_pairs is not None:
                    for pair_id, up_path, down_path in self._stitched_pairs:
                        if self._stop_event.is_set():
                            self.stopped.emit(completed_count)
                            return
                        self.pair_started.emit(pair_id)
                        try:
                            up_image = imread_unicode(up_path)
                            image = imread_unicode(down_path)
                            if up_image is None or image is None:
                                raise ValueError("无法读取拼接图对，请检查上下图路径。")
                            if up_image.shape != image.shape:
                                raise ValueError("上下层拼接图的尺寸及通道数必须一致。")
                            self.log.emit(f"20um：拼接图对 {pair_id}，上图 {up_path}，下图 {down_path}")
                            output = run_traditional_detection(
                                image,
                                self._resolution_mode,
                                self._params,
                                up_image=up_image,
                                log_callback=self.log.emit,
                                cancel_callback=self._stop_event.is_set,
                            )
                        except (DetectionCancelled, InterruptedError):
                            raise
                        except Exception as exc:
                            self.pair_failed.emit(pair_id, str(exc))
                            continue
                        completed_count += 1
                        self.pair_finished.emit(output, pair_id)
                    if self._stop_event.is_set():
                        self.stopped.emit(completed_count)
                    else:
                        self.all_finished.emit(completed_count, 0)
                    return
                if self._crop_pair is not None:
                    up_path, down_path = self._crop_pair
                    if not up_path and not down_path:
                        raise ValueError("20um 请至少选择一张已裁剪图。")
                    up_image = imread_unicode(up_path) if up_path else None
                    down_image = imread_unicode(down_path) if down_path else None
                    if up_path and up_image is None:
                        raise ValueError(f"无法读取 20um 上图：{up_path}")
                    if down_path and down_image is None:
                        raise ValueError(f"无法读取 20um 下图：{down_path}")
                    image = down_image if down_image is not None else up_image
                    cropped_layer = (
                        "up" if down_image is None else "down" if up_image is None else None
                    )
                    self.log.emit(
                        "20um：已读取已裁剪上图和下图"
                        if cropped_layer is None
                        else f"20um：已读取已裁剪{'上图' if cropped_layer == 'up' else '下图'}"
                    )
                    output = run_traditional_detection(
                        image,
                        self._resolution_mode,
                        self._params,
                        up_image=up_image,
                        log_callback=self.log.emit,
                        cancel_callback=self._stop_event.is_set,
                        cropped_pair=cropped_layer is None,
                        cropped_layer=cropped_layer,
                    )
                    self.finished.emit(output)
                    return
                batch_count, ignored_count = stitched_batch_count(
                    self._folder,
                    self._machine_id,
                    self._resolution_mode,
                )
                run_count = batch_count if self._detect_all else 1
                for batch_index in range(run_count):
                    if self._stop_event.is_set():
                        self.stopped.emit(completed_count)
                        return
                    up_image, image = load_stitched_batch(
                        self._folder,
                        self._machine_id,
                        self._resolution_mode,
                        batch_index,
                        require_exact=False,
                        stop_requested=self._stop_event.is_set,
                    )
                    if up_image is None:
                        raise ValueError("20um 机台配方必须生成上下两层图像。")
                    self.log.emit(
                        f"20um：第 {batch_index + 1}/{run_count} 批已生成上图和下图"
                    )
                    output = run_traditional_detection(
                        image,
                        self._resolution_mode,
                        self._params,
                        up_image=up_image,
                        crop_output=None,
                        log_callback=self.log.emit,
                        cancel_callback=self._stop_event.is_set,
                    )
                    completed_count += 1
                    if self._detect_all:
                        self.batch_finished.emit(
                            output,
                            batch_index,
                            run_count,
                        )
                    else:
                        self.finished.emit(output)
                        return
                self.all_finished.emit(completed_count, ignored_count)
                return
            if self._detect_all or (image is None and self._folder):
                paths = sorted(
                    (
                        path for path in Path(self._folder).iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                    ),
                    key=natural_sort_key,
                )
                if not paths:
                    raise ValueError("裁剪图文件夹内没有可检测图片。")
                run_paths = paths if self._detect_all else paths[:1]
                for image_index, path in enumerate(run_paths):
                    if self._stop_event.is_set():
                        self.stopped.emit(completed_count)
                        return
                    image = imread_unicode(str(path))
                    if image is None:
                        raise ValueError(f"无法读取图片：{path}")
                    output = run_traditional_detection(
                        image,
                        self._resolution_mode,
                        self._params,
                        log_callback=self.log.emit,
                    )
                    completed_count += 1
                    if self._detect_all:
                        self.image_finished.emit(
                            output, path.name, image_index, len(paths)
                        )
                    else:
                        self.log.emit(f"5um：单张检测 {path.name}")
                        self.finished.emit(output)
                        return
                self.all_finished.emit(completed_count, 0)
                return
            if image is None:
                raise ValueError("当前没有可检测图像。")
            output: DetectionOutput = run_traditional_detection(
                image,
                self._resolution_mode,
                self._params,
                up_image=up_image,
                crop_output=self._crop_output,
                log_callback=self.log.emit,
            )
            self.finished.emit(output)
        except (DetectionCancelled, InterruptedError):
            self.stopped.emit(completed_count)
        except Exception as exc:  # noqa: BLE001 - 传到 UI 提示
            self.failed.emit(str(exc))
