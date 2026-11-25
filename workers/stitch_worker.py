from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QObject, Signal

from core.image_utils import IMAGE_EXTENSIONS, imread_unicode, natural_sort_key
from core.stitching.image_jointer import ImageJointer
from core.stitching.params import get_joint_params


def _folder_image_paths(folder_path: str) -> list[Path]:
    folder = Path(folder_path)
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=natural_sort_key,
    )


def stitched_batch_count(
    folder_path: str,
    machine_id: str,
    mode: str,
) -> tuple[int, int]:
    params = get_joint_params(machine_id, mode)
    image_count = len(_folder_image_paths(folder_path))
    if image_count < params.joint_count:
        raise ValueError(
            f"文件夹内图片数为 {image_count}，"
            f"机台 {machine_id} 至少需要 {params.joint_count} 张"
        )
    return image_count // params.joint_count, image_count % params.joint_count


def load_stitched_batch(
    folder_path: str,
    machine_id: str,
    mode: str,
    batch_index: int = 0,
    *,
    require_exact: bool = True,
    stop_requested: Callable[[], bool] | None = None,
) -> tuple[np.ndarray | None, np.ndarray]:
    params = get_joint_params(machine_id, mode)
    paths = _folder_image_paths(folder_path)
    if require_exact and len(paths) != params.joint_count:
        raise ValueError(
            f"文件夹内图片数为 {len(paths)}，"
            f"机台 {machine_id} 检测需要 {params.joint_count} 张"
        )
    start = batch_index * params.joint_count
    batch_paths = paths[start:start + params.joint_count]
    if len(batch_paths) != params.joint_count:
        raise ValueError(
            f"第 {batch_index + 1} 批图片数为 {len(batch_paths)}，"
            f"机台 {machine_id} 检测需要 {params.joint_count} 张"
        )
    images: list[np.ndarray] = []
    for path in batch_paths:
        if stop_requested is not None and stop_requested():
            raise InterruptedError("检测已停止")
        image = imread_unicode(str(path))
        if image is None:
            raise ValueError(f"无法读取图片：{path}")
        images.append(image)
    jointer = ImageJointer(params)
    if params.layers == 1:
        return None, jointer.joint_single_layer(images)
    return jointer.joint_up_down(images)


class StitchWorker(QObject):
    batch_finished = Signal(object, object, int)
    finished = Signal(int)
    failed = Signal(str)

    def __init__(
        self,
        folder: str,
        machine_id: str,
        stitch_all: bool,
        mode: str,
    ) -> None:
        super().__init__()
        self._folder = folder
        self._machine_id = machine_id
        self._stitch_all = stitch_all
        self._mode = mode

    def run(self) -> None:
        try:
            params = get_joint_params(self._machine_id, self._mode)
            paths = _folder_image_paths(self._folder)
            if len(paths) < params.joint_count:
                raise ValueError(
                    f"文件夹内图片数为 {len(paths)}，"
                    f"机台 {self._machine_id} 至少需要 {params.joint_count} 张"
                )
            jointer = ImageJointer(params)
            batch_count = len(paths) // params.joint_count if self._stitch_all else 1
            for batch_index in range(batch_count):
                batch_start = time.perf_counter()
                start = batch_index * params.joint_count
                batch_paths = paths[start : start + params.joint_count]
                images = []
                for path in batch_paths:
                    image = imread_unicode(str(path))
                    if image is None:
                        raise ValueError(f"无法读取图片：{path}")
                    images.append(image)

                if params.layers == 1:
                    up_img = None
                    stitched_img = jointer.joint_single_layer(images)
                else:
                    up_img, stitched_img = jointer.joint_up_down(images)
                elapsed = time.perf_counter() - batch_start
                print(f"第 {batch_index + 1} 批拼接耗时 {elapsed:.3f} 秒")
                self.batch_finished.emit(up_img, stitched_img, batch_index)

            ignored_count = len(paths) % params.joint_count if self._stitch_all else 0
            self.finished.emit(ignored_count)
        except Exception as exc:  # noqa: BLE001 - 传到 UI 提示
            self.failed.emit(str(exc))
