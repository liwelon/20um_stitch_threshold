# -*- coding: utf-8 -*-
"""左侧/右侧竖列白色方块 pad 检测与旋正（对齐 FirstPadFinder / PadFinder）。"""
from __future__ import annotations

import cv2
import numpy as np

DEFAULT_THRESH = 210
PAD_SEARCH_RANGE = 200
COL_TOL = 80
PAD_MIN_NUM = 6
PAD_W, PAD_H = 36, 36
PAD_RANGE = 10
MIN_FILL = 0.3

PadRect = tuple[int, int, int, int]


def find_pad_column(
    img: np.ndarray,
    thresh: int = DEFAULT_THRESH,
    search_width: int | None = None,
    pad_min_num: int = PAD_MIN_NUM,
    pad_w: int = PAD_W,
    pad_h: int = PAD_H,
    pad_range: int = PAD_RANGE,
) -> list[PadRect]:
    """找最密 pad 列，返回 [(x, y, w, h), ...]（可能为空）。

    search_width=None 时搜索整图（左右条带可直接传入 ROI）；
    否则仅搜索左侧 search_width 像素。
    """
    if img is None or img.size == 0 or pad_min_num <= 0:
        return []

    min_w, max_w = pad_w - pad_range, pad_w + pad_range
    min_h, max_h = pad_h - pad_range, pad_h + pad_range
    if min_w <= 0 or min_h <= 0 or max_w < min_w or max_h < min_h:
        return []

    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if search_width is not None:
        sw = min(search_width, gray.shape[1])
        if sw <= 0:
            return []
        gray = gray[:, :sw]

    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates: list[PadRect] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if not (min_w <= w <= max_w and min_h <= h <= max_h):
            continue
        if cv2.contourArea(cnt) / (w * h) < MIN_FILL:
            continue
        candidates.append((x, y, w, h))

    if len(candidates) < pad_min_num:
        return []

    candidates.sort(key=lambda r: (r[0], r[1]))

    columns: list[list[PadRect]] = []
    for rect in candidates:
        if not columns or rect[0] - columns[-1][-1][0] > COL_TOL:
            columns.append([rect])
        else:
            columns[-1].append(rect)

    column = max(columns, key=len)
    if len(column) < pad_min_num:
        return []
    return column


def find_first_pad(
    img: np.ndarray,
    thresh: int = DEFAULT_THRESH,
    pad_min_num: int = PAD_MIN_NUM,
    pad_w: int = PAD_W,
    pad_h: int = PAD_H,
    pad_range: int = PAD_RANGE,
    search_width: int = PAD_SEARCH_RANGE,
) -> PadRect | None:
    """对一张灰度/彩色图找 first pad，返回 (x, y, w, h) 或 None。"""
    column = find_pad_column(
        img,
        thresh,
        search_width=search_width,
        pad_min_num=pad_min_num,
        pad_w=pad_w,
        pad_h=pad_h,
        pad_range=pad_range,
    )
    if not column:
        return None
    return min(column, key=lambda r: r[1])


def _centers_of(column: list[PadRect]) -> list[tuple[int, int]]:
    return [(x + w // 2, y + h // 2) for x, y, w, h in column]


def calc_pad_angle(centers: list[tuple[int, int]]) -> float:
    """由 pad 中心点拟合直线，返回相对竖直方向的倾角（度，对齐 C++ PadFinder）。"""
    pts = np.asarray(centers, dtype=np.float32).reshape(-1, 1, 2)
    line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, _x0, _y0 = [float(v) for v in line.reshape(-1)]
    angle = float(np.degrees(np.arctan2(vy, vx)))
    a = angle % 180.0
    if a < 0:
        a += 180.0
    return a - 90.0


def estimate_pad_rotation(
    img: np.ndarray,
    thresh: int = DEFAULT_THRESH,
    search_width: int = PAD_SEARCH_RANGE,
    pad_min_num: int = PAD_MIN_NUM,
    pad_w: int = PAD_W,
    pad_h: int = PAD_H,
    pad_range: int = PAD_RANGE,
) -> tuple[float | None, list[PadRect], list[PadRect]]:
    """左右条带找 pad 列并估旋转角。

    返回 (angle_deg, left_pads, right_pads)；失败时 angle 为 None。
    """
    if img is None or img.size == 0:
        return None, [], []

    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    # 裁剪图可能较窄：条带不超过半幅，保证左右互不重叠
    sw = min(search_width, max(1, width // 2))
    if sw <= 0 or width < 2:
        return None, [], []

    left_roi = gray[:, :sw]
    left_pads = find_pad_column(
        left_roi,
        thresh,
        search_width=None,
        pad_min_num=pad_min_num,
        pad_w=pad_w,
        pad_h=pad_h,
        pad_range=pad_range,
    )

    start_right = width - sw
    right_roi = gray[:, start_right:]
    right_local = find_pad_column(
        right_roi,
        thresh,
        search_width=None,
        pad_min_num=pad_min_num,
        pad_w=pad_w,
        pad_h=pad_h,
        pad_range=pad_range,
    )
    right_pads = [
        (x + start_right, y, w, h) for x, y, w, h in right_local
    ]

    if len(left_pads) < pad_min_num or len(right_pads) < pad_min_num:
        return None, left_pads, right_pads

    left_angle = calc_pad_angle(_centers_of(left_pads))
    right_angle = calc_pad_angle(_centers_of(right_pads))
    angle = (left_angle + right_angle) / 2.0
    return angle, left_pads, right_pads
