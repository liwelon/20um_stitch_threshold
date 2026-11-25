from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtGui import QImage


IMAGE_EXTENSIONS = {".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def natural_sort_key(path: Path) -> list:
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def imread_unicode(file_path: str) -> np.ndarray | None:
    data = np.fromfile(file_path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(file_path: str, image: np.ndarray) -> bool:
    suffix = Path(file_path).suffix or ".png"
    ok, buffer = cv2.imencode(suffix, image)
    if not ok:
        return False
    buffer.tofile(file_path)
    return True


def numpy_to_qimage(image: np.ndarray) -> QImage:
    if image.ndim == 2:
        height, width = image.shape
        return QImage(
            image.data,
            width,
            height,
            image.strides[0],
            QImage.Format.Format_Grayscale8,
        ).copy()

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = rgb.shape
    return QImage(
        rgb.data,
        width,
        height,
        rgb.strides[0],
        QImage.Format.Format_RGB888,
    ).copy()


def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 1e-9:
        return image.copy()
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos_a = abs(matrix[0, 0])
    sin_a = abs(matrix[0, 1])
    new_w = int(height * sin_a + width * cos_a)
    new_h = int(height * cos_a + width * sin_a)
    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]
    return cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
