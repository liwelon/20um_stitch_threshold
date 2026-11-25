"""VisionLab 脚本库：对 v7_live 的 20um CAP 上图裁剪结果检测缺陷。

输入 image 是 wafer_cpp 裁剪、旋正后的上图，像素尺寸应与配方模板一致，
或在模板上方包含 capTopOffset 像素。输出为带缺陷框的 BGR 图像。
"""

import cv2
import numpy as np


# v7_live 的 TorchScript 三分类模型。VisionLab 所用 Python 须能导入 torch。
MODEL_PATH = r"D:\WaferStudio_20um\model\export.pt"
# v7_live 的模板；wafer_cpp 在加载它时取得 tempW0/tempH0，数据库中的 0 不是实际尺寸。
TEMPLATE_PATH = r"D:\WaferStudio_20um\chips\v7\temp.jpg"

# 配方 splitParams.capTopOffset：模板匹配框向上扩展的像素数。
CAP_TOP_OFFSET = 200
# 配方像素比例：每微米 0.4 像素。
PIX_PER_UM = 1 / 2.5
# 成像区四边距，单位微米；右、下值是距模板右、下边的距离。
IMAGING_LEFT_UM = 1100
IMAGING_TOP_UM = 900
IMAGING_RIGHT_UM = 1100
IMAGING_BOTTOM_UM = 2200
# 配方 detectNonImaging：只检测成像区。
DETECT_NON_IMAGING = False

# CAP 灰度阈值和 Canny 双阈值；两种二值结果取并集。
CAP_GRAY_THRESH = 15
IMAGING_CANNY_THRESH1 = 30
IMAGING_CANNY_THRESH2 = 100
# 闭运算的正方形核边长，单位像素。
IMAGING_CANNY_K = 5
# 候选轮廓外接框面积必须严格大于此像素数。
IMAGING_MIN_DEFECT_AREA = 100
# 轮廓圆度超过此值时直接归为点缺陷，不走分类模型。
POINT_CIRCULARITY = 0.85
# 窗片边缘向内收缩的像素数。
CAP_CHIP_SHRINK_PIX = 20
# 两个候选框横、纵方向都不超过此间隔时合并，单位像素。
MERGE_DISTANCE_PIX = 10
# 模型输入和候选方形裁剪的最小边长，单位像素。
CROP_SIZE = 128
# 模型类别顺序，必须与 wafer_cpp ResNetDetector 一致。
CLASS_NAMES = ("cap_chip", "cap_dirty", "cap_scratch")
# TorchScript 输入按 RGB 三通道归一化的均值和标准差。
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)

# 深浅分数取框内最暗 40% 像素的均值，再用 1 - mean / 255 计算。
DARK_SEARCH_RATIO = 0.4
# 点缺陷两组面积和深浅阈值；每组内同时满足，任一组满足即保留。
CAP_POINT_MIN_AREA1 = 10000
CAP_POINT_GRAY_THRESH1 = 0.6
CAP_POINT_MIN_AREA2 = 1000
CAP_POINT_GRAY_THRESH2 = 0.75
# 脏污缺陷两组面积和深浅阈值，判断方式与点缺陷相同。
CAP_DIRTY_MIN_AREA1 = 5000
CAP_DIRTY_GRAY_THRESH1 = 0.7
CAP_DIRTY_MIN_AREA2 = 3000
CAP_DIRTY_GRAY_THRESH2 = 0.85
# 划伤外接框长边、短边的像素阈值，均使用严格大于。
CAP_SCRATCH_MAX_LENGTH = 30
CAP_SCRATCH_MIN_LENGTH = 10
# 合格划伤数量必须严格超过此数才统一保留。
CAP_SCRATCH_COUNT = 0
# 崩边外接框长边和短边的阈值，单位微米；运行时转成像素。
CAP_CHIP_MAX_LENGTH_UM = 500
CAP_CHIP_MIN_LENGTH_UM = 300


def _note(*items):
    if callable(globals().get("log")):
        log(*items)


def _ratio_mean(gray):
    target = int(gray.size * DARK_SEARCH_RATIO)
    if target <= 0:
        return -1.0
    hist = np.bincount(gray.reshape(-1), minlength=256)
    remaining = target
    gray_sum = 0
    for value, count in enumerate(hist):
        used = min(remaining, int(count))
        gray_sum += value * used
        remaining -= used
        if remaining == 0:
            break
    return 1.0 - gray_sum / target / 255.0


def _near(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = max(0, ax - (bx + bw), bx - (ax + aw))
    dy = max(0, ay - (by + bh), by - (ay + ah))
    return dx <= MERGE_DISTANCE_PIX and dy <= MERGE_DISTANCE_PIX


def _merge(rects):
    merged = [False] * len(rects)
    result = []
    for i, rect in enumerate(rects):
        if merged[i]:
            continue
        merged[i] = True
        current = rect
        changed = True
        while changed:
            changed = False
            for j in range(i + 1, len(rects)):
                if merged[j] or not _near(current, rects[j]):
                    continue
                x, y, w, h = current
                xx, yy, ww, hh = rects[j]
                left, top = min(x, xx), min(y, yy)
                current = (left, top, max(x + w, xx + ww) - left,
                           max(y + h, yy + hh) - top)
                merged[j] = True
                changed = True
        result.append(current)
    return result


def _crop_rect(rect, width, height):
    x, y, w, h = rect
    size = max(CROP_SIZE, w, h)
    if size > width or size > height:
        return rect
    cx, cy = x + w // 2, y + h // 2
    return (min(max(cx - size // 2, 0), width - size),
            min(max(cy - size // 2, 0), height - size), size, size)


def _append_if_valid(result, gray, rect, label):
    x, y, w, h = rect
    area = w * h
    if label in ("cap_point", "cap_dirty"):
        depth = _ratio_mean(gray[y:y + h, x:x + w])
        if label == "cap_point":
            accepted = ((area > CAP_POINT_MIN_AREA1 and depth > CAP_POINT_GRAY_THRESH1)
                        or (area > CAP_POINT_MIN_AREA2 and depth > CAP_POINT_GRAY_THRESH2))
        else:
            accepted = ((area > CAP_DIRTY_MIN_AREA1 and depth > CAP_DIRTY_GRAY_THRESH1)
                        or (area > CAP_DIRTY_MIN_AREA2 and depth > CAP_DIRTY_GRAY_THRESH2))
        if accepted:
            result.append((rect, label, depth))
    elif label == "cap_scratch":
        if max(w, h) > CAP_SCRATCH_MAX_LENGTH and min(w, h) > CAP_SCRATCH_MIN_LENGTH:
            result.append((rect, label, None))
    elif label == "cap_chip":
        long_min = int(CAP_CHIP_MAX_LENGTH_UM * PIX_PER_UM)
        short_min = int(CAP_CHIP_MIN_LENGTH_UM * PIX_PER_UM)
        if max(w, h) > long_min and min(w, h) > short_min:
            result.append((rect, label, None))


def process(image, seed=0):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("CAP 输入必须是非空 uint8 图像")
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        bgr = image
    else:
        raise ValueError("CAP 输入只支持灰度或 BGR 三通道图像")

    template = cv2.imread(TEMPLATE_PATH)
    if template is None:
        raise FileNotFoundError(f"无法读取 v7_live 模板：{TEMPLATE_PATH}")
    temp_h, temp_w = template.shape[:2]
    height, width = bgr.shape[:2]
    if width != temp_w or height not in (temp_h, temp_h + CAP_TOP_OFFSET):
        raise ValueError(
            f"CAP 裁剪图尺寸 {width}x{height} 与模板 {temp_w}x{temp_h} 不符；"
            "应输入裁剪后的上图，不要输入合成图或带标注的缩略图"
        )
    top_offset = CAP_TOP_OFFSET if height > temp_h else 0
    x0 = int(IMAGING_LEFT_UM * PIX_PER_UM)
    y0 = int(IMAGING_TOP_UM * PIX_PER_UM) + top_offset
    x1 = temp_w - int(IMAGING_RIGHT_UM * PIX_PER_UM)
    # 与 wafer_cpp 相同：下边界按模板高度计算，不增加上扩边量。
    y1 = temp_h - int(IMAGING_BOTTOM_UM * PIX_PER_UM)
    if x0 >= x1 or y0 >= y1:
        raise ValueError("v7_live 成像区为空，请核对模板和裁剪图")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, IMAGING_CANNY_THRESH1, IMAGING_CANNY_THRESH2)
    dark = cv2.threshold(gray, CAP_GRAY_THRESH, 255, cv2.THRESH_BINARY_INV)[1]
    binary = cv2.bitwise_or(edges, dark)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (IMAGING_CANNY_K, IMAGING_CANNY_K))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # 单独裁剪的上图以整幅图为窗片区域；v7_live 只检测更内侧的成像区。
    binary[:CAP_CHIP_SHRINK_PIX, :] = 0
    binary[height - CAP_CHIP_SHRINK_PIX:, :] = 0
    binary[:, :CAP_CHIP_SHRINK_PIX] = 0
    binary[:, width - CAP_CHIP_SHRINK_PIX:] = 0
    if not DETECT_NON_IMAGING:
        binary[:y0, :] = 0
        binary[y1:, :] = 0
        binary[:, :x0] = 0
        binary[:, x1:] = 0

    roi_x, roi_y = CAP_CHIP_SHRINK_PIX, CAP_CHIP_SHRINK_PIX
    roi = binary[roi_y:height - CAP_CHIP_SHRINK_PIX,
                 roi_x:width - CAP_CHIP_SHRINK_PIX]
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h <= IMAGING_MIN_DEFECT_AREA:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = (4.0 * np.pi * cv2.contourArea(contour) / perimeter ** 2
                       if perimeter > 1e-6 else 0.0)
        if circularity > POINT_CIRCULARITY:
            points.append(((x + roi_x, y + roi_y, w, h), "cap_point"))
        else:
            candidates.append((x, y, w, h))

    accepted = []
    for rect, label in points:
        _append_if_valid(accepted, gray, rect, label)
    merged = _merge(candidates)
    if merged:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("VisionLab 的 Python 环境缺少 PyTorch，无法运行 CAP 分类模型") from exc
        model = torch.jit.load(MODEL_PATH, map_location="cpu")
        model.eval()
        mean = torch.tensor(RGB_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(RGB_STD, dtype=torch.float32).view(1, 3, 1, 1)
        with torch.no_grad():
            for rect in merged:
                cx, cy, cw, ch = _crop_rect(rect, roi.shape[1], roi.shape[0])
                patch = gray[roi_y + cy:roi_y + cy + ch,
                             roi_x + cx:roi_x + cx + cw]
                patch = cv2.resize(patch, (CROP_SIZE, CROP_SIZE))
                rgb = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
                tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0)
                tensor = (tensor.float().div(255.0) - mean) / std
                output = model(tensor)
                class_index = int(torch.softmax(output, dim=1).argmax(dim=1).item())
                label = CLASS_NAMES[class_index]
                x, y, w, h = rect
                if label == "cap_chip":
                    edge_limit = int(CAP_CHIP_MIN_LENGTH_UM * PIX_PER_UM)
                    is_chip = False
                    if x < edge_limit:
                        x -= CAP_CHIP_SHRINK_PIX
                        is_chip = True
                    elif x + w > roi.shape[1] - edge_limit:
                        w += CAP_CHIP_SHRINK_PIX
                        is_chip = True
                    if y < edge_limit:
                        y -= CAP_CHIP_SHRINK_PIX
                        is_chip = True
                    elif y + h > roi.shape[0] - edge_limit:
                        h += CAP_CHIP_SHRINK_PIX
                        is_chip = True
                    if not is_chip:
                        label = "cap_dirty"
                _append_if_valid(accepted, gray,
                                 (x + roi_x, y + roi_y, w, h), label)

    scratches = [item for item in accepted if item[1] == "cap_scratch"]
    if len(scratches) <= CAP_SCRATCH_COUNT:
        accepted = [item for item in accepted if item[1] != "cap_scratch"]
    result = bgr.copy()
    for rect, label, depth in accepted:
        x, y, w, h = rect
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(result, label, (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        _note(label, f"x={x} y={y} w={w} h={h}",
              "" if depth is None else f"depth={depth:.3f}")
    _note(f"CAP candidates={len(points) + len(merged)}, defects={len(accepted)}")
    return result
