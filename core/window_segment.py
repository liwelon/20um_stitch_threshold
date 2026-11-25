"""窗片分割算法的 Python 验证程序。

建议按下面的顺序阅读：

1. 参数与常量：控制粗定位范围、平滑强度和边缘搜索范围。
2. `_center_run`：从一维二值剖面中取得穿过图像中心的窗片区间。
3. `_robust_line`：从包含污点和崩边的候选点中拟合稳定直线。
4. `_fit_horizontal_edge`、`_fit_vertical_edge`：寻找四条真实边缘。
5. `_intersect`：由四条边计算四个角点。
6. `segment_window`：串联上述步骤，生成角点、掩膜和裁剪图。
7. `main`：批量读取测试图片并保存叠加框。

算法分成“粗定位”和“精定位”两层。粗定位只需要找到窗片大致范围，
精定位再利用图像梯度拟合允许轻微倾斜的四条直线。这样既能减少搜索量，
也能避免直接从整幅图中寻找边缘灯带、污点等无关结构。
"""

import cv2
import numpy as np


# =============================================================================
# 第 1 步：配置批量验证路径和算法常量
# =============================================================================

# 粗定位参数。
# 拍摄方式保证窗片经过图像中心，所以可以从中心出发寻找连续窗片区域，
# 不会误选图像两侧同样较亮的灯带。
PROFILE_MARGIN_RATIO = 0.20  # 构造剖面时忽略两端各 10% 的区域
PROFILE_SMOOTH_KERNEL = 51  # 一维高斯平滑核，必须是正奇数
MIN_WINDOW_RATIO = 0.60  # 窗片宽高至少占原图对应尺寸的 60%

# 精定位参数。
# 粗定位给出的边界可能有几十像素误差，因此还要在附近寻找梯度峰值。
EDGE_SEARCH_RADIUS = 100  # 在粗边界附近最多搜索 100 像素
EDGE_SAMPLE_STEP = 64  # 沿边缘每隔 64 像素采集一个候选点
EDGE_TRIM_RATIO = 0.05  # 跳过边缘两端各 5%，减少四角破损干扰
# 上边第一次只向下搜索时：列内出现双峰且间距足够大，取更靠内侧的峰，
# 避免中段灯带假边主导拟合、右端真实边被当成离群点。
EDGE_INNER_PEAK_RATIO = 0.50  # 局部峰至少达到列最大梯度的该比例才参与竞争
EDGE_INNER_PEAK_SEP = 12  # 最外/最内强峰间距达到该像素才启用取靠内峰


# =============================================================================
# 调试可视化配置（不属于算法本体，DEBUG_MODE 关闭时零副作用）
# =============================================================================

_DEBUG_STATE = {}

_EDGE_DEBUG_COLORS = {
    "top": (0, 0, 255),          # 红：上边第一次拟合
    "upper_top": (0, 165, 255),  # 橙：上边第二次拟合（允许向上搜索）
    "bottom": (255, 0, 0),       # 蓝：下边
    "left": (0, 255, 0),         # 绿：左边第一次拟合
    "inner_left": (0, 200, 100), # 深绿：左边第二次拟合（允许向外搜索）
    "right": (0, 255, 255),      # 黄：右边第一次拟合
    "inner_right": (0, 180, 200),# 暗黄：右边第二次拟合（允许向外搜索）
}


def _debug_active():
    return False


def _debug_save(name, image):
    return


def _debug_plot_profile(name, profile):
    """把一维剖面绘成折线图，绿色竖线标出图像中心位置。"""
    if not _debug_active():
        return
    profile_f = np.asarray(profile, dtype=np.float32).ravel()
    length = profile_f.size
    height = 240
    canvas = np.full((height, length, 3), 255, dtype=np.uint8)
    p_min = float(profile_f.min())
    p_max = float(profile_f.max())
    if p_max > p_min:
        norm = (profile_f - p_min) / (p_max - p_min)
    else:
        norm = np.zeros_like(profile_f)
    ys = (height - 20 - norm * (height - 40)).astype(np.int32)
    xs = np.arange(length, dtype=np.int32)
    points = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(canvas, [points], False, (60, 60, 60), 1)
    center_x = length // 2
    cv2.line(canvas, (center_x, 0), (center_x, height), (0, 180, 0), 1)
    _debug_save(name, canvas)


def _debug_visualize_gradient(name, gradient):
    """将浮点梯度图归一化后转为伪彩色热力图。"""
    if not _debug_active():
        return
    normalized = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
    _debug_save(name, heatmap)


def _debug_record_edge(name, kind, xs, ys, line):
    """记录一次边缘拟合的采样点和结果，segment_window 结束时统一叠加输出。"""
    if not _debug_active():
        return
    _DEBUG_STATE.setdefault("edges", []).append({
        "name": name,
        "kind": kind,
        "xs": list(xs),
        "ys": list(ys),
        "line": line,
    })


def _debug_edges_overlay(image):
    """把已记录的采样点和拟合直线一起叠加到原图。拟合失败的直线只画点。"""
    if not _debug_active():
        return
    edges = _DEBUG_STATE.get("edges", [])
    if not edges:
        return
    canvas = image.copy()
    height, width = canvas.shape[:2]
    for edge in edges:
        color = _EDGE_DEBUG_COLORS.get(edge["name"], (200, 200, 200))
        if edge["kind"] == "horizontal":
            for x, y in zip(edge["xs"], edge["ys"]):
                cv2.circle(canvas, (int(x), int(y)), 4, color, -1)
        else:
            for y, x in zip(edge["xs"], edge["ys"]):
                cv2.circle(canvas, (int(x), int(y)), 4, color, -1)
        line = edge["line"]
        if line is None:
            continue
        slope, intercept = line
        if edge["kind"] == "horizontal":
            x0, x1 = 0, width - 1
            y0 = int(round(slope * x0 + intercept))
            y1 = int(round(slope * x1 + intercept))
        else:
            y0, y1 = 0, height - 1
            x0 = int(round(slope * y0 + intercept))
            x1 = int(round(slope * y1 + intercept))
        cv2.line(canvas, (x0, y0), (x1, y1), color, 2)
    _debug_save("11_edges_overlay", canvas)


# =============================================================================
# 第 2 步：从一维二值剖面中取得窗片的粗范围
# =============================================================================


def _center_run(binary_profile):
    """返回穿过剖面中心的连续前景区间。

    参数 `binary_profile` 是一维布尔数组：True 表示窗片候选区域，False
    表示背景。横向剖面返回左右边界，纵向剖面返回上下边界。

    当前拍摄方式保证窗片经过图像中心，所以优先从数组中心向两侧扩展。
    如果中心刚好因为污点变成 False，就先找到距离中心最近的 True，再从
    那里扩展。整条剖面都没有前景时返回 None。
    """
    # ravel 把传入的一行、一列或普通列表统一整理成一维数组。
    binary_profile = np.asarray(binary_profile).ravel()
    center = len(binary_profile) // 2
    foreground = np.flatnonzero(binary_profile)
    if foreground.size == 0:
        return None

    # 中心落在背景时，改用离中心最近的前景点作为扩展起点。
    if not binary_profile[center]:
        center = int(foreground[np.argmin(np.abs(foreground - center))])

    # 向左或向上寻找连续区间的起点。
    start = center
    while start > 0 and binary_profile[start - 1]:
        start -= 1

    # 向右或向下寻找连续区间的终点。
    end = center
    while end + 1 < len(binary_profile) and binary_profile[end + 1]:
        end += 1

    return start, end


# =============================================================================
# 第 3 步：从带有离群点的候选点中拟合直线
# =============================================================================


def _least_squares_line(independent, dependent):
    independent = np.asarray(independent, dtype=np.float32)
    dependent = np.asarray(dependent, dtype=np.float32)
    mean_independent = np.mean(independent, dtype=np.float32)
    mean_dependent = np.mean(dependent, dtype=np.float32)
    independent_offset = independent - mean_independent
    dependent_offset = dependent - mean_dependent
    numerator = np.sum(
        independent_offset * dependent_offset,
        dtype=np.float32,
    )
    denominator = np.sum(
        independent_offset * independent_offset,
        dtype=np.float32,
    )
    if abs(float(denominator)) < 1e-12:
        return None
    slope = numerator / denominator
    intercept = mean_dependent - slope * mean_independent
    return float(slope), float(intercept)


def _robust_line(independent, dependent):
    """拟合 `dependent = slope * independent + intercept`。

    横边拟合时，independent 是 x，dependent 是 y，直线形式为 y=kx+b。
    竖边拟合时，independent 是 y，dependent 是 x，直线形式为 x=ky+b。

    污点、划痕和崩边会制造错误候选点，直接最小二乘会被这些点拉偏。
    这里先枚举任意两点建立候选直线，找出 6 像素误差内支持点最多的
    直线，再反复执行最小二乘和离群点剔除。少于 10 个有效点时返回 None。
    成功时返回 `(slope, intercept, support_count)`。
    """
    # float32 足以表达图像坐标，也与 OpenCV 常用浮点类型一致。
    independent = np.asarray(independent, dtype=np.float32)
    dependent = np.asarray(dependent, dtype=np.float32)
    if independent.size < 10:
        return None

    # 第一阶段类似 RANSAC：枚举两点确定的直线，寻找最大一致点集合。
    keep = None
    best_count = 0
    best_median_residual = np.inf
    for first in range(independent.size - 1):
        for second in range(first + 1, independent.size):
            difference = independent[second] - independent[first]
            if abs(difference) < 1e-6:
                continue

            slope = (
                dependent[second] - dependent[first]
            ) / difference
            intercept = dependent[first] - slope * independent[first]
            # residual 是每个候选点到当前直线在 dependent 方向上的误差。
            residual = np.abs(
                dependent - (slope * independent + intercept)
            )
            candidate_keep = residual <= 6.0
            candidate_count = np.count_nonzero(candidate_keep)
            median_residual = np.median(residual[candidate_keep])

            if (
                candidate_count > best_count
                or (
                    candidate_count == best_count
                    and median_residual < best_median_residual
                )
            ):
                keep = candidate_keep
                best_count = candidate_count
                best_median_residual = median_residual

    # 十个支持点可以覆盖较长的边缘范围，避免由局部污点组成一条假直线。
    if keep is None or best_count < 10:
        return None

    # 第二阶段最多迭代四次，用当前内点做最小二乘，再更新内点集合。
    for _ in range(4):
        if np.count_nonzero(keep) < 10:
            return None

        slope, intercept = _least_squares_line(
            independent[keep], dependent[keep]
        )
        residual = np.abs(dependent - (slope * independent + intercept))
        median_residual = np.median(residual[keep])
        # 误差门限会适应当前数据，但不会小于 6 像素。
        limit = max(6.0, median_residual * 2.5)
        new_keep = residual <= limit

        # 内点集合不再变化，说明拟合已经收敛，可以提前结束。
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    support_count = int(np.count_nonzero(keep))
    if support_count < 10:
        return None
    line = _least_squares_line(independent[keep], dependent[keep])
    if line is None:
        return None
    # 第三个值是最终仍满足误差门限的支持点数，供上边替换时比较两条线。
    return line[0], line[1], support_count


# =============================================================================
# 第 4 步：在粗边界附近采样并拟合四条真实边缘
# =============================================================================


def _pick_horizontal_sample_y(column, start_y, prefer_inner):
    """从一列搜索窗中选出横边采样点的 y。

    默认取梯度最强位置。`prefer_inner=True` 时，若存在两个及以上足够强、
    且最外与最内峰间距不小于 `EDGE_INNER_PEAK_SEP` 的局部峰，则取 y 更大
    （更靠窗片内侧）的峰；否则仍回退到 argmax。
    """
    peak_index = int(np.argmax(column))
    if not prefer_inner or column.size < 3:
        return start_y + peak_index

    peak_value = float(column[peak_index])
    if peak_value <= 0:
        return start_y + peak_index

    threshold = peak_value * EDGE_INNER_PEAK_RATIO
    strong_peaks = []
    for index in range(column.size):
        value = float(column[index])
        if value < threshold:
            continue
        left_value = float(column[index - 1]) if index > 0 else value
        right_value = (
            float(column[index + 1]) if index + 1 < column.size else value
        )
        if value >= left_value and value >= right_value:
            strong_peaks.append(index)

    if (
        len(strong_peaks) >= 2
        and strong_peaks[-1] - strong_peaks[0] >= EDGE_INNER_PEAK_SEP
    ):
        return start_y + strong_peaks[-1]
    return start_y + peak_index


def _fit_horizontal_edge(
    gradient,
    coarse_y,
    left,
    right,
    upper_radius=EDGE_SEARCH_RADIUS,
    debug_name=None,
):
    """在粗横边附近逐列寻找最强梯度，拟合 `y = slope*x + intercept`。

    `gradient` 是二维梯度强度图，形状为 `(图像高度, 图像宽度)`。
    对每个采样列，在 `coarse_y` 附近找到梯度最大的 y，形成 `(x, y)`
    候选点，再交给 `_robust_line` 剔除崩边和污点造成的离群点。

    `upper_radius` 控制向上搜索距离，向下搜索距离固定为
    `EDGE_SEARCH_RADIUS`。传入 0 表示只从粗边界向下搜索；此时若列内出现
    灯带假边与真实边双峰，采样点改取更靠内侧的峰，避免中段假边带偏末尾。

    返回 `(line, support_count)`。拟合失败时 `line` 为 None，支持点数为 0。
    """
    height, width = gradient.shape
    # 不采集左右角附近的点，因为角部通常更容易崩边或被侧边梯度干扰。
    trim = int((right - left) * EDGE_TRIM_RATIO)
    start_x = max(0, left + trim)
    end_x = min(width - 1, right - trim)
    start_y = max(0, coarse_y - upper_radius)
    end_y = min(height - 1, coarse_y + EDGE_SEARCH_RADIUS)
    # 仅第一次上边（只向下搜索）启用双峰取靠内；upper_top 仍用 argmax，
    # 以保留崩边时向外拉回真实上边的能力。
    prefer_inner = upper_radius == 0

    x_points = []
    y_points = []
    for x in range(start_x, end_x + 1, EDGE_SAMPLE_STEP):
        # column 是当前 x 上的一小段竖直搜索区，而不是整列图像。
        column = gradient[start_y:end_y + 1, x]
        if column.size == 0:
            continue
        x_points.append(x)
        y_points.append(
            _pick_horizontal_sample_y(column, start_y, prefer_inner)
        )

    fitted = _robust_line(x_points, y_points)
    if fitted is None:
        line = None
        support_count = 0
    else:
        line = (fitted[0], fitted[1])
        support_count = fitted[2]
    if debug_name is not None:
        _debug_record_edge(debug_name, "horizontal", x_points, y_points, line)
    return line, support_count


def _fit_vertical_edge(
    gradient,
    coarse_x,
    top,
    bottom,
    left_radius=EDGE_SEARCH_RADIUS,
    right_radius=EDGE_SEARCH_RADIUS,
    debug_name=None,
):
    """在粗竖边附近逐行寻找最强梯度，拟合 `x = slope*y + intercept`。

    处理方式与 `_fit_horizontal_edge` 对称：沿 y 方向逐行采样，每行在
    `coarse_x` 左右寻找梯度最大的 x，最后用 `(y, x)` 拟合竖边。

    `left_radius` / `right_radius` 分别控制粗边界左右两侧的搜索距离。
    左边界传入 `left_radius=0` 只向窗片内部搜索，避免被外侧灯带干扰；
    右边界传入 `right_radius=0` 同理。
    """
    height, width = gradient.shape
    # 跳过上下角附近区域，避免横边和角部破损影响竖边拟合。
    trim = int((bottom - top) * EDGE_TRIM_RATIO)
    start_y = max(0, top + trim)
    end_y = min(height - 1, bottom - trim)
    start_x = max(0, coarse_x - left_radius)
    end_x = min(width - 1, coarse_x + right_radius)

    y_points = []
    x_points = []
    for y in range(start_y, end_y + 1, EDGE_SAMPLE_STEP):
        # row 是当前 y 上的一小段水平搜索区。
        row = gradient[y, start_x:end_x + 1]
        if row.size == 0:
            continue
        y_points.append(y)
        x_points.append(start_x + int(np.argmax(row)))

    fitted = _robust_line(y_points, x_points)
    line = None if fitted is None else (fitted[0], fitted[1])
    if debug_name is not None:
        _debug_record_edge(debug_name, "vertical", y_points, x_points, line)
    return line


# =============================================================================
# 第 5 步：由横边和竖边求角点
# =============================================================================


def _intersect(horizontal_line, vertical_line):
    """计算横线 `y=a*x+b` 与竖线 `x=c*y+d` 的交点。

    将第一条方程代入第二条即可解出 x 和 y。如果分母接近 0，说明两条
    线的组合无法稳定求交，返回 None，防止产生非常大的错误坐标。
    """
    horizontal_slope, horizontal_intercept = horizontal_line
    vertical_slope, vertical_intercept = vertical_line
    denominator = 1.0 - vertical_slope * horizontal_slope
    if abs(denominator) < 1e-6:
        return None

    x = (
        vertical_slope * horizontal_intercept + vertical_intercept
    ) / denominator
    y = horizontal_slope * x + horizontal_intercept
    return x, y


# =============================================================================
# 第 6 步：执行完整窗片分割流程
# =============================================================================


def segment_window(image):
    """检测窗片四个角点，并生成掩膜、掩膜图和裁剪图。

    输入是 OpenCV 读取的 BGR 三通道图像，形状为 `(高度, 宽度, 3)`。
    成功时返回包含 box、mask、window 和 crop 的字典；任何关键边界或
    交点无法稳定计算时返回 None。
    """
    height, width = image.shape[:2]
    blue, green, _ = cv2.split(image)

    # 6.1 构造颜色分数和亮度图。
    # 原图是 uint8，直接做 blue-green 会发生无符号溢出，所以先转 int16。
    # 窗片偏青，外围偏绿，B-G 能较稳定地区分窗片和左右背景。
    # 加 128 把负数平移到可显示范围，再裁剪并转回 uint8。
    color_score = np.clip(
        blue.astype(np.int16) - green.astype(np.int16) + 128,
        0,
        255,
    ).astype(np.uint8)
    # 每个像素取 B、G、R 最大值，含义接近 HSV 中的亮度 V。
    brightness = np.max(image, axis=2).astype(np.uint8)

    _debug_save("01_color_score", color_score)
    _debug_save("02_brightness", brightness)

    # 构造剖面时忽略两端区域，减少周围灯珠和边框的干扰。
    row_start = int(height * PROFILE_MARGIN_RATIO)
    row_end = int(height * (1.0 - PROFILE_MARGIN_RATIO))
    column_start = int(width * PROFILE_MARGIN_RATIO)
    column_end = int(width * (1.0 - PROFILE_MARGIN_RATIO))

    # 6.2 把二维图像压缩成两个一维剖面。
    # horizontal_profile 对每个 x 取中间行区域的中值，用于找左右边界。
    # vertical_profile 对每个 y 取中间列区域的亮度中值，用于找上下边界。
    horizontal_profile = np.median(
        color_score[row_start:row_end],
        axis=0,
    ).astype(np.uint8)
    vertical_profile = np.median(
        brightness[:, column_start:column_end],
        axis=1,
    ).astype(np.uint8)

    # 四边共用的梯度参考只在粗定位无效或直线拟合失败时接管。
    horizontal_edges = np.abs(
        cv2.Sobel(brightness, cv2.CV_32F, 0, 1, ksize=3)
    )
    color_vertical_edges = np.abs(
        cv2.Sobel(color_score, cv2.CV_32F, 1, 0, ksize=3)
    )
    brightness_vertical_edges = np.abs(
        cv2.Sobel(brightness, cv2.CV_32F, 1, 0, ksize=3)
    )
    vertical_edges = color_vertical_edges + brightness_vertical_edges
    horizontal_reference_profile = np.percentile(
        horizontal_edges[
            :, column_start:column_end:EDGE_SAMPLE_STEP
        ],
        80,
        axis=1,
    ).astype(np.float32)
    vertical_reference_profile = np.percentile(
        vertical_edges[row_start:row_end:EDGE_SAMPLE_STEP, :],
        80,
        axis=0,
    ).astype(np.float32)
    # 高斯平滑去掉剖面上的小尖峰。reshape 把一维数组变成 OpenCV
    # 所需的单行或单列图像，处理后再用 ravel 还原为一维数组。
    horizontal_profile = cv2.GaussianBlur(
        horizontal_profile.reshape(1, -1),
        (PROFILE_SMOOTH_KERNEL, 1),
        0,
    ).ravel()
    vertical_profile = cv2.GaussianBlur(
        vertical_profile.reshape(-1, 1),
        (1, PROFILE_SMOOTH_KERNEL),
        0,
    ).ravel()
    horizontal_reference_profile = cv2.GaussianBlur(
        horizontal_reference_profile.reshape(-1, 1),
        (1, PROFILE_SMOOTH_KERNEL),
        0,
    ).ravel()
    vertical_reference_profile = cv2.GaussianBlur(
        vertical_reference_profile.reshape(1, -1),
        (PROFILE_SMOOTH_KERNEL, 1),
        0,
    ).ravel()

    _debug_plot_profile("03_horizontal_profile", horizontal_profile)
    _debug_plot_profile("04_vertical_profile", vertical_profile)

    # Otsu 根据剖面的直方图自动选择阈值，不依赖固定亮度阈值。
    horizontal_otsu, horizontal_binary = cv2.threshold(
        horizontal_profile,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    vertical_otsu, vertical_binary = cv2.threshold(
        vertical_profile,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    _debug_plot_profile("05_horizontal_binary", horizontal_binary)
    _debug_plot_profile("06_vertical_binary", vertical_binary)

    # 6.3 从图像中心向两侧扩展，取得粗略的左右和上下范围。
    horizontal_run = _center_run(horizontal_binary > 0)
    vertical_run = _center_run(vertical_binary > 0)
    side_width = int(width * (1.0 - MIN_WINDOW_RATIO) / 2.0)
    side_height = int(height * (1.0 - MIN_WINDOW_RATIO) / 2.0)
    right_start = width - side_width
    bottom_start = height - side_height
    frame_margin = EDGE_SEARCH_RADIUS + EDGE_SEARCH_RADIUS // 2
    horizontal_reference_run = (
        frame_margin + int(np.argmax(
            vertical_reference_profile[frame_margin:side_width]
        )),
        right_start + int(np.argmax(
            vertical_reference_profile[right_start:width - frame_margin]
        )),
    )
    vertical_reference_run = (
        frame_margin + int(np.argmax(
            horizontal_reference_profile[frame_margin:side_height]
        )),
        bottom_start + int(np.argmax(
            horizontal_reference_profile[bottom_start:height - frame_margin]
        )),
    )

    # B-G 颜色差在某些图片中可能无法形成足够宽的左右区间。出现这种情况
    # 时，改用亮度剖面两侧的最大上升和下降梯度寻找左右边界。
    horizontal_run_invalid = (
        horizontal_run is None
        or horizontal_run[1] - horizontal_run[0] + 1 < width * MIN_WINDOW_RATIO
        or horizontal_run[0] < EDGE_SEARCH_RADIUS
        or horizontal_run[1] >= width - EDGE_SEARCH_RADIUS
        or (
            horizontal_reference_run[0] - horizontal_run[0]
            > EDGE_SEARCH_RADIUS
        )
        or (
            horizontal_run is not None
            and horizontal_run[0] > side_width
            and horizontal_reference_run[0] < horizontal_run[0]
            and width - horizontal_reference_run[0]
            > horizontal_run[1] - horizontal_run[0] + 1
        )
    )
    if horizontal_run_invalid:
        center_x = width // 2
        if (
            horizontal_run is not None
            and horizontal_reference_run[0] > horizontal_run[0]
            and horizontal_reference_run[1] < horizontal_run[1]
            and horizontal_reference_run[1] - horizontal_reference_run[0] + 1
            >= width * MIN_WINDOW_RATIO
        ):
            horizontal_run = horizontal_reference_run
            horizontal_run_invalid = False

    if horizontal_run_invalid:
        left_limit = EDGE_SEARCH_RADIUS
        right_limit = width - EDGE_SEARCH_RADIUS - 1
        left_reference = vertical_reference_profile[left_limit:center_x]
        right_reference = vertical_reference_profile[center_x:right_limit]
        left_threshold = float(left_reference.max()) * 0.5
        right_threshold = float(right_reference.max()) * 0.5
        left_peaks = np.flatnonzero(
            (left_reference[1:-1] >= left_reference[:-2])
            & (left_reference[1:-1] >= left_reference[2:])
            & (left_reference[1:-1] >= left_threshold)
        ) + 1
        right_peaks = np.flatnonzero(
            (right_reference[1:-1] >= right_reference[:-2])
            & (right_reference[1:-1] >= right_reference[2:])
            & (right_reference[1:-1] >= right_threshold)
        ) + 1
        left_candidates = left_peaks
        right_candidates = right_peaks
        left_candidate = (
            left_limit + int(left_candidates[-1])
            if left_candidates.size
            else left_limit
        )
        right_candidate = (
            center_x + int(right_candidates[0])
            if right_candidates.size
            else right_limit
        )
        if (
            left_candidate < center_x
            and right_candidate > center_x
            and right_candidate - left_candidate + 1 >= width * MIN_WINDOW_RATIO
        ):
            horizontal_run = (left_candidate, right_candidate)
        else:
            horizontal_run = horizontal_reference_run


    vertical_run_invalid = (
        vertical_run is None
        or vertical_run[1] - vertical_run[0] + 1 < height * MIN_WINDOW_RATIO
    )
    if vertical_run_invalid:
        vertical_run = vertical_reference_run

    # 找不到完整粗范围时，后续局部搜索没有可靠起点，直接报告失败。
    if horizontal_run is None or vertical_run is None:
        return None

    left, right = horizontal_run
    top, bottom = vertical_run
    window_width = right - left + 1
    window_height = bottom - top + 1

    if _debug_active():
        coarse = image.copy()
        cv2.rectangle(coarse, (left, top), (right, bottom), (0, 255, 0), 3)
        _debug_save("07_coarse_box", coarse)

    # 面积很小的中心区域更可能是反光或其他物体，不把它当作窗片。
    if window_width < width * MIN_WINDOW_RATIO or window_height < height * MIN_WINDOW_RATIO:
        return None

    # 6.4 计算边缘梯度，准备精定位。
    # Sobel 的 dy=1 检测横边，dx=1 检测竖边。取绝对值后，无论像素由暗
    # 变亮还是由亮变暗，都会得到正的边缘强度。
    color_horizontal_gradient = np.abs(
        cv2.Sobel(color_score, cv2.CV_32F, 0, 1, ksize=3)
    )
    horizontal_gradient = horizontal_edges
    # 上边同时使用颜色差和亮度梯度。两种证据相加后，微弱上边更突出。
    top_gradient = color_horizontal_gradient + horizontal_gradient
    vertical_gradient = vertical_edges

    _debug_visualize_gradient("08_top_gradient", top_gradient)
    _debug_visualize_gradient("09_horizontal_gradient", horizontal_gradient)
    _debug_visualize_gradient("10_vertical_gradient", vertical_gradient)

    # 6.5 拟合上边界。
    # 第一次只从粗上边界向下搜索。这是原有策略，对上边完整的数据稳定，
    # 但严重崩边会使纵向中值剖面的粗上边界落入缺口内部。
    top_line, top_support = _fit_horizontal_edge(
        top_gradient,
        top,
        left,
        right,
        upper_radius=0,
        debug_name="top",
    )
    # 第二次允许向上和向下各搜索 100 像素，用残留的完整边缘把上边界
    # 提回崩边之前的位置。
    upper_top_line, upper_top_support = _fit_horizontal_edge(
        top_gradient,
        top,
        left,
        right,
        debug_name="upper_top",
    )
    # 原策略失败时只能使用双向结果。两者都成功时，不直接选择更高的中心
    # 点，而是比较左右两个端点；只有新直线在两端都不低于原直线，才说明
    # 它把整条上边向外扩展。这样可以拒绝“一端上移、一端下移”的异常斜线。
    # 灯带也会在两端同时更靠上，因此还要求新直线的支持点更多，避免用
    # 更短的灯带边缘替换已经稳定的窗片上边。
    # 图像坐标的 y 向下增大，所以数值更小表示位置更靠上。
    if top_line is None:
        top_line = upper_top_line
    elif upper_top_line is not None:
        current_left_y = top_line[0] * left + top_line[1]
        current_right_y = top_line[0] * right + top_line[1]
        upper_left_y = upper_top_line[0] * left + upper_top_line[1]
        upper_right_y = upper_top_line[0] * right + upper_top_line[1]
        if (
            upper_left_y <= current_left_y
            and upper_right_y <= current_right_y
            and upper_top_support > top_support
        ):
            top_line = upper_top_line
    if top_line is None:
        top = vertical_reference_run[0]
        top_line, _ = _fit_horizontal_edge(
            top_gradient, top, left, right, debug_name="fallback_top",
        )

    bottom_line, _ = _fit_horizontal_edge(
        horizontal_gradient, bottom, left, right, debug_name="bottom",
    )
    if bottom_line is None:
        bottom = vertical_reference_run[1]
        bottom_line, _ = _fit_horizontal_edge(
            horizontal_gradient,
            bottom,
            left,
            right,
            debug_name="fallback_bottom",
        )

    # 6.6 拟合左右边界。
    # 左右外侧就是灯带（灯珠圆形边缘的梯度极强），双向搜索会被拉到灯带上。
    # 与上边界一致：第一次只向窗片内部搜索得到稳态直线，第二次允许双向
    # 搜索，若结果在上下两个端点都比第一次更靠内，才用它替换第一次的结果，
    # 拒绝“一端向内、一端向外”的漂移线。
    left_line = _fit_vertical_edge(
        vertical_gradient, left, top, bottom,
        left_radius=0,
        right_radius=3 * EDGE_SEARCH_RADIUS,
        debug_name="left",
    )
    inner_left_line = _fit_vertical_edge(
        vertical_gradient, left, top, bottom,
        debug_name="inner_left",
    )
    # 左边界：x 越大越靠内。仅当双向结果的上下端点都不比第一次更靠外（x 不减小）
    # 才替换。
    if left_line is None:
        left_line = inner_left_line
    elif inner_left_line is not None:
        current_top_x = left_line[0] * top + left_line[1]
        current_bottom_x = left_line[0] * bottom + left_line[1]
        inner_top_x = inner_left_line[0] * top + inner_left_line[1]
        inner_bottom_x = inner_left_line[0] * bottom + inner_left_line[1]
        if (
            inner_top_x >= current_top_x
            and inner_bottom_x >= current_bottom_x
        ):
            left_line = inner_left_line
    if left_line is None:
        left = horizontal_reference_run[0]
        left_line = _fit_vertical_edge(
            vertical_gradient,
            left,
            top,
            bottom,
            left_radius=0,
            right_radius=3 * EDGE_SEARCH_RADIUS,
            debug_name="fallback_left",
        )

    right_line = _fit_vertical_edge(
        vertical_gradient, right, top, bottom,
        left_radius=3 * EDGE_SEARCH_RADIUS,
        right_radius=0,
        debug_name="right",
    )
    inner_right_line = _fit_vertical_edge(
        vertical_gradient, right, top, bottom,
        debug_name="inner_right",
    )
    # 右边界：x 越小越靠内。仅当双向结果的上下端点都不比第一次更靠外（x 不增大）
    # 才替换。
    if right_line is None:
        right_line = inner_right_line
    elif inner_right_line is not None:
        current_top_x = right_line[0] * top + right_line[1]
        current_bottom_x = right_line[0] * bottom + right_line[1]
        inner_top_x = inner_right_line[0] * top + inner_right_line[1]
        inner_bottom_x = inner_right_line[0] * bottom + inner_right_line[1]
        if (
            inner_top_x <= current_top_x
            and inner_bottom_x <= current_bottom_x
        ):
            right_line = inner_right_line
    if right_line is None:
        right = horizontal_reference_run[1]
        right_line = _fit_vertical_edge(
            vertical_gradient,
            right,
            top,
            bottom,
            left_radius=3 * EDGE_SEARCH_RADIUS,
            right_radius=0,
            debug_name="fallback_right",
        )
    lines = (top_line, bottom_line, left_line, right_line)

    _debug_edges_overlay(image)

    # 任意一条边拟合失败，都无法构成可信四边形。
    if any(line is None for line in lines):
        return None

    # 6.6 按左上、右上、右下、左下的顺序计算四个交点。
    corners = [
        _intersect(top_line, left_line),
        _intersect(top_line, right_line),
        _intersect(bottom_line, right_line),
        _intersect(bottom_line, left_line),
    ]
    if any(point is None for point in corners):
        return None

    # 交点最初是浮点坐标。绘图和掩膜需要整数坐标，因此四舍五入后转 int32。
    # clip 保证交点不会越过图像边界。
    box = np.rint(corners).astype(np.int32)
    box[:, 0] = np.clip(box[:, 0], 0, width - 1)
    box[:, 1] = np.clip(box[:, 1], 0, height - 1)

    if _debug_active():
        final = image.copy()
        cv2.polylines(final, [box], True, (0, 0, 255), 5, cv2.LINE_AA)
        _debug_save("12_final", final)

    # 6.7 创建单通道掩膜：窗片四边形内部为 255，外部为 0。
    window_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(window_mask, box, 255)
    window = cv2.bitwise_and(image, image, mask=window_mask)

    # boundingRect 返回包住四边形的水平矩形，用于得到尺寸更小的裁剪图。
    x, y, crop_width, crop_height = cv2.boundingRect(box)
    crop = window[y:y + crop_height, x:x + crop_width]

    return {
        "box": box,
        "mask": window_mask,
        "window": window,
        "crop": crop,
    }

