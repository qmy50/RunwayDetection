import cv2
import numpy as np
import math
import config


def draw_full_image_line(img, k, b, color=(0, 0, 255), thickness=2):
    """
    在图像上绘制直线 x = k*y + b。
    
    参数：
        img (numpy.ndarray): 要绘制的图像（会直接修改）
        k (float): 斜率 (dx/dy)
        b (float): 截距
        color (tuple): BGR颜色，默认为绿色 (0,255,0)
        thickness (int): 线条粗细，默认为2
    """
    if abs(k) > 1e5 or np.isnan(k) or np.isinf(k):
        return  # 直接跳过，不画线
    if np.isnan(b) or np.isinf(b):
        return
    h, w = img.shape[:2]          # 获取图像高度和宽度

    # 选择图像顶部 (y=0) 和底部 (y=h-1) 两个点
    y1 = 0
    y2 = int(h - 1)
    x1 = int(k * y1 + b)          # 对应 x 坐标
    x2 = int(k * y2 + b)

    # 用 cv2.line 绘制直线，超出图像的部分会自动裁剪
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def calculate_midline(slope1, intercept1, slope2, intercept2, 
                     frame_height, y_min=0, y_max=None):
    """
    计算两条直线的中线（平分线）
    :param slope1: 左线斜率 (k1)
    :param intercept1: 左线截距 (b1) → y = k1*x + b1 → x = (y - b1)/k1
    :param slope2: 右线斜率 (k2)
    :param intercept2: 右线截距 (b2)
    :param frame_height: 图像高度（用于采样）
    :param y_min: 采样起始y坐标（默认0）
    :param y_max: 采样结束y坐标（默认图像高度）
    :return: 中线的斜率、截距，中线采样点列表
    """
    # 1. 处理默认参数
    if y_max is None:
        y_max = frame_height
    
    # 2. 计算灭点（两条线的交点）
    slope_diff = slope1 - slope2
    if abs(slope_diff) < 1e-6:
        # 两条线平行 → 中线是两条线的水平/垂直平分线
        print("两条线平行，计算平行中线")
        # 取多个y值计算中点
        mid_points = []
        for y in np.linspace(y_min, y_max, 20):
            # 计算两条线上的x坐标
            x1 = (y - intercept1) / slope1 if abs(slope1) > 1e-6 else 0
            x2 = (y - intercept2) / slope2 if abs(slope2) > 1e-6 else frame_height
            # 中点
            mid_x = (x1 + x2) / 2
            mid_y = y
            mid_points.append((mid_x, mid_y))
    else:
        # 计算灭点坐标
        vp_x = (intercept2 - intercept1) / slope_diff
        vp_y = slope1 * vp_x + intercept1
        # 3. 在y轴方向取采样点（从灭点下方到画面底部）
        mid_points = []
        # 采样范围：灭点y到画面底部，避免越界
        sample_y_start = max(vp_y, y_min)
        sample_y_end = min(y_max, frame_height)
        if sample_y_start >= sample_y_end:
            sample_y_start = y_min
            sample_y_end = y_max
        
        # 取20个采样点（越多拟合越准）
        sample_ys = np.linspace(sample_y_start, sample_y_end, 20)
        for y in sample_ys:
            # 计算两条线上当前y对应的x坐标
            x1 = (y - intercept1) / slope1 if abs(slope1) > 1e-6 else 0
            x2 = (y - intercept2) / slope2 if abs(slope2) > 1e-6 else frame_height
            # 计算中点
            mid_x = (x1 + x2) / 2
            mid_y = y
            mid_points.append((mid_x, mid_y))
    
    # 4. 拟合中线（用最小二乘法）
    mid_points = np.array(mid_points)
    if len(mid_points) < 2:
        return 0, 0, []  # 无足够点，返回默认值
    
    # 提取x/y坐标，过滤无效值
    xs = mid_points[:, 0]
    ys = mid_points[:, 1]
    valid_mask = ~(np.isnan(xs) | np.isinf(xs) | np.isnan(ys) | np.isinf(ys))
    xs = xs[valid_mask]
    ys = ys[valid_mask]
    
    if len(xs) < 2:
        return 0, 0, mid_points.tolist()
    
    # 拟合y = k*x + b → 转换为x = k*y + b（车道线更适合y→x拟合）
    coeffs = np.polyfit(ys, xs, 1)  # 1次多项式（直线）
    mid_slope = 1 / coeffs[0] if abs(coeffs[0]) > 1e-6 else 0  # 转换回y=k*x+b的斜率
    mid_intercept = -coeffs[1] / coeffs[0] if abs(coeffs[0]) > 1e-6 else coeffs[1]
    
    return mid_slope, mid_intercept, mid_points.tolist()

def ransac_polyfit(x_data, y_data, degree=1, max_trials=100, residual_threshold=30):
        best_coeffs = None  # 保存最优的多项式系数（最终车道曲线）
        best_inlier_count = 0  # 最优模型的内点数量

        # 数据量不足时，退化为普通拟合
        if len(x_data) < degree + 2:
            return np.polyfit(x_data, y_data, degree) if len(x_data) > degree else None

        n_samples = len(x_data)
        sample_size = degree + 1  # 拟合2次多项式需要至少3个点（degree=2 → 3个点）

        # 核心：多次随机抽样+拟合+验证，找到最优模型
        for _ in range(max_trials):  # 最多试100次
            # 步骤1：随机选少量样本（3个点）作为“候选内点”
            sample_indices = np.random.choice(n_samples, sample_size, replace=False)
            sample_x = x_data[sample_indices]
            sample_y = y_data[sample_indices]

            # 步骤2：用候选内点拟合一个临时多项式模型
            try:
                coeffs = np.polyfit(sample_y, sample_x, degree)  # 拟合y→x的2次多项式（车道曲线）

                # 步骤3：用临时模型验证所有点，区分“内点”和“外点”
                y_pred = np.polyval(coeffs, y_data)  # 用模型预测所有点的x值
                residuals = np.abs(x_data - y_pred)  # 计算真实x与预测x的误差（残差）
                inlier_mask = residuals < residual_threshold  # 误差<30像素 → 内点（真实车道点）
                inlier_count = np.sum(inlier_mask)  # 统计内点数量

                # 步骤4：保留内点最多的模型（最优模型）
                if inlier_count > best_inlier_count:
                    best_coeffs = coeffs
                    best_inlier_count = inlier_count
                    # 内点足够多时提前终止（加速）
                    if inlier_count > 0.7 * n_samples:
                        break
            except:
                continue
        # 最终：用所有内点重新拟合，得到更精准的模型
        if best_coeffs is not None and best_inlier_count > sample_size:
            y_pred = np.polyval(best_coeffs, y_data)
            residuals = np.abs(x_data - y_pred)
            inlier_mask = residuals < residual_threshold
            if np.sum(inlier_mask) > sample_size:
                try:
                    return np.polyfit(y_data[inlier_mask], x_data[inlier_mask], degree)
                except:
                    return best_coeffs 
                
        return best_coeffs

# ===================== 【新增】以指定点为对称中心的工具函数 =====================
def get_point_symmetry_about_vp(p, vp):
    """
    以灭点vp为对称中心，计算点p的对称点
    :param p: 原始点 (x, y)
    :param vp: 对称中心（灭点）(vp_x, vp_y)
    :return: 对称点 (x', y')
    """
    x, y = p
    vp_x, vp_y = vp
    # 对称中心变换公式：x' = 2*vp_x - x  ;  y' = 2*vp_y - y
    sym_x = int(2 * vp_x - x)
    sym_y = int(2 * vp_y - y)
    return (sym_x, sym_y)

def mirror_ipm_points_about_vp(source_points, vp, w, h):
    """
    以灭点vp为对称中心，对称生成另一侧IPM点
    :param source_points: 有效侧IPM点 [down, up]
    :param vp: 灭点 (vp_x, vp_y)
    :param w/h: 画面宽高（边界裁剪）
    :return: 对称后的IPM点 [down, up]
    """
    d_sym = get_point_symmetry_about_vp(source_points[0], vp)
    u_sym = get_point_symmetry_about_vp(source_points[1], vp)
    # 边界裁剪，避免超出画面
    d_sym = (np.clip(d_sym[0], 0, w), np.clip(d_sym[1], 0, h))
    u_sym = (np.clip(u_sym[0], 0, w), np.clip(u_sym[1], 0, h))
    return [d_sym, u_sym]

def calculate_line_intersection_with_horizon(line_slope, line_intercept, horizon_k, horizon_b):
    """
    计算检测线与地平线的交点（新灭点）
    :param line_slope/line_intercept: 有效侧检测线参数 y = k1*x + b1
    :param horizon_k/horizon_b: 地平线参数 y = k2*x + b2
    :return: 交点 (x, y)，None表示平行线无交点
    """
    # 联立方程：k1*x + b1 = k2*x + b2 → x = (b2 - b1)/(k1 - k2)
    if abs(line_slope - horizon_k) < 1e-4:  # 平行线
        return None
    intersect_x = (horizon_b - line_intercept) / (line_slope - horizon_k)
    intersect_y = line_slope * intersect_x + line_intercept
    return (int(intersect_x), int(intersect_y))

# 重构对称函数：满足「同一y值下x关于vp_x对称」
def get_mirror_line_by_vp(slope, intercept, vp_x, img_w,y_list=[100,300]):
    """
    对称规则：同一y值下，x' = 2*vp_x - x（y值不变，x关于vp_x对称）
    步骤：
    1. 取原直线上两个不同y值的点，计算其对称点（y不变，x对称）
    2. 用两个对称点拟合新直线的斜率和截距
    """
    mirror_pts = []
    
    for y in y_list:
        # 计算原直线在该y值下的x
        if abs(slope) < 1e-6:  # 水平线
            x = vp_x  # 特殊情况
        else:
            x = (y - intercept) / slope
        x = np.clip(x, 0, img_w-1)
        
        # 计算对称点：y不变，x = 2*vp_x - x
        x_mirror = 2 * vp_x - x
        x_mirror = np.clip(x_mirror, 0, img_w-1)
        mirror_pts.append((int(x_mirror), int(y)))
    
    # 用两个对称点计算新直线的斜率和截距
    (x1, y1), (x2, y2) = mirror_pts
    dx = x2 - x1
    dy = y2 - y1
    
    if abs(dx) < 1e-6:  # 垂直线
        new_slope = 999999
        new_intercept = x1
    else:
        new_slope = dy / dx
        new_intercept = y1 - new_slope * x1
    
    return new_slope, new_intercept

def mirror_line_by_vp_x(k, b, vp_x):
    """
    以 x = vp_x 为对称轴，生成直线 x = k*y + b 的镜像直线
    镜像公式：新直线 x' = -k * y + (2*vp_x - b)
    """
    k_mirror = -k
    b_mirror = 2 * vp_x - b
    return k_mirror, b_mirror

# ========== PNP检测用辅助函数 ==========
def edge_detection_in_mask(img_cv, mask_cv, low_threshold, high_threshold):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges_in_mask = cv2.bitwise_and(edges, edges, mask=mask_cv)
    return edges_in_mask

def expand_mask_dilation(mask, kernel_size, iterations):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    expanded_mask = cv2.dilate(mask, kernel, iterations=iterations)
    return expanded_mask

def approx(con, side_size=4):
    num = 0.0005
    ep = num * cv2.arcLength(con, True)
    con = cv2.approxPolyDP(con, ep, True)
    while True:
        if len(con) <= side_size:
            break
        else:
            num *= 1.5
            ep = num * cv2.arcLength(con, True)
            con = cv2.approxPolyDP(con, ep, True)
    return con

def detect_points_from_edges(edges_in_mask, min_contour_length=30):
    """
    从边缘图中检测四边形顶点，返回四个顶点坐标（按凸包顺序）
    如果检测失败，返回None
    """
    contours, _ = cv2.findContours(edges_in_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_points = []
    for cnt in contours:
        length = cv2.arcLength(cnt, closed=True)
        if length >= min_contour_length:
            all_points.extend(cnt.reshape(-1, 2))
    if len(all_points) < 4:
        return None
    all_points = np.array(all_points)
    hull = cv2.convexHull(all_points)
    con = approx(hull, side_size=4)
    if len(con) != 4:
        return None
    # 提取四个顶点
    points = np.zeros((4, 2), dtype=int)
    print("最外侧四个顶点坐标：")
    for i, point in enumerate(con):
        print(f"顶点 {i+1}: ({point[0,0]}, {point[0,1]})")
        points[i] = point[0]
    if np.max(points,axis=0)[0] - np.min(points,axis=0)[0] > 0.5 * edges_in_mask.shape[0]:
        config.DRAW_PNP = True
    return points

def distance(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def validate_coordinate(coord, frame_shape):
    x, y = coord
    h, w = frame_shape[:2]
    x = max(0, min(int(round(x)), w-1))
    y = max(0, min(int(round(y)), h-1))
    return (x, y)

def get_rotation_matrix(rvec):
    """将旋转向量转换为旋转矩阵（替代手动欧拉角计算，更准确）"""
    R, _ = cv2.Rodrigues(rvec)
    return R

def get_camera_position(rvec, tvec):
    """
    正确计算相机在世界坐标系中的位置
    核心公式：cam_pos = -R.T @ tvec（正交矩阵逆=转置）
    """
    R = get_rotation_matrix(rvec)
    R_inv = R.T  # 旋转矩阵的逆 = 转置
    tvec_reshaped = tvec.reshape(3, 1)
    cam_pos = -np.dot(R_inv, tvec_reshaped)
    cam_x = cam_pos[0][0]
    cam_y = cam_pos[1][0]
    cam_z = cam_pos[2][0]
    return cam_x, cam_y, cam_z

def get_euler_angle(rotation_vector):
    """计算欧拉角（roll/pitch/yaw），并返回角度值（方便调试）"""
    theta = cv2.norm(rotation_vector, cv2.NORM_L2)
    if theta < 1e-8:
        return 0, 0, 0, 0, 0, 0
    w = math.cos(theta / 2)
    x = math.sin(theta / 2) * rotation_vector[0][0] / theta
    y = math.sin(theta / 2) * rotation_vector[1][0] / theta
    z = math.sin(theta / 2) * rotation_vector[2][0] / theta
    ysqr = y * y
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + ysqr)
    pitch = math.atan2(t0, t1)
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    yaw = math.asin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    roll = math.atan2(t3, t4)
    # 转换为角度（度）
    roll_deg = math.degrees(roll)
    pitch_deg = math.degrees(pitch)
    yaw_deg = math.degrees(yaw)
    return roll, pitch, yaw, roll_deg, pitch_deg, yaw_deg

def canny_vertical_edges(img, pt1, pt2, blur_ksize=5, canny_low=50, canny_high=150):
    x1, y1 = pt1
    x2, y2 = pt2
    x_min = max(0, min(x1, x2))
    x_max = min(img.shape[1], max(x1, x2))
    y_min = max(0, min(y1, y2))
    y_max = min(img.shape[0], max(y1, y2))
    h, w = img.shape[:2]
    
    if x_min >= x_max or y_min >= y_max:
        return np.zeros((h,w), dtype=np.uint8), np.zeros((h,w), dtype=np.uint8)

    roi_img = img[y_min:y_max, x_min :x_max ]
    gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_roi, (blur_ksize, blur_ksize), 1.2)
    sobel_x = cv2.Sobel(blur, cv2.CV_64F, dx=1, dy=0, ksize=3)
    sobel_y = cv2.Sobel(blur, cv2.CV_64F, dx=0, dy=1, ksize=3)
    grad_dir = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
    roi_edges = cv2.Canny(blur, canny_low, canny_high)
    
    roi_vertical_mask = np.zeros_like(roi_edges)
    mask1 = (np.abs(grad_dir) <= 80)
    mask2 = (np.abs(grad_dir) >= 100)
    mask3 = (roi_edges == 255)
    roi_vertical_mask[(mask1 | mask2) & mask3] = 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 3))
    roi_vertical_edges = cv2.morphologyEx(roi_vertical_mask, cv2.MORPH_CLOSE, kernel)
    
    full_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    full_edges = np.zeros_like(full_gray)
    full_vertical_edges = np.zeros_like(full_gray)
    full_edges[y_min:y_max, x_min :x_max ] = roi_edges
    full_vertical_edges[y_min:y_max, x_min :x_max ] = roi_vertical_edges
    
    return full_edges, full_vertical_edges

def Hough_detection(vertical_edges,h):
    lines = cv2.HoughLinesP(
        vertical_edges,
        rho=1,
        theta=np.pi/180,
        threshold = 40,
        minLineLength=int(0.6*h),
        maxLineGap=100
    )
    if lines is None:
        return None
    lines = lines.astype(np.float32)
    return lines

def average(lines, cx, left_slope_mean=None, right_slope_mean=None, slope_threshold=0.5):
    left = []
    right = []
    if lines is None:
        return left, right
    
    for line in lines:
        x1, y1, x2, y2 = line
        if x1 - x2 == 0 or abs(y1 - y2) < 10:
            continue
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length < 5:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.01:
            continue
        
        mid_x = (x1 + x2) / 2
        y_int = y1 - slope * x1
        
        if mid_x < cx:
            if left_slope_mean is None or abs(slope - left_slope_mean) < slope_threshold:
                left.append((slope, y_int))
        else:
            if right_slope_mean is None or abs(slope - right_slope_mean) < slope_threshold:
                right.append((slope, y_int))
    
    return left, right

def get_line_intersection(k1, b1, k2, b2):
    """
    计算两条直线 (x = k1*y + b1) 与 (x = k2*y + b2) 的交点
    返回 (x, y)，若平行则返回 None
    """
    if abs(k1 - k2) < 1e-6:
        return None
    y = (b2 - b1) / (k1 - k2)
    x = k1 * y + b1
    return (int(x), int(y))

def get_point_on_line(k, b, ref_y):
    """给定直线 x = k*y + b 和参考 y，返回线上点 (x, y)"""
    x = k * ref_y + b
    return (int(x), int(ref_y))

def get_shortest_line_end_y(k_l, b_l, k_r, b_r, vp_x, vp_y, frame_width, frame_height):
    """
    计算左右线从灭点延伸至图像边界时，较短那条线的终点 y 坐标
    直线参数均为 x = k*y + b
    """
    # 左线：从灭点出发，沿直线向下，直到触碰到图像边界
    def line_end_y(k, b, vp_x, vp_y, is_left):
        # 尝试 y = frame_height
        x_bottom = k * frame_height + b
        if 0 <= x_bottom <= frame_width:
            return frame_height  # 终点 y 即为底边
        # 否则用 x 边界计算 y
        if is_left:
            target_x = 0
        else:
            target_x = frame_width
        # 由 x = k*y + b  => y = (x - b)/k
        y_bound = (target_x - b) / k if abs(k) > 1e-6 else vp_y
        return np.clip(y_bound, 0, frame_height)
    
    left_end_y = line_end_y(k_l, b_l, vp_x, vp_y, is_left=True)
    right_end_y = line_end_y(k_r, b_r, vp_x, vp_y, is_left=False)
    
    # 比较线段长度（垂直方向距离灭点的长度）
    left_len = abs(left_end_y - vp_y)
    right_len = abs(right_end_y - vp_y)
    
    return int(left_end_y if left_len < right_len else right_end_y), int(vp_y)

def get_division_points_by_vp_and_short_line(k, b, vp_y, line_end_y_min, line_end_y_max,
                                             x_min=0, x_max=None, division_num=10):
    """
    在直线 x = k*y + b 上，从灭点 y 到 line_end_y_max 之间均匀取 division_num 个点
    """
    if x_max is None:
        x_max = 1e9
    y_start = min(vp_y, line_end_y_max)
    y_end   = max(vp_y, line_end_y_max)
    if abs(y_end - y_start) < 10:
        y_start = line_end_y_min
        y_end   = line_end_y_max
    
    points = []
    for i in range(division_num):
        ratio = i / (division_num - 1)
        y = int(y_start + ratio * (y_end - y_start))
        x = int(k * y + b)
        x = np.clip(x, x_min, x_max)
        points.append((x, y))
    return points

# ===================== 【新增】对称点工具函数 =====================
def get_mirror_point(p, mid_x, w, h):
    x, y = p
    mirror_x = int(2 * mid_x - x)  # 中线对称公式
    mirror_x = np.clip(mirror_x, 0, w)
    mirror_y = np.clip(y, 0, h)
    return (mirror_x, mirror_y)

# 使用新对称函数 mirror_line_by_vp_x
def mirror_ipm_points(source_points, vp_x, w, h):
    """
    source_points: [down, up] 每个点为 (x, y)
    返回镜像侧 [down, up]
    """
    # 先将点转换为直线？不，我们只需对称点本身
    d_sym = (int(2*vp_x - source_points[0][0]), source_points[0][1])
    u_sym = (int(2*vp_x - source_points[1][0]), source_points[1][1])
    d_sym = (np.clip(d_sym[0],0,w), np.clip(d_sym[1],0,h))
    u_sym = (np.clip(u_sym[0],0,w), np.clip(u_sym[1],0,h))
    return [d_sym, u_sym]
# =================================================================

def smooth_line(hist, max_hist=5, use_weight=True):
    if not hist:
        return None
    if len(hist) == 1:
        return hist[0]
    if use_weight:
        weights = np.exp(np.linspace(0, 1, len(hist)))
        weights /= weights.sum()
        return np.average(hist, axis=0, weights=weights)
    else:
        return np.mean(hist, axis=0)


# ------------------- 测试示例 -------------------
if __name__ == "__main__":
    # 示例1：适配你之前的车道线场景（x = k*y + b）
    img_size = (1920, 1080)  # 图像尺寸 (宽, 高)
    slope = 0.05  # 斜率k（x = 0.05*y + 500）
    intercept = 500  # 截距b
    
    # 绘制直线
    result_img, pt1, pt2 = draw_full_image_line(
        img_size=img_size,
        slope=slope,
        intercept=intercept,
        line_color=(0, 255, 0),  # 绿色
        line_thickness=3
    )

    # 打印端点信息
    print(f"直线端点1：{pt1}，端点2：{pt2}")
    
    # 显示结果
    cv2.imshow("Full Image Line", result_img)
    # 保存结果（可选）
    # cv2.imwrite("full_line.png", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()