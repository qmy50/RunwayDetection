import cv2
import numpy as np
import math
import config

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

def Hough_detection(vertical_edges, h):
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

def average(lines, cx, left_k_mean=None, right_k_mean=None, k_threshold=0.5):
    """
    将霍夫线段转换为直线参数 x = k*y + b，并按中线 cx 分为左右两组。
    参数:
        lines: 线段数组 [x1,y1,x2,y2]
        cx: 图像中心x坐标，用于区分左右
        left_k_mean, right_k_mean: 上一帧左右直线的k值，用于阈值过滤
        k_threshold: k值差异阈值（注意：新k值通常较小，原阈值0.5可能过大，建议根据实际情况调整）
    返回:
        left, right: 列表元素为 (k, b)
    """
    left = []
    right = []
    if lines is None:
        return left, right
    
    for line in lines:
        x1, y1, x2, y2 = line
        if y1 == y2 or abs(x1 - x2) < 10:  # 避免水平线和短线段
            continue
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length < 5:
            continue
        # 计算新参数 k = Δx/Δy, b = x1 - k*y1
        k = (x2 - x1) / (y2 - y1)
        b = x1 - k * y1
        
        mid_x = (x1 + x2) / 2
        
        if mid_x < cx:
            if left_k_mean is None or abs(k - left_k_mean) < k_threshold:
                left.append((k, b))
        else:
            if right_k_mean is None or abs(k - right_k_mean) < k_threshold:
                right.append((k, b))
    
    return left, right

def get_line_intersection(k1, b1, k2, b2):
    """
    计算两条直线 x = k1*y + b1 和 x = k2*y + b2 的交点。
    返回 (x, y) 或 None (平行)
    """
    if abs(k1 - k2) < 1e-4:
        return None
    y = (b2 - b1) / (k1 - k2)
    x = k1 * y + b1
    return (int(x), int(y))

def get_point_on_line(k, b, ref_y, offset=0):
    """
    在直线 x = k*y + b 上，给定参考y坐标 + 偏移量，计算点坐标。
    """
    target_y = ref_y + offset
    x = k * target_y + b
    return (int(x), int(target_y))

def get_five_division_points(k, b, y_min, y_max):
    """
    在直线 x = k*y + b 上，从 y_min 到 y_max 均匀取5个点（包括两端）。
    """
    points = []
    for i in range(5):
        ratio = i / 4.0
        y = int(y_min + ratio * (y_max - y_min))
        x = int(k * y + b)
        points.append((x, y))
    return points