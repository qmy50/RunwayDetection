import cv2
import numpy as np
from utils import draw_full_image_line

# 世界坐标点（矩形，Z=0平面），顺序：左下、左上、右上、右下
points_3D = np.array([
    [401.25, 16.025, 0],   # 左上
    [401.25,  -16.025, 0],   # 右上
    [ 0,  -16.025, 0],   # 右下
    [ 0, 16.025, 0]    # 左下
], dtype=np.float64)

camera_matrix = np.array([
    [554.3827,   0.0,    320.5],
    [0.0,    554.3827,   240.5],
    [0.0,      0.0,      1.0]
], dtype=np.float64)

class PNPExtractor:
    """
    从图像中检测四边形（基于HSV-S通道边缘检测），提取对边直线参数 x = k*y + b，
    并利用PnP算法计算相机相对于原点的高度和Y方向偏移。
    """
    def __init__(self, canny_low=80, canny_high=180, min_area=1000,
                 show_debug=False, debug_wait=1):
        """
        参数:
            canny_low, canny_high: Canny边缘检测阈值
            min_area: 轮廓最小面积过滤
            show_debug: 是否显示中间步骤图像（S通道、边缘、结果）
            debug_wait: 调试窗口显示等待时间（毫秒），0表示按键等待
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_area = min_area
        self.show_debug = show_debug
        self.debug_wait = debug_wait
        self.end_flag = False

    def _detect_quadrilateral(self, image):
        """从HSV-S通道检测四边形顶点，返回 (4,2) 或 None"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]

        if self.show_debug:
            cv2.imshow("S Channel", s_channel)
            cv2.waitKey(self.debug_wait)

        blurred = cv2.GaussianBlur(s_channel, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        if self.show_debug:
            cv2.imshow("Edges", edges)
            cv2.waitKey(self.debug_wait)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_quad = None
        max_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                if area > max_area:
                    max_area = area
                    best_quad = approx.reshape(4, 2)
        return best_quad

    def _order_points_by_angle(self, pts):
        """极角逆时针排序，起点为角度最小的点"""
        if pts is None or len(pts) != 4:
            return np.zeros((4,2), dtype=np.float64) \
        
        pts = np.array(pts).reshape(4, 2)
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        sorted_pts = pts[sorted_indices].astype(np.float64)
        return sorted_pts 

    def _fit_line_x_ky_b(self, p1, p2):
        """拟合直线 x = k*y + b，返回 (k, b)"""
        x1, y1 = p1
        x2, y2 = p2
        if abs(y2 - y1) < 1e-6:
            return float('inf'), float(x1)
        k = (x2 - x1) / (y2 - y1)
        b = x1 - k * y1
        return k, b

    def _compute_camera_pose(self, image_points):
        """
        给定图像点（已排序为 [左下,左上,右上,右下]），使用PnP计算相机在世界坐标系中的位置。
        返回 (camera_position, rvec, tvec) 或 (None, None, None) 若失败。
        camera_position: (x, y, z) 世界坐标系下的相机光心位置。
        """
        # 世界点已经按相同顺序定义
        world_points = points_3D

        # 调用 solvePnP（假设无畸变）
        ret, rvec, tvec = cv2.solvePnP(world_points, image_points,
                                       camera_matrix, None,
                                       flags=cv2.SOLVEPNP_ITERATIVE)
        if not ret:
            return None, None, None

        # 将旋转向量转换为旋转矩阵
        R, _ = cv2.Rodrigues(rvec)
        # 相机位置 = -R^T * tvec
        camera_position = -R.T @ tvec.flatten()
        return camera_position.flatten(), rvec, tvec

    def process_frame(self, image, draw_result=False):
        img_copy = image.copy() if draw_result else image
        quad = self._detect_quadrilateral(image)

        # 统一返回格式
        result = {
            "height": None,
            "y_offset": None,
            "image": img_copy,
            "end_flag":self.end_flag
        }

        if quad is None or len(quad) != 4:
            return result

        # 将图像点排序为 [左上, 右上, 右下, 左下]
        ordered_pts = self._order_points_by_angle(quad)
        # print(ordered_pts.shape)
        if max(ordered_pts[2,0],ordered_pts[3,0]) >= 478:
            self.end_flag = True

        # 绘制对边直线（仍使用原有直线拟合，仅用于可视化）
        k1, b1 = self._fit_line_x_ky_b(ordered_pts[0], ordered_pts[3])  # 左上->左下（左边）
        k2, b2 = self._fit_line_x_ky_b(ordered_pts[1], ordered_pts[2])  # 右上->右下（右边）
        if draw_result:
            draw_full_image_line(img_copy, k1, b1)
            draw_full_image_line(img_copy, k2, b2)

        # PnP 解算相机位姿
        camera_pos, rvec, tvec = self._compute_camera_pose(ordered_pts)
        if camera_pos is not None:
            result["height"] = camera_pos[2]      # Z 轴高度（假设 Z 向上）
            result["y_offset"] = camera_pos[1]    # Y 方向偏移（右手系）

        if draw_result:
            for i, pt in enumerate(ordered_pts):
                cv2.circle(img_copy, tuple(pt.astype(int)), 6, (0, 255, 255), -1)
                cv2.putText(img_copy, str(i), tuple(pt.astype(int) + [10, 10]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        result["image"] = img_copy
        result["end_flag"] = self.end_flag
        return result


# ========== 测试示例 ==========
if __name__ == "__main__":
    img_path = r"/home/tianbot/XTDrone/control/keyboard/frame_1200.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print("无法读取图片")
        exit()

    extractor = PNPExtractor(canny_low=80, canny_high=180,
                             min_area=1000, show_debug=False, debug_wait=0)

    result = extractor.process_frame(img, draw_result=True)

    if result["height"] is not None:
        print(f"相机高度 (Z): {result['height']:.3f}")
        print(f"相机 Y 方向偏移: {result['y_offset']:.3f}")
    else:
        print("未检测到四边形或 PnP 解算失败")

    cv2.imshow("Result", result['image'])
    cv2.waitKey(0)
    cv2.destroyAllWindows()