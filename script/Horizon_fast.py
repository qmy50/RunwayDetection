import cv2
import numpy as np

class HorizonDetector:
    """
    纯传统视觉地平线检测（无SAM、无深度学习、无PyTorch）
    核心：HSV天空分割 + Canny边缘检测 + 霍夫直线 + 直线拟合
    """
    def __init__(self):
        """
        初始化：无模型、无加载、纯CPU传统算法
        """
        pass  # 空初始化，超快

    def _crop_roi_safely(self, image, roi):
        """安全裁剪ROI（内部方法）"""
        h_img, w_img = image.shape[:2]
        x_roi, y_roi, w_roi, h_roi = roi
        
        x1 = max(0, x_roi)
        y1 = max(0, y_roi)
        x2 = min(w_img, x_roi + w_roi)
        y2 = min(h_img, y_roi + h_roi)
        
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"无效的ROI：{roi}")
        
        roi_image = image[y1:y2, x1:x2].copy()
        return roi_image, (x1, y1, x2, y2)

    @staticmethod
    def _detect_horizon_from_mask( mask, roi_y1, roi_y2, roi_x1, roi_x2,
                                  min_line_length=50, angle_tolerance=10):
        """从掩码检测地平线（纯传统算法）"""
        # 1. 边缘检测
        edges = cv2.Canny(mask, 50, 150)
        
        # 2. 过滤ROI边界线
        edge_mask = np.ones_like(edges)
        edge_mask[max(0, roi_y1-5):min(edges.shape[0], roi_y1+5), :] = 0
        edge_mask[max(0, roi_y2-5):min(edges.shape[0], roi_y2+5), :] = 0
        edge_mask[:, max(0, roi_x1-5):min(edges.shape[1], roi_x1+5)] = 0
        edge_mask[:, max(0, roi_x2-5):min(edges.shape[1], roi_x2+5)] = 0
        edges = edges * edge_mask

        # 3. 形态学膨胀
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel)
        
        # 4. 霍夫直线检测
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=20,
            minLineLength=min_line_length,
            maxLineGap=10
        )
        
        if lines is None:
            return None
        
        # 5. 筛选水平直线
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if abs(angle) <= angle_tolerance or abs(angle - 180) <= angle_tolerance:
                horizontal_lines.append([x1, y1, x2, y2])
        
        if len(horizontal_lines) == 0:
            return None
        
        # 6. 拟合直线
        points = np.array(horizontal_lines).reshape(-1, 2)
        x = points[:, 0]
        y = points[:, 1]
        
        try:
            coeff = np.polyfit(x, y, 1)
            return tuple(coeff)  # (k, b)
        except:
            return None

    @staticmethod
    def detect_horizon(image, roi, roi_sky_point=[[150, 10]],hsv_margin=[30, 50, 30]):
        """
        【主函数】纯传统视觉地平线检测
        :param image: OpenCV BGR图像
        :param roi: (x,y,w,h)
        :param roi_sky_point: 天空采样点
        :return: k, b → y=kx+b
        """
        x_roi, y_roi, w_roi, h_roi = roi
        roi_image = image[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi].copy()
        
        # 转HSV
        hsv_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)

        # 取天空采样点邻域
        sky_x, sky_y = roi_sky_point[0]
        sky_x = np.clip(sky_x, 0, w_roi-1)
        sky_y = np.clip(sky_y, 0, h_roi-1)

        patch_size = 10
        x1 = max(0, sky_x - patch_size//2)
        x2 = min(w_roi, sky_x + patch_size//2)
        y1 = max(0, sky_y - patch_size//2)
        y2 = min(h_roi, sky_y + patch_size//2)
        patch = hsv_roi[int(y1):int(y2), int(x1):int(x2)]
        
        if patch.size == 0:
            return None
        
        # 计算均值
        h_mean = np.mean(patch[:, :, 0])
        s_mean = np.mean(patch[:, :, 1])
        v_mean = np.mean(patch[:, :, 2])
        
        # 阈值
        lower = np.array([
            max(0, h_mean - hsv_margin[0]),
            max(0, s_mean - hsv_margin[1]),
            max(0, v_mean - hsv_margin[2])
        ], dtype=np.uint8)
        upper = np.array([
            min(180, h_mean + hsv_margin[0]),
            min(255, s_mean + hsv_margin[1]),
            min(255, v_mean + hsv_margin[2])
        ], dtype=np.uint8)
        
        # 天空掩膜
        roi_mask = cv2.inRange(hsv_roi, lower, upper)
        
        # 形态学去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel)
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)
        
        # 恢复全图mask
        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        full_mask[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi] = roi_mask

        # 拟合地平线
        horizon_params = HorizonDetector._detect_horizon_from_mask(
            full_mask,
            roi_y1=y_roi, roi_y2=y_roi+h_roi,
            roi_x1=x_roi, roi_x2=x_roi+w_roi
        )
        return horizon_params

    @staticmethod
    def draw_roi(image,roi):
        img = image.copy()
        x_roi, y_roi, w_roi, h_roi = roi
        x1 = x_roi
        y1 = y_roi
        x2 = x_roi + w_roi
        y2 = y_roi + h_roi
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.imshow("ROI",img)

# ====================== 测试示例 ======================
if __name__ == "__main__":
    from utils import draw_full_image_line  # 保留你原来的绘图函数
    
    IMAGE_PATH = r"D:\yolo\segmentation\runway_dataset\images\train\frame_000450.jpg"
    ROI = (400, 320, 300, 60)
    ROI_SKY_POINT = [[150, 10]]
    
    # 初始化（超快！）
    detector = HorizonDetector()
    
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError("图片不存在")
    
    # 检测地平线（纯传统算法）
    horizon_params = detector.detect_horizon(image, ROI, ROI_SKY_POINT)
    
    if horizon_params is not None:
        k, b = horizon_params
        draw_full_image_line(image, 1/k, -(b + 5)/k)
        print(f"地平线：y = {k:.6f}x + {b:.2f}")
    else:
        print("检测失败")
    
    detector.draw_roi(image, ROI)
    cv2.waitKey(0)
    cv2.destroyAllWindows()