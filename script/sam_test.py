import os
import sys
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from utils import draw_full_image_line
import torch

class HorizonDetector:
    """
    基于SAM的地平线检测类（支持GPU加速）
    核心功能：输入图像+ROI，输出地平线参数 (k, b) → y = k*x + b
    """
    def __init__(self, sam_checkpoint_path = None, model_type="vit_b", use_gpu=True):
        """
        初始化检测器
        :param sam_checkpoint_path: SAM权重文件路径
        :param model_type: SAM模型类型（vit_b/vit_l/vit_h）
        :param use_gpu: 是否优先使用GPU（有CUDA时自动启用）
        """
        self.sam_checkpoint = sam_checkpoint_path
        self.model_type = model_type
        self.use_gpu = use_gpu
        self.device = self._get_device()
        self.predictor = self._init_sam()

    def _get_device(self):
        """自动选择设备（优先GPU）"""
        if self.use_gpu and torch.cuda.is_available():
            print("✅ 使用 GPU 加速")
            return torch.device("cuda")
        else:
            print("⚠️ 未检测到CUDA，使用CPU运行")
            return torch.device("cpu")

    def _init_sam(self):
        """初始化SAM预测器（内部方法）"""
        if not os.path.exists(self.sam_checkpoint):
            raise FileNotFoundError(f"SAM权重文件不存在：{self.sam_checkpoint}")
        
        # 加载模型并放到指定设备
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam = sam.to(self.device)
        return SamPredictor(sam)

    def _crop_roi_safely(self, image, roi):
        """安全裁剪ROI（内部方法）"""
        h_img, w_img = image.shape[:2]
        x_roi, y_roi, w_roi, h_roi = roi
        
        # 边界检查
        x1 = max(0, x_roi)
        y1 = max(0, y_roi)
        x2 = min(w_img, x_roi + w_roi)
        y2 = min(h_img, y_roi + h_roi)
        
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"无效的ROI：{roi}，超出图像范围或尺寸为0")
        
        roi_image = image[y1:y2, x1:x2].copy()
        return roi_image, (x1, y1, x2, y2)

    @staticmethod
    def _detect_horizon_from_mask( mask, roi_y1, roi_y2, roi_x1, roi_x2,
                                  min_line_length=50, angle_tolerance=2):
        """从掩码检测地平线（内部方法）"""
        # 1. 边缘检测
        edges = cv2.Canny(mask, 50, 150)
        
        # 2. 过滤ROI边界线
        edge_mask = np.ones_like(edges)
        # 上下边界过滤
        edge_mask[max(0, roi_y1-5):min(edges.shape[0], roi_y1+5), :] = 0
        edge_mask[max(0, roi_y2-5):min(edges.shape[0], roi_y2+5), :] = 0
        # 左右边界过滤
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
            coefficients = np.polyfit(x, y, 1)
            return tuple(coefficients)  # (k, b)
        except:
            return None
        
    @staticmethod
    def detect_horizon(image, roi, roi_sky_point=[[150, 10]],hsv_margin=[30, 50, 50]):
        x_roi, y_roi, w_roi, h_roi = roi
        roi_image = image[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi].copy()
        
        # 2. 转换ROI到HSV
        hsv_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        # cv2.imshow("HSV ROI", hsv_roi)  # 调试显示HSV图像
        # 3. 提取天空点的HSV值
        sky_x, sky_y = roi_sky_point[0]  # 假设 roi_sky_point 是 [[x, y]]
        # 确保点位于ROI内
        sky_x = np.clip(sky_x, 0, w_roi-1)
        sky_y = np.clip(sky_y, 0, h_roi-1)
        
        # 取小邻域计算均值和标准差
        patch_size = 10
        x1 = max(0, sky_x - patch_size//2)
        x2 = min(w_roi, sky_x + patch_size//2)
        y1 = max(0, sky_y - patch_size//2)
        y2 = min(h_roi, sky_y + patch_size//2)
        patch = hsv_roi[int(y1):int(y2), int(x1):int(x2)]
        
        if patch.size == 0:
            print("天空点邻域为空")
            return None
        
        h_mean = np.mean(patch[:, :, 0])
        s_mean = np.mean(patch[:, :, 1])
        v_mean = np.mean(patch[:, :, 2])
        h_std = np.std(patch[:, :, 0])
        s_std = np.std(patch[:, :, 1])
        v_std = np.std(patch[:, :, 2])
        
        # 设定阈值范围
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
        
        # 4. 生成ROI内的天空掩膜
        roi_mask = cv2.inRange(hsv_roi, lower, upper)
        
        # 可选：形态学操作去除噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel)
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)
        
        # 5. 将掩膜放回原图尺寸
        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        full_mask[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi] = roi_mask
        
        # 6. 调用原有的地平线检测逻辑（复用 _detect_horizon_from_mask）
        horizon_params = HorizonDetector._detect_horizon_from_mask(
            full_mask,
            roi_y1=y_roi, roi_y2=y_roi+h_roi,
            roi_x1=x_roi, roi_x2=x_roi+w_roi
        )
        return horizon_params

    def detect_horizon_sam(self, image, roi, roi_sky_point=[[150, 10]],hsv_margin=[30, 50, 50]):
        """
        核心方法：检测地平线
        :param image: 输入图像（OpenCV读取的BGR格式）
        :param roi: ROI参数 (x, y, w, h)
        :param roi_sky_point: ROI内的天空点（相对ROI的坐标）
        :return: 地平线参数 (k, b)，None表示检测失败
        """
        # 1. 检查输入图像
        if image is None or len(image.shape) != 3:
            raise ValueError("无效的输入图像，必须是OpenCV读取的BGR格式图像")
        
        # 2. 裁剪ROI
        roi_image, (x1, y1, x2, y2) = self._crop_roi_safely(image, roi)
        
        # 3. SAM分割天空
        self.predictor.set_image(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
        masks, _, _ = self.predictor.predict(
            point_coords=np.array(roi_sky_point),
            point_labels=np.array([1]),
            multimask_output=False
        )
        
        # 4. 还原mask到原图尺寸
        h_img, w_img = image.shape[:2]
        full_sky_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        roi_sky_mask = (masks[0] * 255).astype(np.uint8)
        full_sky_mask[y1:y2, x1:x2] = roi_sky_mask
        
        # 5. 检测地平线
        horizon_params = self._detect_horizon_from_mask(
            full_sky_mask,
            roi_y1=y1, roi_y2=y2,
            roi_x1=x1, roi_x2=x2
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


# ====================== 测试使用示例 ======================
if __name__ == "__main__":
    # 配置参数
    SAM_CHECKPOINT = r"D:\program\HoriLane\models\sam_vit_b_01ec64.pth"
    IMAGE_PATH = r"D:\yolo\segmentation\runway_dataset\images\train\frame_000450.jpg"
    ROI = (400, 320, 300, 60)  # (x, y, w, h)
    ROI_SKY_POINT = [[150, 10]]  # ROI内的天空点
    
    # 初始化检测器
    detector = HorizonDetector(SAM_CHECKPOINT, model_type="vit_b", use_gpu=True)
    
    # 读取图像
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"图像读取失败：{IMAGE_PATH}")
    
    # 检测地平线
    horizon_params = detector.detect_horizon(image, ROI, ROI_SKY_POINT)
    draw_full_image_line(image,1/horizon_params[0],-(horizon_params[1] + 5)/horizon_params[0])
    detector.draw_roi(image,ROI)
    # 输出结果
    if horizon_params is not None:
        k, b = horizon_params
        print(f"地平线参数：k={k:.6f}, b={b:.2f} | 直线方程：y = {k:.6f}*x + {b:.2f}")
    else:
        print("地平线检测失败")
    cv2.waitKey(0)
    cv2.destroyAllWindows()