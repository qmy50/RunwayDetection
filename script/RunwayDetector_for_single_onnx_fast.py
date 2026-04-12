import cv2
import numpy as np
from utils import draw_full_image_line
from Horizon_fast import HorizonDetector
import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
UFLD_DIR = BASE_DIR / "UFLDv2"
sys.path.append(str(BASE_DIR))
sys.path.append(str(UFLD_DIR))  
from UFLDv2.demo_img_onnx import UFLDDetector
import config


class RunwayDetector:
    """跑道检测类，只保留 UFLD 核心逻辑"""
    def __init__(self, width, height):
        # 初始化模型
        self.detector = HorizonDetector()
        
        self.width = width
        self.height = height
        
        # 初始化状态变量
        self.counter = 0
        self.last_valid_left = None
        self.last_valid_right = None
        self.last_valid_vp = [610, 261]
        self.last_valid_horizon = []

        self.UFLD_Detector = UFLDDetector(r'/home/tianbot/XTDrone/RunwayDetectionV1/script/model/model_best.onnx')

    def process_frame(self, image):
        """
        处理单帧图像（纯 UFLD 版本）
        返回:
            left_line, right_line, horizon_line, vanishing_point, processed_frame
        """
        # ==================== UFLD 检测（唯一保留的检测）====================
        result = self.UFLD_Detector.detect(image, draw=True)
        left_lines = result.get('left_line', None)
        right_lines = result.get('right_line', None)
        # if left_lines is not None: 
        #     print("left line is not None")
        # if right_lines is not None:
        #     print("right line is not None")

        vp_x = 0
        vp_y = 0
        # ==================== 灭点计算 ====================
        if left_lines is not None and right_lines is not None:
            k_l, b_l = left_lines
            k_r, b_r = right_lines
            if abs(k_l - k_r) > 1e-6:
                vp_y = int((b_r - b_l) / (k_l - k_r))
                vp_x = int(k_l * vp_y + b_l)
        if vp_x == 0 or vp_y == 0:
            return {
                "vanishing_point": None,
                "processed_frame": image
            }

        # # ==================== 地平线检测 ====================
        # roi = (vp_x - 200, vp_y - 20, 400, 70)
        # ROI_SKY_POINT = [[200, 21]]
        # # cv2.rectangle(image,(roi[0],roi[1]),(roi[0]+roi[2],roi[1]+roi[3]),color=(0,255,0),thickness=5)
        # # cv2.circle(image,(int(ROI_SKY_POINT[0][0]),ROI_SKY_POINT[0][1]),radius=30,color=(255,255,255),thickness=20)
        # horizon_result = self.detector.detect_horizon(image, roi, ROI_SKY_POINT)
        # k_h, b_h = horizon_result if horizon_result is not None else (10,10)
        # b_h += 3
        # # cv2.circle(image, (vp_x, vp_y), 8, (255, 255, 0), -1)
        # # 绘制地平线
        # if k_h != 0:
        #     draw_full_image_line(image, 1/k_h, -b_h/k_h if k_h != 0 else 0,thickness = 8)

        # # ==================== 绘制最终车道线 ====================
        # draw_full_image_line(image, left_lines[0], left_lines[1], color=(0,0,255), thickness = 8)
        # draw_full_image_line(image, right_lines[0], right_lines[1], color=(0,0,255), thickness = 8)

        return {
            "left_line": left_lines,
            "right_line": right_lines,
            "vanishing_point": (vp_x, vp_y),
            "processed_frame": image
        }