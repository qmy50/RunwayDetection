from ultralytics import YOLO
import cv2
import numpy as np
from utils import draw_full_image_line, smooth_line, get_shortest_line_end_y, get_division_points_by_vp_and_short_line, mirror_ipm_points
from ipm_transform_copy import BirdViewConverter
from sam_test import HorizonDetector
from filterpy.kalman import KalmanFilter
from KalmanFilter import HorizonKF,right_line_KF,left_line_KF

class RunwayDetector:
    """跑道检测类，封装状态管理和检测逻辑"""
    def __init__(self, yolo_model_path, sam_checkpoint, width, height):
        # 初始化模型
        self.model = YOLO(yolo_model_path)
        self.detector = HorizonDetector(sam_checkpoint, model_type="vit_b", use_gpu=True)
        
        # 视频尺寸
        self.width = width
        self.height = height
        
        # 常量定义
        self.SLOPE_THRESHOLD = 0.5
        self.MAX_HIST = 5
        self.MAX_EXTRA_FRAMES = 350
        self.YOLO_ON = False
        self.alpha = 0.7  # 灭点平滑因子
        
        # 初始化状态变量
        self.counter = 0
        self.yolo_last = 261.18
        self.last_valid_left = None
        self.last_valid_right = None
        self.first_left = True
        self.first_right = True
        self.last_valid_vp = [610, 261]
        self.last_valid_horizon = []
        self.division_frame_counter = 0
        self.enable_extra_drawing = True
        
        # IPM相关
        IPM_leftup = (530,290)
        IPM_leftdown = (200,504)
        IPM_rightup = (690,290)
        IPM_rightdown = (1100,504)
        self.left_valid_ipm = [IPM_leftdown, IPM_leftup]
        self.right_valid_ipm = [IPM_rightdown, IPM_rightup]
        self.IPM_points = [IPM_leftdown, IPM_rightdown, IPM_rightup, IPM_leftup]
        
        # 平滑相关
        self.left_raw_hist = []
        self.right_raw_hist = []
        
        # 地平线卡尔曼滤波
        self.horizon_kf = HorizonKF(1,[70,70])
        self.horizon_kf_init = False

        self.right_kf = right_line_KF(1,[5,5])
        self.left_kf = left_line_KF(1,[5,5])

    def process_frame(self, frame, out=None):
        """
        处理单帧图像，返回核心检测结果
        
        参数:
            frame: 输入视频帧
            out: 可选，cv2.VideoWriter对象，用于写入处理后的帧
        
        返回:
            dict: 包含以下核心结果
                - left_line: (k, b) 左线斜率和截距 (x = k*y + b)
                - right_line: (k, b) 右线斜率和截距 (x = k*y + b)
                - horizon_line: (k, b) 地平线斜率和截距 (y = k*x + b)
                - vanishing_point: (x, y) 灭点坐标
                - processed_frame: 处理后的图像（带绘制）
        """
        image = frame.copy()
        
        # 计数和额外绘制控制
        if self.enable_extra_drawing:
            self.division_frame_counter += 1
            if self.division_frame_counter > self.MAX_EXTRA_FRAMES:
                self.enable_extra_drawing = False
        
        # YOLO区域计算
        x1_yolo, y1_yolo, x2_yolo, y2_yolo = 0, self.yolo_last, self.width, self.height
        cx = (x1_yolo + x2_yolo) / 2
        cy = (y1_yolo + y2_yolo) / 2
        w = x2_yolo - x1_yolo
        h = y2_yolo - y1_yolo
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        
        # IPM变换和线提取
        left_list, right_list = [], []
        birdViewConverter = BirdViewConverter(image.copy(), self.IPM_points)
        birdViewConverter.birdview_to_origin()
        _, left_list = birdViewConverter.draw_line(birdViewConverter.bird_left_lines_x, birdViewConverter.bird_left_lines_y)
        _, right_list = birdViewConverter.draw_line(birdViewConverter.bird_right_lines_x, birdViewConverter.bird_right_lines_y)

        # 检测成功判断
        left_ipm_success = len(left_list) > 0 and (self.first_left or 
            (self.last_valid_left is not None and abs(left_list[0][0] - self.last_valid_left[0])/abs(self.last_valid_left[0]) < 0.2))
        right_ipm_success = len(right_list) > 0 and (self.first_right or 
            (self.last_valid_right is not None and abs(right_list[0][0] - self.last_valid_right[0])/abs(self.last_valid_right[0]) < 0.2))

        visible_y_min = max(0, y1)
        visible_y_max = min(self.height, y2)

        # 左线处理
        left_raw = None
        if left_ipm_success:
            cv2.putText(image, "left ipm success", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 1, cv2.LINE_AA)
            left_raw = np.average(left_list, axis=0)
        elif right_ipm_success and self.last_valid_vp[0] != 1 and self.last_valid_right is not None:
            cv2.putText(image, "left ipm fail, use right symmetric line", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
            k_r, b_r = self.last_valid_right
            vp_x = self.last_valid_vp[0]
            k_l = -k_r
            b_l = 2 * vp_x - b_r
            left_raw = (k_l, b_l)

        # 左线平滑
        left_lines = self.last_valid_left
        if left_raw is not None and not np.isnan(left_raw).any():
            self.left_raw_hist.append(left_raw)
            if len(self.left_raw_hist) > self.MAX_HIST:
                self.left_raw_hist.pop(0)
            left_lines = smooth_line(self.left_raw_hist, self.MAX_HIST)
            self.left_kf.predict()
            left_lines = self.left_kf.update(left_lines)
            self.last_valid_left = left_lines

        # 右线处理
        right_raw = None
        if right_ipm_success:
            cv2.putText(image, "right ipm success", (900, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 1, cv2.LINE_AA)
            right_raw = np.average(right_list, axis=0)
        elif left_ipm_success and self.last_valid_vp[0] != 1 and self.last_valid_left is not None:
            cv2.putText(image, "right ipm fail, use left symmetric line", (900, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
            k_l, b_l = self.last_valid_left
            vp_x = self.last_valid_vp[0]
            k_r = -k_l
            b_r = 2 * vp_x - b_l
            right_raw = (k_r, b_r)

        # 右线平滑
        right_lines = self.last_valid_right
        if right_raw is not None and not np.isnan(right_raw).any():
            self.right_raw_hist.append(right_raw)
            if len(self.right_raw_hist) > self.MAX_HIST:
                self.right_raw_hist.pop(0)
            right_lines = smooth_line(self.right_raw_hist, self.MAX_HIST)
            self.right_kf.predict()
            right_lines = self.right_kf.update(right_lines)
            self.last_valid_right = right_lines

        # 地平线检测和平滑
        vp_x, vp_y = self.last_valid_vp
        roi = (200, vp_y - 40, self.width-400, 70)
        ROI_SKY_POINT = [[(self.width-400)/2, 20]]
        horizon_result = self.detector.detect_horizon(image, roi, ROI_SKY_POINT)
        
        smoothed_horizon = []
        if horizon_result is not None:
            self.horizon_kf.predict()   
            smoothed_horizon = self.horizon_kf.update(horizon_result)

        # 地平线参数处理
        if not smoothed_horizon:
            k_h, b_h = self.last_valid_horizon if self.last_valid_horizon else (0, 0)
        else:
            k_h, b_h = smoothed_horizon
            b_h += 5
            self.last_valid_horizon = (k_h, b_h)
        
        # 绘制地平线
        if k_h != 0:    
            draw_full_image_line(image, 1/k_h, -(b_h)/k_h if k_h != 0 else 0)
        # self.detector.draw_roi(image, roi)

        # 计算灭点
        new_vp_x, new_vp_y = vp_x, vp_y
        if left_ipm_success and right_ipm_success and left_lines is not None and right_lines is not None:
            k_l, b_l = left_lines
            k_r, b_r = right_lines
            if abs(k_l - k_r) > 1e-6:
                new_vp_y = int((b_r - b_l) / (k_l - k_r))
                new_vp_x = int(k_l * new_vp_y + b_l)
        elif left_ipm_success and not right_ipm_success and left_lines is not None:
            k_l, b_l = left_lines
            # 转换地平线参数为 x = k*y + b 形式
            m_h, c_h = (k_h, b_h)
            k_h_conv = 1.0 / m_h if abs(m_h) > 1e-6 else 0
            b_h_conv = -c_h / m_h if abs(m_h) > 1e-6 else 0
            new_vp_y = int((b_h_conv - b_l) / (k_l - k_h_conv))
            new_vp_x = int(k_l * new_vp_y + b_l)
        elif right_ipm_success and not left_ipm_success and right_lines is not None:
            k_r, b_r = right_lines
            m_h, c_h = (k_h, b_h)
            k_h_conv = 1.0 / m_h if abs(m_h) > 1e-6 else 0
            b_h_conv = -c_h / m_h if abs(m_h) > 1e-6 else 0
            new_vp_y = int((b_h_conv - b_r) / (k_r - k_h_conv))
            new_vp_x = int(k_r * new_vp_y + b_r)
        
        # 灭点平滑
        if self.counter == 0:
            smoothed_vp = [new_vp_x, new_vp_y]
        else:
            smoothed_vp = [
                int(self.alpha * new_vp_x + (1-self.alpha) * self.last_valid_vp[0]),
                int(self.alpha * new_vp_y + (1-self.alpha) * self.last_valid_vp[1])
            ]
        self.last_valid_vp = smoothed_vp
        vp_x, vp_y = smoothed_vp
        
        # 绘制灭点
        cv2.circle(image, (vp_x, vp_y), 8, (255, 255, 0), -1)

        # 绘制左右线
        IPM_flag_left, IPM_flag_right = False, False
        current_ipm_leftup, current_ipm_leftdown = self.left_valid_ipm[1], self.left_valid_ipm[0]
        current_ipm_rightup, current_ipm_rightdown = self.right_valid_ipm[1], self.right_valid_ipm[0]

        # 绘制左线
        if left_lines is not None:
            k_l, b_l = left_lines[0], left_lines[1]
            if abs(k_l) > 0.01:
                x_start_l = int(k_l * visible_y_min + b_l)
                x_end_l = int(k_l * visible_y_max + b_l)
                cv2.line(image, (x_start_l, visible_y_min), (x_end_l, visible_y_max), (0,0,255), 3)
                
                # 等分点处理
                if self.enable_extra_drawing and vp_y != 1 and right_lines is not None and left_ipm_success:
                    k_r_tmp, b_r_tmp = right_lines[0], right_lines[1]
                    short_end_y, vp_y_min = get_shortest_line_end_y(k_l, b_l, k_r_tmp, b_r_tmp, vp_x, vp_y, self.width, self.height)
                    left_division_points = get_division_points_by_vp_and_short_line(k_l, b_l, vp_y, vp_y_min, short_end_y,
                                                                                x_min=0, x_max=self.width, division_num=20)
                    if len(left_division_points) >= 20:
                        current_ipm_leftup = left_division_points[10]
                        current_ipm_leftdown = left_division_points[19]
                        IPM_flag_left = True

        # 绘制右线
        if right_lines is not None:
            k_r, b_r = right_lines[0], right_lines[1]
            if abs(k_r) > 0.01:
                x_start_r = int(k_r * visible_y_min + b_r)
                x_end_r = int(k_r * visible_y_max + b_r)
                cv2.line(image, (x_start_r, visible_y_min), (x_end_r, visible_y_max), (0, 0, 255), 3)
                
                # 等分点处理
                if self.enable_extra_drawing and vp_y != 1 and left_lines is not None and right_ipm_success:
                    k_l_tmp, b_l_tmp = left_lines[0], left_lines[1]
                    short_end_y, vp_y_min = get_shortest_line_end_y(k_l_tmp, b_l_tmp, k_r, b_r, vp_x, vp_y, self.width, self.height)
                    right_division_points = get_division_points_by_vp_and_short_line(k_r, b_r, vp_y, vp_y_min, short_end_y,
                                                                                x_min=0, x_max=self.width, division_num=20)
                    if len(right_division_points) >= 20:
                        current_ipm_rightup = right_division_points[10]
                        current_ipm_rightdown = right_division_points[19]
                        IPM_flag_right = True

        # 更新IPM点
        if IPM_flag_left:
            self.left_valid_ipm = [current_ipm_leftdown, current_ipm_leftup]
        if IPM_flag_right:
            self.right_valid_ipm = [current_ipm_rightdown, current_ipm_rightup]
        
        # 对称补全IPM点
        mid_x = vp_x if vp_x != 1 else self.width // 2
        if IPM_flag_left and not IPM_flag_right:
            self.right_valid_ipm = mirror_ipm_points(self.left_valid_ipm, mid_x, self.width, self.height)
        elif not IPM_flag_left and IPM_flag_right:
            self.left_valid_ipm = mirror_ipm_points(self.right_valid_ipm, mid_x, self.width, self.height)
        
        # 更新IPM_points
        self.IPM_points = [
            self.left_valid_ipm[0],
            self.right_valid_ipm[0],
            self.right_valid_ipm[1],
            self.left_valid_ipm[1]
        ]

        # 写入视频（如果提供了out）
        if out is not None:
            out.write(image)
        
        # 更新计数器
        self.counter += 1
        if self.counter > 3:
            self.first_left = False
            self.first_right = False
        
        # 构造返回结果
        result = {
            "left_line": left_lines if left_lines is not None else (0.0, 0.0),
            "right_line": right_lines if right_lines is not None else (0.0, 0.0),
            "horizon_line": (k_h, b_h),
            "vanishing_point": (vp_x, vp_y),
            "processed_frame": image
        }
        
        return result

# 使用示例
def main():
    # 视频路径配置
    video_path = r"D:\program\test_video_3.mp4"
    output_path = r"D:\yolo\qq_3045834499\yolov8-42\runs\detect\predict\save_viedo_1.avi"
    yolo_model_path = r"D:\yolo\runs\detect\train4\weights\best.pt"
    sam_checkpoint = r"D:\program\HoriLane\models\sam_vit_b_01ec64.pth"
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # 获取视频参数
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 初始化检测器
    detector = RunwayDetector(yolo_model_path, sam_checkpoint, width, height)
    
    # 逐帧处理
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理帧，获取核心结果
        result = detector.process_frame(frame, out)
        
        # 打印核心结果（可选）
        print(f"Frame {detector.counter}:")
        print(f"  左线: k={result['left_line'][0]:.4f}, b={result['left_line'][1]:.4f}")
        print(f"  右线: k={result['right_line'][0]:.4f}, b={result['right_line'][1]:.4f}")
        print(f"  地平线: k={result['horizon_line'][0]:.4f}, b={result['horizon_line'][1]:.4f}")
        print(f"  灭点: ({result['vanishing_point'][0]}, {result['vanishing_point'][1]})")
        
        # 显示处理后的帧
        # cv2.imshow("Runway Detection", result['processed_frame'])
        
        # 键盘控制
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite("save_image.jpg", result['processed_frame'])
        elif key == ord('q'):
            break
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("视频检测完成！")

if __name__ == "__main__":
    main()