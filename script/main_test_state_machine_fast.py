from ultralytics import YOLO
import cv2
import numpy as np
from tqdm import tqdm

from utils_final import canny_vertical_edges, Hough_detection, average, get_line_intersection
from utils import draw_full_image_line
from KalmanFilter import YOLO_KF, right_line_KF, left_line_KF, HorizonKF
from RunwayDetector_for_video import RunwayDetector
import config
from Horizon_fast import HorizonDetector


class VideoProcessor:
    def __init__(self, video_path, output_path, yolo_model_path, sam_checkpoint):
        # 视频相关
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video.")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            (self.width, self.height)
        )

        # 模型
        self.yolo_model = YOLO(yolo_model_path)
        self.runway_detector = RunwayDetector(yolo_model_path, sam_checkpoint, self.width, self.height)

        # 卡尔曼
        self.yolo_kf = YOLO_KF(1)
        self.left_line_kf = left_line_KF(1, [5, 5])
        self.right_line_kf = right_line_KF(1, [5, 5])
        self.horizon_kf = HorizonKF(1, [70, 70])

        # 历史
        self.left_line_hist = []
        self.right_line_hist = []
        self.last_valid_horizon = None
        self.last_valid_left = None
        self.last_valid_right = None
        self.left_k_mean = None
        self.right_k_mean = None

        self.MAX_HIST = 5
        self.K_THRESHOLD = 0.5
        self.frame_counter = 0

        # 进度条
        self.pbar = tqdm(total=self.total_frames, desc="处理视频帧", colour="green")

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 复制原图，用于绘制结果
            display_frame = frame.copy()
            # 用于UFLD处理的帧（默认全图）
            ufld_frame = frame
            # ROI坐标（默认全图）
            roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, self.width, self.height
            # 是否有有效YOLO框
            has_yolo = False

            # ===================== 前期YOLO辅助（前50%帧） =====================
            if self.frame_counter < 0.5 * self.total_frames:
                results = self.yolo_model.predict(frame, verbose=False)
                if len(results) > 0 and len(results[0].boxes) > 0:
                    box = results[0].boxes[0]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    # YOLO框卡尔曼平滑
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1
                    self.yolo_kf.predict()
                    cx, cy, w, h = self.yolo_kf.update(np.array([[cx], [cy], [w], [h]]))
                    x1 = int(cx - w / 2)
                    y1 = int(cy - h / 2)
                    x2 = int(cx + w / 2)
                    y2 = int(cy + h / 2)

                    # 边界保护（防止越界）
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(self.width, x2)
                    y2 = min(self.height, y2)

                    # 标记有YOLO框
                    has_yolo = True
                    roi_x1, roi_y1, roi_x2, roi_y2 = x1, y1, x2, y2
                    # 裁剪ROI区域给UFLD
                    ufld_frame = frame[y1:y2, x1:x2].copy()

                    # 绘制YOLO框
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

                    # 原有YOLO辅助线、灭点、地平线（全图画）
                    self._process_yolo_aux(display_frame, x1, y1, x2, y2)

            # ===================== UFLD跑道检测（ROI内/全图） =====================
            if has_yolo:
                # 有YOLO：在裁剪ROI上跑UFLD，结果画回原图
                self.runway_detector.process_frame(
                    ufld_frame,
                    out=None,          # 不直接写文件
                    IPM_assist=False,
                    roi=(roi_x1, roi_y1, roi_x2, roi_y2),  # 传入原图ROI坐标
                    display_frame=display_frame  # 结果画到原图
                )
            else:
                # 无YOLO：全图跑UFLD
                self.runway_detector.process_frame(
                    display_frame,
                    out=None,
                    IPM_assist=False
                )

            # 写入输出
            self.out.write(display_frame)

            self.frame_counter += 1
            self.pbar.update(1)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        self._cleanup()

    def _process_yolo_aux(self, frame, x1, y1, x2, y2):
        """YOLO辅助：边缘、直线、灭点、地平线（全图画）"""
        pt1 = (x1, y1)
        pt2 = (x2, y2)
        visible_y_min = max(0, y1)
        visible_y_max = min(self.height, y2)
        h = y2 - y1
        cx = (x1 + x2) / 2

        edges, vertical_edges = canny_vertical_edges(frame, pt1, pt2)
        original_lines = Hough_detection(vertical_edges, h)

        left_list, right_list = [], []
        if original_lines is not None:
            original_lines = original_lines[0:2, :, :].reshape(-1, 4)
            left_list, right_list = average(
                original_lines, cx,
                left_k_mean=self.left_k_mean,
                right_k_mean=self.right_k_mean,
                k_threshold=self.K_THRESHOLD
            )

        # 左线
        left_k_b = self._update_line(left_list, self.left_line_hist, self.last_valid_left, self.left_k_mean)
        if left_k_b is not None:
            self.last_valid_left = left_k_b
            self.left_k_mean = left_k_b[0]
            self.left_line_kf.predict()
            left_k_b = self.left_line_kf.update(left_k_b)
            self._draw_line(frame, left_k_b, visible_y_min, visible_y_max, (0, 0, 255))

        # 右线
        right_k_b = self._update_line(right_list, self.right_line_hist, self.last_valid_right, self.right_k_mean)
        if right_k_b is not None:
            self.last_valid_right = right_k_b
            self.right_k_mean = right_k_b[0]
            self.right_line_kf.predict()
            right_k_b = self.right_line_kf.update(right_k_b)
            self._draw_line(frame, right_k_b, visible_y_min, visible_y_max, (0, 0, 255))

        # 灭点 + 地平线
        self._draw_vanishing_point_and_horizon(frame, left_k_b, right_k_b)

    def _update_line(self, line_list, hist_list, last_valid, k_mean):
        if len(line_list) > 0:
            k_b = np.average(line_list, axis=0)
            hist_list.append(k_b)
            if len(hist_list) > self.MAX_HIST:
                hist_list.pop(0)
            return k_b
        elif len(hist_list) >= 2:
            weight = np.exp(np.linspace(0, 1, len(hist_list)))
            weight /= np.sum(weight)
            return np.average(hist_list, axis=0, weights=weight)
        elif last_valid is not None:
            return last_valid
        return None

    def _draw_line(self, frame, k_b, y_min, y_max, color):
        k, b = k_b[0], k_b[1]
        if abs(k) < 100:
            x_start = int(k * y_min + b)
            x_end = int(k * y_max + b)
            cv2.line(frame, (x_start, y_min), (x_end, y_max), color, 3)

    def _draw_vanishing_point_and_horizon(self, frame, left_k_b, right_k_b):
        if left_k_b is None or right_k_b is None:
            return

        k_left, b_left = left_k_b
        k_right, b_right = right_k_b
        intersection = get_line_intersection(k_left, b_left, k_right, b_right)
        if intersection is None:
            return

        vp_x, vp_y = intersection
        cv2.circle(frame, (int(vp_x), int(vp_y)), 8, (255, 255, 0), -1)

        roi = (200, int(vp_y) - 40, self.width - 400, 70)
        ROI_SKY_POINT = [[(self.width - 400) / 2, 20]]
        horizon_result = HorizonDetector.detect_horizon(frame, roi, ROI_SKY_POINT)

        smoothed_horizon = None
        if horizon_result is not None:
            self.horizon_kf.predict()
            smoothed_horizon = self.horizon_kf.update(horizon_result)

        if smoothed_horizon is None:
            k_h, b_h = self.last_valid_horizon if self.last_valid_horizon else (0, 0)
        else:
            k_h, b_h = smoothed_horizon
            self.last_valid_horizon = (k_h, b_h)

        if k_h != 0:
            draw_full_image_line(frame, 1 / k_h, -b_h / k_h)

    def _cleanup(self):
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        self.pbar.close()
        print("处理完成，视频已保存")


# ===================== main 函数 =====================
def main():
    video_path = r"D:\program\HoriLane\RunwayDetection\test_video.mp4"
    output_path = r"D:\yolo\qq_3045834499\yolov8-42\runs\detect\predict\save_viedo.mp4"
    yolo_model_path = r"D:\yolo\runs\detect\train4\weights\best.pt"
    sam_checkpoint = r"D:\program\HoriLane\models\sam_vit_b_01ec64.pth"

    processor = VideoProcessor(
        video_path, output_path,
        yolo_model_path, sam_checkpoint
    )
    processor.run()


if __name__ == "__main__":
    main()