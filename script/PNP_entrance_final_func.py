import cv2
import numpy as np
import math
import os
from ultralytics import YOLO
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.optimize import linear_sum_assignment  # 匈牙利算法
from utils import edge_detection_in_mask, detect_points_from_edges, distance, validate_coordinate, get_euler_angle, get_camera_position,expand_mask_dilation 
import config

# ========== 配置 ==========
MODEL_PATH = r'D:\program\Entrance\models\best.pt'
VIDEO_PATH = r'D:\yolo\qq_3045834499\yolov8-42\my_dataset\test\test_video.mp4'
OUTPUT_PATH = r'D:\program\Entrance\runway_predict\pnp_output_video.mp4'
DILATE_KERNEL = 3
DILATE_ITER = 2
CONF_THRESHOLD = 0.25
CANNY_LOW = 100
CANNY_HIGH = 150

# 世界坐标系下跑道四边形的3D点（单位：毫米）
points_3D = np.array([[-149, -105, 0],
                      [-149, 105, 0],
                      [149, 105, 0],
                      [149, -105, 0]], dtype="double")

# 相机内参（从PNP_1.py复制）
camera_matrix = np.array([[1.2169e+03, 0, 846.0019],
                          [0, 1.2163e+03, 632.0667],
                          [0, 0, 1]], dtype="double")

# 畸变系数
dist = np.array([0.2217, -0.8115, 4.233e-4, -0.0014, 0.8558], dtype="double")

# ========== 主程序 ==========
class EntranceDetector:
    def __init__(self, MODEL_PATH, MODE='show'):
        self.model = YOLO(MODEL_PATH)
        
        # 轨迹记录
        self.pose_x, self.pose_y, self.pose_z = [], [], []
        self.P_old = None
        self.mode = MODE
        self.esc_pressed = False  # 新增：标记是否按下ESC
    
    def process_frame(self, frame, out=None):        
        # 如果按下ESC，直接返回帧
        if self.esc_pressed:
            return frame
        
        # 1. YOLO预测（得到分割掩码）
        results = self.model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
        h, w = frame.shape[:2]
        
        # 2. 提取膨胀掩码（取第一个掩码）
        expanded_masks = []  # 存储膨胀后的掩码
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        for r in results:
            if r.masks is not None:
                for poly in r.masks.xy:
                    mask_img = np.zeros((h, w), dtype=np.uint8)
                    poly = poly.astype(np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask_img, [poly], 255)     # 将多边形区域填充为白色（255）
                    expanded = expand_mask_dilation(mask_img, DILATE_KERNEL, DILATE_ITER)
                    expanded_masks.append(expanded)
                    combined_mask = cv2.bitwise_or(combined_mask, expanded)   # 合并所有膨胀掩码
        
        # 如果没有检测到掩码，直接返回原图
        if len(expanded_masks) == 0:
            return frame
        
        # 3. 边缘检测（以第一个掩码为例）
        edges = edge_detection_in_mask(frame, expanded_masks[0], CANNY_LOW, CANNY_HIGH)
        
        # 4. 检测多边形顶点
        points = detect_points_from_edges(edges, min_contour_length=30)
        if points is None or not config.DRAW_PNP:
            # 未检测到四边形，直接返回帧
            return frame
        
        # 5. 顶点顺序匹配（使用匈牙利算法）
        match_success = True
        A, B, C, D = None, None, None, None
        
        if self.P_old is None:
            # 第一帧，直接采用检测到的顺序
            if len(points) >= 4:
                A, B, C, D = points[0], points[1], points[2], points[3]
                self.P_old = [A, B, C, D]
            else:
                match_success = False
        else:
            # 构建距离矩阵 (4x4)
            dist_matrix = np.zeros((4, 4))
            for i, pt in enumerate(points):
                if points[i][1] >= 0.99 * h:
                    config.COME_TO_END = True
                for j, p_old in enumerate(self.P_old):
                    dist_matrix[i, j] = distance(pt, p_old)
            
            # 匈牙利算法求最小权匹配
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            
            # 根据匹配结果重新排列当前帧点
            matched = [None] * 4
            for r, c in zip(row_ind, col_ind):
                matched[c] = points[r]
            
            # 检查是否所有四个点都分配
            if any(x is None for x in matched):
                print("匈牙利匹配异常")
                match_success = False
            else:
                A, B, C, D = matched[0], matched[1], matched[2], matched[3]
                self.P_old = [A, B, C, D]
        
        if not match_success or config.COME_TO_END:
            # 匹配失败或到达终点，返回原帧
            return frame
        
        # 6. 构建2D点数组（注意顺序对应3D点）
        points_2D = np.array([A, B, C, D], dtype="double")
        
        # 7. PnP求解
        ret, rvec, tvec = cv2.solvePnP(points_3D, points_2D, camera_matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ret:
            # 解算失败，返回原帧
            return frame
        
        # 8. 绘制坐标轴
        axis_3d = np.array([(0,0,0), (200,0,0), (0,200,0), (0,0,200)], dtype=np.float32)
        axis_2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist)
        O_ = validate_coordinate(axis_2d[0][0], frame.shape)
        X_end = validate_coordinate(axis_2d[1][0], frame.shape)
        Y_end = validate_coordinate(axis_2d[2][0], frame.shape)
        Z_end = validate_coordinate(axis_2d[3][0], frame.shape)
        
        # 9. 绘制顶点（白色圆点）和标签
        A_int = (int(A[0]), int(A[1]))
        B_int = (int(B[0]), int(B[1]))
        C_int = (int(C[0]), int(C[1]))
        D_int = (int(D[0]), int(D[1]))
        
        # 10. 显示位姿信息
        xt, yt, zt = tvec.ravel()
        xr, yr, zr = rvec.ravel()

        if self.mode == 'show':
            # 绘制坐标轴
            cv2.line(frame, O_, X_end, (50, 255, 50), 3)
            cv2.line(frame, O_, Y_end, (50, 50, 255), 3)
            cv2.line(frame, O_, Z_end, (255, 50, 50), 3)
            cv2.putText(frame, "X", X_end, cv2.FONT_HERSHEY_SIMPLEX, 1, (5,150,5), 2)
            cv2.putText(frame, "Y", Y_end, cv2.FONT_HERSHEY_SIMPLEX, 1, (5,5,150), 2)
            cv2.putText(frame, "Z", Z_end, cv2.FONT_HERSHEY_SIMPLEX, 1, (150,5,5), 2)
        
            # 绘制顶点和标签
            cv2.circle(frame, A_int, 5, (255,255,255), -1)
            cv2.circle(frame, B_int, 5, (255,255,255), -1)
            cv2.circle(frame, C_int, 5, (255,255,255), -1)
            cv2.circle(frame, D_int, 5, (255,255,255), -1)
            cv2.putText(frame, "A", (A_int[0]+15, A_int[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            cv2.putText(frame, "B", (B_int[0]+15, B_int[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            cv2.putText(frame, "C", (C_int[0]+15, C_int[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            cv2.putText(frame, "D", (D_int[0]+15, D_int[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        
            # 显示旋转/平移向量
            rvec_str = f'rotation_vector: ({xr:.2f}, {yr:.2f}, {zr:.2f})'
            tvec_str = f'translation_vector: ({xt:.2f}, {yt:.2f}, {zt:.2f})'
            cv2.putText(frame, tvec_str, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
            cv2.putText(frame, rvec_str, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        
        # 11. 计算并记录相机位置
        roll, pitch, yaw, roll_deg, pitch_deg, yaw_deg = get_euler_angle(rvec)
        cam_x, cam_y, cam_z = get_camera_position(rvec, tvec)
        
        # 记录轨迹（只记录有效数据）
        self.pose_x.append(cam_x)
        self.pose_y.append(cam_y)
        self.pose_z.append(cam_z)
        
        # 显示位置和欧拉角
        position_str = f'position: ({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f}) mm'
        euler_str = f'euler(deg): roll={roll_deg:.1f}, pitch={pitch_deg:.1f}, yaw={yaw_deg:.1f}'
        if self.mode == 'show':
            cv2.putText(frame, position_str, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
            cv2.putText(frame, euler_str, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        
        # 显示实时画面并检测ESC
        if self.mode == 'show':
            # cv2.imshow("Result", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                self.esc_pressed = True
        
        # 写入视频帧（确保out不为空）
        if out is not None and not self.esc_pressed:
            out.write(frame)
        
        return frame
            
def main(PLOT=True):
    # 初始化视频读取
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {VIDEO_PATH}")
    
    # 获取视频参数
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(0.4 * cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 限制处理帧数
    
    # 创建输出视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    
    # 初始化检测器
    entrancedetector = EntranceDetector(MODEL_PATH=MODEL_PATH, MODE='show')
    
    # 进度条
    pbar = tqdm(total=total_frames, desc="处理视频帧")
    counter = 0
    
    # 帧处理循环
    while counter < total_frames and not entrancedetector.esc_pressed:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理帧
        frame = entrancedetector.process_frame(frame, out)
        
        # 写入帧（双重保障）
        if frame is not None:
            out.write(frame)
        
        counter += 1
        pbar.update(1)
    
    # 资源释放
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    pbar.close()

    # ========== 绘制轨迹（关键：修正缩进！） ==========
    if PLOT:
        # 检查轨迹数据是否为空
        if len(entrancedetector.pose_x) == 0:
            print("警告：没有收集到任何轨迹数据，无法绘制3D轨迹图")
            print(f"总处理帧数: {counter}, PnP成功帧数: 0")
            return
        
        # 绘制3D轨迹图
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection='3d')
        
        # 轨迹数据
        pose_x = np.array(entrancedetector.pose_x)
        pose_y = np.array(entrancedetector.pose_y)
        pose_z = np.array(entrancedetector.pose_z)
        frame_nums = np.arange(len(pose_x))
        norm_frame = frame_nums / frame_nums.max()
        
        # 绘制散点和轨迹线
        scatter = ax.scatter3D(pose_x, pose_y, pose_z, c=norm_frame, cmap='viridis', s=5, alpha=0.8)
        ax.plot3D(pose_x, pose_y, pose_z, color='gray', linewidth=1, alpha=0.6)
        
        # 颜色条
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label(f"Frame Number (total:{len(pose_x)})")
        cbar.set_ticks(np.linspace(0, 1, 5))
        cbar.set_ticklabels(np.linspace(0, frame_nums.max(), 5).astype(int))
        
        # 坐标轴标签
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_title('Camera Trajectory (World Coordinate)')
        
        # 保存图片（取消注释）
        plt.savefig('trajectory.png', dpi=300, bbox_inches='tight')
        print("轨迹图已保存为 trajectory.png")
        
        # 显示图片
        plt.show()
    
    # 输出统计信息
    print(f"总处理帧数: {counter}, PnP成功帧数: {len(entrancedetector.pose_x)}")

if __name__ == "__main__":
    main(PLOT=True)