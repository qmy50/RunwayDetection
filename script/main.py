from ultralytics import YOLO
import cv2
import numpy as np
from utils_final import canny_vertical_edges, Hough_detection, average, get_line_intersection, get_point_on_line, get_five_division_points
from utils import get_shortest_line_end_y,get_division_points_by_vp_and_short_line,draw_full_image_line
from KalmanFilter import YOLO_KF, right_line_KF, left_line_KF,HorizonKF
from tqdm import tqdm
from PNP_entrance_final_func import EntranceDetector
from IPM_func import RunwayDetector
import config
from sam_test import HorizonDetector

model = YOLO(r"D:\yolo\runs\detect\train4\weights\best.pt")


def main():
    yolo_kf = YOLO_KF(1)
    left_line_kf = left_line_KF(1,[5,5])
    right_line_kf = right_line_KF(1,[5,5])
    horizon_kf = HorizonKF(1,[70,70])
    cap = cv2.VideoCapture(r"D:\yolo\qq_3045834499\yolov8-42\my_dataset\test\test_video.mp4")
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频帧率: {fps} FPS")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    output_video_path = r"D:\yolo\qq_3045834499\yolov8-42\runs\detect\predict\save_viedo.avi"
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    pbar = tqdm(total=total_frames, desc="处理视频帧",colour="green")

    left_line_hist = []
    right_line_hist = []
    last_valid_horizon = None
    
    last_valid_left = None
    last_valid_right = None 
    YOLO_ON = True
    MAX_HIST = 5
    counter = 0
    GOON = True
    yolo_last = 0
    IPM_ON = False
    ipm_left = False
    ipm_right = False

    left_k_mean = None
    right_k_mean = None
    K_THRESHOLD = 0.5  # 注意：新k值通常较小，此阈值可能需要调整

    division_frame_counter = 0
    enable_extra_drawing = False
    MAX_EXTRA_FRAMES = 20
    
    coord_file = open('division_points.txt', 'w', encoding='utf-8')
    coord_file.write("帧号,左线1/5点(x,y),左线4/5点(x,y),右线1/5点(x,y),右线4/5点(x,y)\n")
    yolo_model_path = r"D:\yolo\runs\detect\train4\weights\best.pt"
    sam_checkpoint = r"D:\program\HoriLane\models\sam_vit_b_01ec64.pth"
    MODEL_PATH = r'D:\program\Entrance\models\best.pt'

    runwayDetector = RunwayDetector(yolo_model_path, sam_checkpoint, width, height)
    entranceDetector = EntranceDetector(MODEL_PATH=MODEL_PATH,MODE='show')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        image = frame.copy()
        
        if YOLO_ON:
            results = model.predict(frame, verbose=False) 
            if len(results) > 0 and len(results[0].boxes) > 0:
                GOON = True
                box = results[0].boxes[0]
                x1_yolo, y1_yolo, x2_yolo, y2_yolo = map(int, box.xyxy[0].tolist())

                if abs(x1_yolo - x2_yolo) > 0.95 * width:
                    YOLO_ON = False
                    yolo_last = y1_yolo
                    enable_extra_drawing = True
                    division_frame_counter = 0
                elif not config.COME_TO_END :
                    # x1_crop = max(0,x1_yolo)
                    # y1_crop = max(0,y1_yolo)
                    # x2_crop = min(width,x2_yolo)
                    # y2_crop = min(height,y2_yolo)
                    # cropped_image = image[y1_crop:y2_crop,x1_crop:x2_crop]
                    # if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
                    processed_cropped = entranceDetector.process_frame(image, out=None)
                    # print(config.COME_TO_END)
                    if processed_cropped is not None:
                        image = processed_cropped
            else:
                GOON = False
        else:
            GOON = True
            x1_yolo, y1_yolo, x2_yolo, y2_yolo = 0, yolo_last, width, height
            if enable_extra_drawing:
                division_frame_counter += 1
                if division_frame_counter > MAX_EXTRA_FRAMES:
                    enable_extra_drawing = False
        
        if GOON and not IPM_ON:
            if YOLO_ON:
                x1_yolo, y1_yolo, x2_yolo, y2_yolo = map(int, box.xyxy[0].tolist())
            cx = (x1_yolo + x2_yolo) / 2
            cy = (y1_yolo + y2_yolo) / 2
            w = x2_yolo - x1_yolo
            h = y2_yolo - y1_yolo
            
            yolo_kf.predict()
            cx, cy, w, h = yolo_kf.update(np.array([[cx],[cy],[w],[h]]))
            
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            
            pt1 = (x1, y1)
            pt2 = (x2, y2)

            edges, vertical_edges = canny_vertical_edges(image, pt1, pt2)
            # cv2.imshow("edges", edges)
            # cv2.imshow("vertical_edges", vertical_edges)
            original_lines = Hough_detection(vertical_edges, h)
            
            left_list, right_list = [], []
            if original_lines is not None:
                original_lines = original_lines[0:2, :, :]
                original_lines = original_lines.reshape(-1, 4)
                left_list, right_list = average(original_lines, cx, 
                                               left_k_mean=left_k_mean,
                                               right_k_mean=right_k_mean,
                                               k_threshold=K_THRESHOLD)

            visible_y_min = max(0, y1)
            visible_y_max = min(height, y2)

            '''LEFT LINE (x = k_left * y + b_left)'''
            left_k_b = None
            if len(left_list) > 0:
                left_k_b = np.average(left_list, axis=0)  # [k, b]
                left_line_hist.append(left_k_b)
                if len(left_line_hist) > MAX_HIST:
                    left_line_hist.pop(0)
                left_k_mean = left_k_b[0]
            
            if len(left_line_hist) >= 2:
                weight = np.exp(np.linspace(0, 1, len(left_line_hist)))
                weight = weight / np.sum(weight)
                left_k_b = np.average(left_line_hist, axis=0, weights=weight)
                left_k_mean = left_k_b[0]
                
            flag_l = left_k_b is not None and not np.isnan(left_k_b).any()
            if flag_l:                
                last_valid_left = left_k_b
                if left_k_mean is None:
                    left_k_mean = left_k_b[0]
            else:
                left_k_b = last_valid_left 
                if left_k_b is not None and left_k_mean is None:
                    left_k_mean = left_k_b[0]


            '''RIGHT LINE (x = k_right * y + b_right)'''
            right_k_b = None
            if len(right_list) > 0:
                right_k_b = np.average(right_list, axis=0)
                right_line_hist.append(right_k_b)
                if len(right_line_hist) > MAX_HIST:
                    right_line_hist.pop(0)
                right_k_mean = right_k_b[0]
                    
            if len(right_line_hist) >= 2:
                weight = np.exp(np.linspace(0, 1, len(right_line_hist)))
                weight = weight / np.sum(weight)
                right_k_b = np.average(right_line_hist, axis=0, weights=weight)
                right_k_mean = right_k_b[0]

            flag_r = right_k_b is not None and not np.isnan(right_k_b).any()
            if flag_r:
                last_valid_right = right_k_b
                if right_k_mean is None:
                    right_k_mean = right_k_b[0]
            else:
                right_k_b = last_valid_right 
                if right_k_b is not None and right_k_mean is None:
                    right_k_mean = right_k_b[0]
            

            if left_k_b is not None:
                left_line_kf.predict()
                left_k_b = left_line_kf.update(left_k_b)  # 返回 [k, b]
                
                k_left, b_left = left_k_b[0], left_k_b[1]
                if abs(k_left) < 100:  # 避免垂直线导致除零，这里直接用公式画线
                    # 画线：x = k*y + b，在 y_min 到 y_max 之间
                    x_start_l = int(k_left * visible_y_min + b_left)
                    x_end_l = int(k_left * visible_y_max + b_left)
                    cv2.line(image, (x_start_l, visible_y_min), (x_end_l, visible_y_max), (0, 0, 255), 3)
                    
                    if enable_extra_drawing and right_k_b is not None:
                        k_r_tmp, b_r_tmp = right_k_b[0], right_k_b[1]
                        short_end_y, vp_y_min = get_shortest_line_end_y(left_k_b[0], left_k_b[1], k_r_tmp, b_r_tmp, vp_x, vp_y, width, height)
                        left_division_points = get_division_points_by_vp_and_short_line(left_k_b[0], left_k_b[1], vp_y, vp_y_min, short_end_y,
                                                                                    x_min=0, x_max=width, division_num=20)
                        if len(left_division_points) >= 20:
                            current_ipm_leftup = left_division_points[10]
                            current_ipm_leftdown = left_division_points[19]
                            cv2.circle(image, (current_ipm_leftup[0], current_ipm_leftup[1]), 5, (0,255,0), -1)
                            cv2.circle(image, (current_ipm_leftdown[0], current_ipm_leftdown[1]), 5, (0,255,255), -1)
                            ipm_left = True
            
            if right_k_b is not None:
                right_line_kf.predict()
                right_k_b = right_line_kf.update(right_k_b)
                
                k_right, b_right = right_k_b[0], right_k_b[1]
                if abs(k_right) < 100:
                    x_start_r = int(k_right * visible_y_min + b_right)
                    x_end_r = int(k_right * visible_y_max + b_right)
                    cv2.line(image, (x_start_r, visible_y_min), (x_end_r, visible_y_max), (0, 0, 255), 3)
                                    # 等分点处理
                    if enable_extra_drawing and  left_k_b is not None:
                        k_l_tmp, b_l_tmp = left_k_b[0], left_k_b[1]
                        short_end_y, vp_y_min = get_shortest_line_end_y(k_l_tmp, b_l_tmp, right_k_b[0], right_k_b[1], vp_x, vp_y, width, height)
                        right_division_points = get_division_points_by_vp_and_short_line(right_k_b[0], right_k_b[1], vp_y, vp_y_min, short_end_y,
                                                                                    x_min=0, x_max=width, division_num=20)
                        if len(right_division_points) >= 20:
                            current_ipm_rightup = right_division_points[10]
                            current_ipm_rightdown = right_division_points[19]
                            cv2.circle(image, (current_ipm_rightup[0], current_ipm_rightup[1]), 5, (0,255,0), -1)
                            cv2.circle(image, (current_ipm_rightdown[0], current_ipm_rightdown[1]), 5, (0,255,255), -1)
                            ipm_right = True

            if ipm_right and ipm_left:
                IPM_ON = True
                runwayDetector.left_valid_ipm = [current_ipm_leftdown, current_ipm_leftup]
                runwayDetector.right_valid_ipm = [current_ipm_rightdown, current_ipm_rightup]
                runwayDetector.IPM_points = [current_ipm_leftdown, current_ipm_rightdown, current_ipm_rightup, current_ipm_leftup]

            if YOLO_ON:
                cv2.rectangle(image, pt1, pt2, (255, 0, 255), 2)

            '''计算并绘制灭点（使用新交点公式）'''
            vp_x, vp_y = -1, -1
            if left_k_b is not None and right_k_b is not None:
                k_left, b_left = left_k_b[0], left_k_b[1]
                k_right, b_right = right_k_b[0], right_k_b[1]
                intersection = get_line_intersection(k_left, b_left, k_right, b_right)
                if intersection is not None:
                    vp_x, vp_y = intersection
                    cv2.circle(image, (vp_x, vp_y), 8, (255, 255, 0), -1)
            
                    '''绘制地平线'''
                    roi = (200, vp_y - 40, width-400, 70)
                    ROI_SKY_POINT = [[(width-400)/2, 20]]
                    horizon_result = HorizonDetector.detect_horizon(frame, roi, ROI_SKY_POINT)
                    
                    smoothed_horizon = []
                    if horizon_result is not None:
                        horizon_kf.predict()   
                        smoothed_horizon = horizon_kf.update(horizon_result)

                    # 地平线参数处理
                    if not smoothed_horizon:
                        k_h, b_h = last_valid_horizon if last_valid_horizon else (0, 0)
                    else:
                        k_h, b_h = smoothed_horizon
                        last_valid_horizon = (k_h, b_h)

                    # 绘制地平线
                    if k_h != 0:    
                        draw_full_image_line(image, 1/k_h, -(b_h)/k_h if k_h != 0 else 0)

            out.write(image)
            # cv2.imshow("Runway Detection", image) 
            
            key = cv2.waitKey(1)
            if key == ord('s'):
                cv2.imwrite("save_image.jpg", image)
            elif key == ord('q'):
                break
        
        elif IPM_ON:
            runwayDetector.process_frame(frame,out = out)
        
        counter += 1
        pbar.update(1)

    coord_file.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    pbar.close()
    print("处理完成，视频已保存")

if __name__ == "__main__":
    main()