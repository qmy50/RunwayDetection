import cv2
import numpy as np
from ultralytics import YOLO
import os
from tqdm import tqdm

MODEL_PATH = r'D:\program\Entrance\models\best.pt'
# MODEL_PATH = r"D:\yolo\runs\segment\runway_train\runway_seg7\weights\best.pt"
model = YOLO(MODEL_PATH)
DRAW_PNP = False

def edge_detection_in_mask(img_cv, mask_cv, low_threshold=100, high_threshold=150):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)  # 高斯去噪，减少伪边缘
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges_in_mask = cv2.bitwise_and(edges, edges, mask=mask_cv)
    return edges_in_mask

def expand_mask_dilation(mask, kernel_size=5, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    expanded_mask = cv2.dilate(mask, kernel, iterations=iterations)
    return expanded_mask

def approx(con, side_size = 4):  
    num = 0.0005
    ep = num * cv2.arcLength(con, True)
    con = cv2.approxPolyDP(con, ep, True)
    while True:
        if len(con) <= side_size:  # 顶点数达标，退出循环
            break
        else:  # 顶点数超标，增大系数，重新逼近
            num = num * 1.5  # 系数每次扩大1.5倍（快速精简顶点）
            ep = num * cv2.arcLength(con, True)
            con = cv2.approxPolyDP(con, ep, True)
    return con  # 返回精简后的轮廓
    
def detect_polygon_in_mask(ori_img,edges_in_mask, target_sides=4, epsilon_ratio=0.02):
    global DRAW_PNP
    w = ori_img.shape[1]
    contours, _ = cv2.findContours(edges_in_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_points = []
    points = np.zeros((4,2),dtype = int)
    for cnt in contours:
        contour_length = cv2.arcLength(cnt, closed=True)
        if contour_length >= 30:
            all_points.extend(cnt.reshape(-1,2))
    all_points = np.array(all_points)
    hull = cv2.convexHull(all_points)
    con = approx(hull,side_size=4)

    print("最外侧四个顶点坐标：")
    for i, point in enumerate(con):
        print(f"顶点 {i+1}: ({point[0,0]}, {point[0,1]})")
        points[i,0],points[i,1] = point[0,0],point[0,1]
        if abs(point[0,0] > 0 and point[0,1]) > 0.45 * w:
            DRAW_PNP = True
    # color_img = cv2.cvtColor(edges_in_mask, cv2.COLOR_GRAY2BGR)
    # 绘制四个顶点（红色圆点）
    if DRAW_PNP:
        cv2.drawContours(ori_img, [con], 0, (255, 0, 0), 2)
        for i in range(target_sides):
            cv2.circle(ori_img, (points[i,0],points[i,1]), 5, (0, 0, 255), -1)
    return ori_img,points

def predict_single_image_with_expanded_mask(
    img, 
    output_path, 
    conf_threshold=0.25,
    dilate_kernel=5,
    dilate_iter=1,
    mask_color=(0, 255, 0),
    alpha=0.5
):
    orig_h, orig_w = img.shape[:2]
    results = model.predict(img, conf=conf_threshold, verbose=False)
    
    combined_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    expanded_masks_list = []
    
    for r in results:
        if r.masks is not None:
            # 使用多边形点生成掩码
            for poly in r.masks.xy:
                # 创建空白掩码
                mask_img = np.zeros((orig_h, orig_w), dtype=np.uint8)
                # 填充多边形
                poly = poly.astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask_img, [poly], 255)
                
                # 膨胀
                expanded_mask = expand_mask_dilation(mask_img, dilate_kernel, dilate_iter)
                expanded_masks_list.append(expanded_mask)
                combined_mask = cv2.bitwise_or(combined_mask, expanded_mask)
    
    if len(expanded_masks_list) > 0:
        colored_mask = np.zeros_like(img)
        colored_mask[combined_mask > 0] = mask_color
    
    # cv2.imwrite(output_path, final_img)
    print(f"膨胀后的掩码图像已保存至: {output_path}")
    print(f"共处理 {len(expanded_masks_list)} 个掩码")
    
    return expanded_masks_list

if __name__ == "__main__":
    # 配置参数

    VIDEO_PATH = r'D:\yolo\qq_3045834499\yolov8-42\my_dataset\test\test_video.mp4'
    OUTPUT_PATH = r'D:\program\Entrance\runway_predict\expanded_mask_video_1.mp4'
    
    # 膨胀参数（可调整：核越大/迭代越多，掩码扩得越大）
    DILATE_KERNEL = 1    # 膨胀核大小（奇数，如3/5/7/9）
    DILATE_ITER = 2      # 迭代次数（越大膨胀越明显）
    
    # 创建输出目录
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {VIDEO_PATH}")
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 视频帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 视频高度
    total_frames = int(0.4 * cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码格式
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    print(f"开始处理视频，总帧数: {total_frames}")
    pbar = tqdm(total=total_frames, desc="处理视频帧", unit="frame", colour="green")
    counter = 0
    conf_threshold = 0.25
    while cap.isOpened() and counter < total_frames:
        ret, frame = cap.read()
        if not ret:
            break  # 读取完所有帧，退出循环
        try:
            expanded_masks = predict_single_image_with_expanded_mask(
                img = frame.copy(),
                output_path=OUTPUT_PATH,

                conf_threshold=0.25,
                dilate_kernel=DILATE_KERNEL,
                dilate_iter=DILATE_ITER
            )
            
            edges = edge_detection_in_mask(frame, expanded_masks[0] if expanded_masks else np.zeros_like(frame[:,:,0]), low_threshold=100, high_threshold=150)
            # cv2.imshow("Edges in Expanded Mask", edges)
            ori_img, points = detect_polygon_in_mask(frame, edges, target_sides=4, epsilon_ratio=0.02)
            # cv2.imshow("Polygon Detection", color_img)
            out.write(ori_img)
            
        except Exception as e:
            print(f"处理出错: {e}")
        # out.write(frame)
        pbar.update(1)
        counter += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()