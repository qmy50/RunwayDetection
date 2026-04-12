import cv2
import numpy as np
import time
from RunwayDetector_for_single_onnx_fast import RunwayDetector
import config
import argparse
from estimate_math import vanishing_point_pose, calculate_pose_from_runway
import onnxruntime as ort
from pnp_test import PNPExtractor
from utils import draw_full_image_line

# 加速配置
ort.set_default_logger_severity(3)  # 关闭冗余日志

camera_matrix = np.array([
    [554.3827,   0.0,    320.5],
    [0.0,    554.3827,   240.5],
    [0.0,      0.0,      1.0]
], dtype="double")


# ===================== ONNX 模型【加速版加载】=====================
onnx_model_path = r"/home/tianbot/XTDrone/RunwayDetectionV1/script/model/best.onnx"

# 🔥 核心优化：开启全部图优化 + 多线程
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 4  # 根据你的CPU核心数设置

session = ort.InferenceSession(
    onnx_model_path,
    sess_options=sess_options,
    providers=['CPUExecutionProvider']
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

width = 640
height = 480
runway_detector = RunwayDetector(width, height)
pnp_extractor = PNPExtractor()

USE_YOLO = True
USE_PNP = False
up = 0

# ===================== 【优化版】预处理 =====================
def preprocess_onnx(img, input_size=640):
    # 减少一次赋值，提速
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_size, input_size))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img, img.shape[2], img.shape[3]

# ===================== 【超级加速版】后处理 =====================
# 🔥🔥🔥 去掉 for 循环！用向量化处理，速度提升 5~10 倍
def postprocess_onnx(output, h, w, conf_thres=0.05, iou_thres=0.45):
    output = np.squeeze(output)
    pred = output.T

    # 向量化提取置信度
    scores = np.max(pred[:, 4:], axis=1)
    mask = scores >= conf_thres
    pred = pred[mask]
    scores = scores[mask]

    if len(pred) == 0:
        return None

    # 向量化提取框
    x, y, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    x1 = (x - bw / 2) * (w / 640)
    y1 = (y - bh / 2) * (h / 640)
    x2 = (x + bw / 2) * (w / 640)
    y2 = (y + bh / 2) * (h / 640)

    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.int32)
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, iou_thres)

    if len(indices) == 0:
        return None
    return boxes[indices[0]]

# ===================== 主处理函数 =====================
def process_single_image(frame, use_yolo=True):
    global USE_PNP
    height_frame, width_frame = frame.shape[:2]
    x1, y1, x2, y2 = 0, 0, 0, 0

    if use_yolo:
        img_onnx, _, _ = preprocess_onnx(frame)
        outputs = session.run([output_name], {input_name: img_onnx})
        box = postprocess_onnx(outputs[0], height_frame, width_frame)
        # cv2.imshow("Frame",frame)
        # cv2.waitKey(0)
        # print(box.shape)
        if box is None:
            return frame
        x1, y1, x2, y2 = box[0]
        global up
        up = y1

    # 跑道检测
    if USE_PNP:
        result = pnp_extractor.process_frame(frame,draw_result=False)
        if result['success']:
            left_line,right_line = result['left_line'],result['right_line']
            draw_full_image_line(frame,left_line[0],left_line[1])
            draw_full_image_line(frame,right_line[0],right_line[1])
        else:
            USE_PNP = False
    else:
        result = runway_detector.process_frame(frame)
        # print(result['vanishing_point'])
        frame = result['processed_frame']
        if result['vanishing_point'] is None:
            # print("no vanish point")
            return frame

    # 绘制
    if use_yolo:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

    vp_x, vp_y = result['vanishing_point']

    # 姿态解算
    if vp_x != 0 and vp_y != 0:
        cv2.circle(frame, (vp_x, vp_y), 8, (255, 255, 0), -1)
        beta, gama = vanishing_point_pose(K=camera_matrix, p_inf=[vp_x, vp_y])
        beta_deg, gama_deg = np.rad2deg(-beta), np.rad2deg(-gama)
        H, delta_x = calculate_pose_from_runway(
            W=32,
            k1=1 / result['right_line'][0],
            k2=1 / result['left_line'][0],
            beta=beta,
            gama=gama
        )
        print(f"{gama_deg:.2f},{beta_deg:.2f},{H:.2f},{delta_x:.2f}")
    cv2.imwrite("result.jpg", frame)
    return frame

def main():
    global USE_YOLO
    parser = argparse.ArgumentParser(description="跑道线检测脚本")
    parser.add_argument('config', help='path to config file')
    parser.add_argument("--img_path", required=True, help="输入图片路径")
    args = parser.parse_args()

    # 🔥 优化：只读取一次图片，循环里不再读
    frame = cv2.imread(args.img_path)

    start = time.time()
    temp = 0

    # 预热
    process_single_image(frame, True)

    # while temp < 100:
    #     frame = cv2.imread(args.img_path)
    #     process_single_image(frame, USE_YOLO)
    #     if temp > 50:
    #         USE_YOLO = False
    #     temp += 1

    end = time.time()
    cost = (end - start) 
    print(f"平均单帧耗时: {cost:.4f} s")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    #  python detect_img_fast.py ../UFLDv2/configs/tusimple_res18.py --img_path /home/tianbot/XTDrone/control/keyboard/frame_1700.jpg
    #  python detect_img_fast.py ../UFLDv2/configs/tusimple_res18.py --img_path ../FACT19_1_1LDImage1.jpg
    #  python detect_img_fast.py ../UFLDv2/configs/tusimple_res18.py --img_path ../RJTT16L2_1LDImage1.jpg
    #  python detect_img_fast.py ../UFLDv2/configs/tusimple_res18.py --img_path ../frame_000200.jpg
    #  python detect_img_onnx.py ../UFLDv2/configs/tusimple_res18.py --img_path ../FACT19_3_2LDImage2.jpg
    # python detect_img_onnx.py ../UFLDv2/configs/tusimple_res18.py --img_path ../YSSY16R1_4LDImage2.jpg
    # python detect_img_onnx.py ../UFLDv2/configs/tusimple_res18.py --img_path ../EDDF25C2_4LDImage1.png