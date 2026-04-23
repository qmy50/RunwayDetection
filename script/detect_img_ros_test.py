#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
import onnxruntime as ort
from std_msgs.msg import Float32, Bool
from RunwayDetector_for_single_onnx_fast import RunwayDetector
from estimate_math import vanishing_point_pose, calculate_pose_from_runway
from pnp_test import PNPExtractor
from utils import draw_full_image_line
from KalmanFilter import KalmanFilter1D
import os
from datetime import datetime

# 日志文件用于误差分析
LOG_DIR = os.path.join("/home/tianbot/XTDrone/RunwayDetectionV1","runway_analysis_logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(LOG_DIR, f"height_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
log_file = open(log_filename, 'w')
log_file.write("timestamp,source,height_m,height_smooth,offset_y,gt_height,gt_offset_y,land_flag,remarks\n")
log_file.flush()  

height_kf = KalmanFilter1D(Q=0.05, R=0.5)  
onnx_model_path = r"/home/tianbot/XTDrone/RunwayDetectionV1/script/model/best.onnx"
IMAGE_TOPIC = "/plane_0/image_raw"
camera_matrix = np.array([
    [554.3827,   0.0,    320.5],
    [0.0,    554.3827,   240.5],
    [0.0,      0.0,      1.0]
], dtype="double")

ort.set_default_logger_severity(3)
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 8

session = ort.InferenceSession(
    onnx_model_path,
    sess_options=sess_options,
    providers=['CPUExecutionProvider']
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

runway_detector = RunwayDetector(640, 480)
pnp_extractor = PNPExtractor()
bridge = CvBridge()

USE_YOLO = True
USE_PNP = True
USE_UFLD = False
LAND = False
groundtruth_height = 0.0
groundtruth_offset_y = 0.0

def preprocess_onnx(img, input_size=640):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_size, input_size))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img

def postprocess_onnx(output, h, w, conf_thres=0.05, iou_thres=0.45):
    output = np.squeeze(output)
    pred = output.T

    scores = np.max(pred[:, 4:], axis=1)
    mask = scores >= conf_thres
    pred = pred[mask]
    scores = scores[mask]

    if len(pred) == 0:
        return None

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

def image_callback(msg):
    global USE_YOLO, USE_PNP,USE_UFLD,LAND
    # if LAND:
    #     return
    try:
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        rospy.logerr(e)
        return
    
    def write_log(source, height, height_smooth, offset_y,gt_height,gt_offset_y, land_flag, remarks=""):
        timestamp = rospy.get_time()
        # 如果某个值为 None，写入 "nan"
        height_str = f"{height:.3f}" if height is not None else "nan"
        height_smooth_str = f"{height_smooth:.3f}" if height_smooth is not None else "nan"
        offset_y_str = f"{offset_y:.3f}" if offset_y is not None else "nan"
        gt_height_str = f"{gt_height:.3f}" if gt_height is not None else "nan"
        gt_offset_y_str = f"{gt_offset_y:.3f}" if gt_offset_y is not None else "nan"
        land_flag_str = "True" if land_flag else "False"
        log_file.write(f"{timestamp},{source},{height_str},{height_smooth_str},{offset_y_str},{gt_height_str},{gt_offset_y_str},{land_flag_str},{remarks}\n")
        log_file.flush()

    h, w = frame.shape[:2]
    box = None

    if USE_YOLO:
        rospy.loginfo_throttle(1.0,"✅ USE YOLO")
        img_onnx = preprocess_onnx(frame)
        outputs = session.run([output_name], {input_name: img_onnx})
        box = postprocess_onnx(outputs[0], h, w)

    result = {}
    if USE_PNP :
        result = pnp_extractor.process_frame(frame, draw_result=False)
        y_offset = result['y_offset']
        if y_offset is not None:
            rospy.loginfo_throttle(1.0,"✅ USE PNP")
            H = result['height']
            delta_y = y_offset
            rospy.loginfo_throttle(1.0,f"高度: {H:.1f}m | 偏移: {delta_y:.2f}")
            pub_pnp.publish(np.clip(delta_y,-15,15))
            if (H is not None and 0 <= H <= 15) or result['end_flag']:
                USE_UFLD = True
                USE_PNP = False
            
            # if delta_y is not None:
            #     pub_pnp.publish(delta_y)
        # if result['left_line'] is not None and result['right_line'] is not None:
        # #     rospy.loginfo(" PNP success")
        # #     HAS_PNP = True
        #     left,right = result['left_line'],result['right_line']
        #     draw_full_image_line(frame, left[0], left[1])
        #     draw_full_image_line(frame, right[0], right[1])
        # elif HAS_PNP:
        #     USE_PNP = False

    elif USE_UFLD:
        rospy.loginfo_throttle(1.0,"✅ USE UFLDv2")
        result = runway_detector.process_frame(frame)
        if result['vanishing_point'] is not None:
            vp_x, vp_y = result['vanishing_point']
            cv2.circle(frame, (vp_x, vp_y), 6, (0, 255, 255), -1)

            beta, gama = vanishing_point_pose(K=camera_matrix, p_inf=[vp_x, vp_y])
            beta_deg = np.rad2deg(-beta)
            gama_deg = np.rad2deg(-gama)

            H, delta_y = calculate_pose_from_runway(
                W=32.05,
                k1=1/result['right_line'][0],
                k2=1/result['left_line'][0],
                beta = beta, gama = gama
            )
            H_smooth = height_kf.update(H)
            write_log("UFLD", H, H_smooth, delta_y, groundtruth_height,groundtruth_offset_y,LAND, "UFLD mode")
            
            if 0 <= H <= 9:
                rospy.logwarn("✅ landing")
                pub_land.publish(True)
                LAND = True
                
            # pub_xoffset.publish(Float32(delta_x))
            # pub_vertiacl.publish(Float32(H))
            rospy.loginfo(f"高度: {H:.1f}m | 偏移: {delta_y:.2f}")

    if box is not None:
        x1, y1, x2, y2 = box[0]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        if y2 >= 475:
            USE_UFLD = True
            USE_PNP = False
        cv2.putText(frame, "RUNWAY", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0),2)
        runway_center_x = (x1 + x2)/2
        offset_x = np.clip((runway_center_x - w / 2) / (w / 2),-10,10)
        pub_xoffset.publish(offset_x)
        runway_center_y = (y1 + y2)/2
        offset_y = (runway_center_y - h / 3.2) / (h / 3.2)
        pub_vertiacl.publish(offset_y)
    if USE_PNP:
        cv2.imshow("Runway Detect", frame)
        cv2.waitKey(1)

def shutdown_hook():
    global log_file
    log_file.close()
    rospy.loginfo("日志文件已关闭，保存路径: %s", log_filename)

def pose_callback(msg):
    global groundtruth_height,groundtruth_offset_y
    groundtruth_height = msg.pose.position.z
    groundtruth_offset_y = msg.pose.position.y
    # rospy.loginfo_throttle(1.0,f"当前高度为:{groundtruth_height}")

if __name__ == "__main__":
    rospy.init_node("runway_detector_node")
    rospy.on_shutdown(shutdown_hook)
    rospy.Subscriber(IMAGE_TOPIC, Image, image_callback, queue_size=10)

    rospy.Subscriber('plane_0/mavros/vision_pose/pose',PoseStamped,pose_callback,queue_size=10)

    pub_xoffset = rospy.Publisher('runway_offset',Float32,queue_size=1)
    pub_vertiacl = rospy.Publisher('runway_offset_vertical',Float32,queue_size=10)
    pub_pnp = rospy.Publisher('runway_offset_pnp',Float32,queue_size=10)
    pub_land = rospy.Publisher('runway_land',Bool,queue_size=10)

    rospy.loginfo("✅ 跑道视觉定位节点启动完成")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()

# python detect_img_ros_test.py ../UFLDv2/configs/tusimple_res18.py