#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import onnxruntime as ort
from std_msgs.msg import Float32, Bool
from RunwayDetector_for_single_onnx_fast import RunwayDetector
from estimate_math import vanishing_point_pose, calculate_pose_from_runway
from pnp_test import PNPExtractor
from utils import draw_full_image_line

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

    h, w = frame.shape[:2]
    box = None

    if USE_YOLO:
        rospy.loginfo("✅ USE YOLO")
        img_onnx = preprocess_onnx(frame)
        outputs = session.run([output_name], {input_name: img_onnx})
        box = postprocess_onnx(outputs[0], h, w)

    result = {}
    if USE_PNP :
        result = pnp_extractor.process_frame(frame, draw_result=False)
        rospy.loginfo("✅ USE PNP")
        y_offset = result['y_offset']
        if y_offset is not None:
            H = result['height']
            delta_y = y_offset
            rospy.loginfo(f"高度: {H:.1f}m | 偏移: {delta_y:.2f}")
            pub_pnp.publish(np.clip(delta_y,-15,15))

            if (H is not None and 0 <= H <= 20) or result['end_flag']:
                USE_UFLD = True
                USE_PNP = False
            
            # if delta_y is not None:
            #     pub_pnp.publish(delta_y)
        # if result['left_line'] is not None and result['right_line'] is not None:
        # #     rospy.loginfo("✅ PNP success")
        # #     HAS_PNP = True
        #     left,right = result['left_line'],result['right_line']
        #     draw_full_image_line(frame, left[0], left[1])
        #     draw_full_image_line(frame, right[0], right[1])
        # elif HAS_PNP:
        #     USE_PNP = False

    elif USE_UFLD:
        rospy.loginfo("✅ USE UFLDv2")
        result = runway_detector.process_frame(frame)
        if result['vanishing_point'] is not None:
            vp_x, vp_y = result['vanishing_point']
            cv2.circle(frame, (vp_x, vp_y), 6, (0, 255, 255), -1)

            beta, gama = vanishing_point_pose(K=camera_matrix, p_inf=[vp_x, vp_y])
            beta_deg = np.rad2deg(-beta)
            gama_deg = np.rad2deg(-gama)

            H, delta_y = calculate_pose_from_runway(
                W=32,
                k1=1/result['right_line'][0],
                k2=1/result['left_line'][0],
                beta = -beta, gama = -gama
            )
            
            if 0 <= H <= 10:
                rospy.logwarn("✅ landing")
                pub_land.publish(True)
                LAND = True
                
            # pub_xoffset.publish(Float32(delta_x))
            # pub_vertiacl.publish(Float32(H))
            rospy.loginfo(f"高度: {H:.1f}m | 偏移: {delta_y:.2f}")

    if box is not None:
        x1, y1, x2, y2 = box[0]
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        # if y2 >= 475:
        #     USE_UFLD = True
        #     USE_PNP = False
        # cv2.putText(frame, "RUNWAY", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0),2)
        runway_center_x = (x1 + x2)/2
        offset_x = (runway_center_x - w / 2) / (w / 2)
        pub_xoffset.publish(offset_x)
        runway_center_y = (y1 + y2)/2
        offset_y = (runway_center_y - h / 3) / (h / 3)
        pub_vertiacl.publish(offset_y)
    if USE_PNP:
        cv2.imshow("Runway Detect", frame)
        cv2.waitKey(1)

if __name__ == "__main__":
    rospy.init_node("runway_detector_node")
    rospy.Subscriber(IMAGE_TOPIC, Image, image_callback, queue_size=10)

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