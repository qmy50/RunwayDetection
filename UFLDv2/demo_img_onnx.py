import cv2
import numpy as np
import onnxruntime as ort
from UFLDv2.utils.common import merge_config
from UFLDv2.utils_lane import draw_full_image_line, ransac_polyfit

class UFLDDetector:
    def __init__(self, onnx_path):
        # 初始化 ONNX，自动获取 4 个输出
        # sess_options = ort.SessionOptions()
        # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # sess_options.intra_op_num_threads = 4  # 根据你的CPU核心数设置
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]  # ✅ 关键

        self.args, self.cfg = merge_config()
        self.cfg.batch_size = 1
        assert self.cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']
        self.cls_num_per_lane = 56

        # Tusimple 固定配置
        self.train_height = 320
        self.train_width = 800
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # self.img_w = 1920
        # self.img_h = 1080
        self.img_w = 640
        self.img_h = 480

        # 锚点
        self.row_anchor = np.linspace(160, 710, 56) / self.img_h
        self.col_anchor = np.linspace(0, 1, 41)

    def softmax(self, x, axis=0):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=axis)
    
    def pred2coords(self, pred, local_width=1, original_image_width=640, original_image_height=480):
        loc_row, loc_col, exist_row, exist_col,_ = pred  # ✅ 直接解开 4 个输出
        batch_size, num_grid_row, num_cls_row, num_lane_row = loc_row.shape
        batch_size, num_grid_col, num_cls_col, num_lane_col = loc_col.shape
        max_indices_row = loc_row.argmax(1)
        valid_row =exist_row.argmax(1)

        max_indices_col = loc_col.argmax(1)
        valid_col = exist_col.argmax(1)

        right_coords = []
        left_coords = []

        row_lane_idx = [0, 1]
        col_lane_idx = [0, 1]

        for i in row_lane_idx:
            left_tmp = []
            right_tmp = []
            tmp = []
            if valid_row[0, :, i].sum() > 5:
                for k in range(valid_row.shape[1]):
                    if valid_row[0, k, i]:
                        all_ind = np.array(range(max(0, max_indices_row[0, k, i] - local_width),
                                                          min(num_grid_row - 1, max_indices_row[0, k, i] + local_width) + 1))

                        out_tmp = (self.softmax(loc_row[0, all_ind, k, i],axis = 0) * all_ind.astype(np.float32)).sum() + 0.5
                        out_tmp = out_tmp / (num_grid_row - 1) * self.img_w
                        tmp = (int(out_tmp), int(self.cfg.row_anchor[k] * self.img_h))
                        if i == 0:
                            left_tmp.append(tmp)
                        else:
                            right_tmp.append(tmp)
                left_coords.extend(left_tmp)
                right_coords.extend(right_tmp)

        for i in col_lane_idx:
            tmp = []
            if valid_col[0, :, i].sum() > 5:
                for k in range(valid_col.shape[1]):
                    if valid_col[0, k, i]:
                        all_ind = np.array(range(max(0, max_indices_col[0, k, i] - local_width),
                                                          min(num_grid_col - 1, max_indices_col[0, k, i] + local_width) + 1))

                        out_tmp = (self.softmax(loc_col[0, all_ind, k, i],axis =0) * all_ind.astype(np.float32)).sum() + 0.5

                        out_tmp = out_tmp / (num_grid_col - 1) * self.img_h
                        tmp = (int(self.cfg.col_anchor[k] * self.img_w), int(out_tmp))
                        if i == 0:
                            left_tmp.append(tmp)
                        else:
                            right_tmp.append(tmp)
                left_coords.extend(left_tmp)
                right_coords.extend(right_tmp)

        return left_coords, right_coords

    def detect(self, img,draw = True):
        img_h, img_w = img.shape[:2]
        img_display = img.copy() 

        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Resize
        img_resized = cv2.resize(frame_rgb, (self.train_width, self.train_height), interpolation=cv2.INTER_LINEAR)

        # 3. 归一化 / 255
        img_float = img_resized.astype(np.float32) / 255.0

        # 4. Normalize (减均值，除方差)
        img_norm = (img_float - self.mean) / self.std

        # 5. HWC -> CHW
        img_chw = img_norm.transpose(2, 0, 1)

        # 6. 增加 batch 维度
        img_input = img_chw[np.newaxis, :, :, :]

        # model_outputs = self.session.get_outputs()
        # self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        # print('----------------------------')
        # print(self.output_names)

        # ✅ 关键：一次性获取 4 个输出
        pred = self.session.run(None, {self.input_name: img_input.astype(np.float32)})

        left_coords, right_coords = self.pred2coords(pred)
        # for cords in left_coords:
        #     cv2.circle(img_display,(cords[0],cords[1]),radius=5,color=(0,255,0),thickness=-1)
        # for cords in right_coords:
        #     cv2.circle(img_display,(cords[0],cords[1]),radius=5,color=(255,255,0),thickness=-1)
        # cv2.imshow('result',img_display)
        # cv2.waitKey(0)
        result = {
            'left_coords': np.array(left_coords),
            'right_coords': np.array(right_coords),
            'left_line': None,
            'right_line': None
        }

        if len(left_coords) > 0:
            # print("have left")
            result['left_line'] = ransac_polyfit(result['left_coords'][:, 0], result['left_coords'][:, 1],
                                                 degree=1, max_trials=100, residual_threshold=30)
            # print(result['left_coords'])

        if len(right_coords) > 0:
            # print("have right")
            result['right_line'] = ransac_polyfit(result['right_coords'][:, 0], result['right_coords'][:, 1],
                                                  degree=1, max_trials=100, residual_threshold=30)

        if draw:
            img_copy = img.copy()
            for coords in result['left_coords']:
                cv2.circle(img_copy,(coords[0],coords[1]),radius=5,color=(0,0,255))
            # if result['left_line'] is not None:
            #     draw_full_image_line(img_copy, result['left_line'][0], result['left_line'][1])
            #     cv2.imshow('left_line',img_copy)
            #     cv2.waitKey(0)
            # if result['right_line'] is not None:
            #     draw_full_image_line(img_copy, result['right_line'][0], result['right_line'][1])
            #     cv2.imshow('right_line',img_copy)
            #     cv2.waitKey(0)
            result['image'] = img_copy
        cv2.destroyAllWindows()
            

        return result

if __name__ == "__main__":
    # 配置路径
    CONFIG_PATH = "configs/tusimple_res18.py"
    IMG_INPUT_PATH = r"/home/tianbot/XTDrone/RunwayDetectionV1/FACT19_1_1LDImage1.jpg"
    IMG_OUTPUT_PATH = "result.jpg"

    # 读取图片
    img = cv2.imread(IMG_INPUT_PATH)
    if img is None:
        raise FileNotFoundError("图片不存在")

    # 初始化检测器
    detector = UFLDDetector(CONFIG_PATH, img_w=img.shape[1], img_h=img.shape[0])
    
    # 检测
    result = detector.detect(img)
    detector.visualize(result)

    # 保存
    if 'image' in result:
        cv2.imwrite(IMG_OUTPUT_PATH, result['image'])
        print(f"结果已保存: {IMG_OUTPUT_PATH}")

    # 打印
    print("左车道线:", result['left_line'])
    print("右车道线:", result['right_line'])

# python demo_img_onnx.py configs/tusimple_res18.py 