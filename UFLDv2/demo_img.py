import torch, os, cv2
import torch
from UFLDv2.utils.common import merge_config, get_model
import tqdm
import torchvision.transforms as transforms
from PIL import Image  # 新增，用于帧预处理
from UFLDv2.utils_lane import draw_full_image_line, ransac_polyfit
import numpy as np

class UFLDDetector:
    def __init__(self, config_path=None):
        """
        初始化UFLD检测器
        :param config_path: 配置文件路径，如果为None则使用默认配置
        """
        torch.backends.cudnn.benchmark = True

        # 加载配置
        self.args, self.cfg = merge_config()
        self.cfg.batch_size = 1
        assert self.cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']
        self.cls_num_per_lane = 56
        self.img_w, self.img_h = 1280, 720

        # 加载模型
        self.net = get_model(self.cfg)
        state_dict = torch.load(self.cfg.test_model, map_location='cpu')['model']
        compatible_state_dict = {}
        for k, v in state_dict.items():
            compatible_state_dict[k[7:] if 'module.' in k else k] = v
        self.net.load_state_dict(compatible_state_dict, strict=False)
        self.net.eval().cuda()  # 直接放到GPU

        # 图像预处理变换
        self.img_transforms = transforms.Compose([
            transforms.Resize((int(self.cfg.train_height / self.cfg.crop_ratio), self.cfg.train_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def pred2coords(self, pred, local_width=1):
        """
        将预测结果转换为坐标
        :param pred: 模型预测结果
        :param local_width: 局部宽度
        :return: 左车道坐标, 右车道坐标
        """
        batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
        batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

        max_indices_row = pred['loc_row'].argmax(1).cpu()
        valid_row = pred['exist_row'].argmax(1).cpu()

        max_indices_col = pred['loc_col'].argmax(1).cpu()
        valid_col = pred['exist_col'].argmax(1).cpu()

        pred['loc_row'] = pred['loc_row'].cpu()
        pred['loc_col'] = pred['loc_col'].cpu()

        right_coords = []
        left_coords = []
        middle = self.img_w / 2
        coords = []

        row_lane_idx = [0, 1]
        col_lane_idx = [0, 1]

        for i in row_lane_idx:
            left_tmp = []
            right_tmp = []
            tmp = []
            if valid_row[0, :, i].sum() > num_cls_row / 3:
                for k in range(valid_row.shape[1]):
                    if valid_row[0, k, i]:
                        all_ind = torch.tensor(list(range(max(0, max_indices_row[0, k, i] - local_width),
                                                          min(num_grid_row - 1, max_indices_row[0, k, i] + local_width) + 1)))

                        out_tmp = (pred['loc_row'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                        out_tmp = out_tmp / (num_grid_row - 1) * self.img_w
                        tmp = (int(out_tmp), int(self.cfg.row_anchor[k] * self.img_h))
                        if tmp[0] < middle:
                            left_tmp.append(tmp)
                        else:
                            right_tmp.append(tmp)
                left_coords.extend(left_tmp)
                right_coords.extend(right_tmp)

        for i in col_lane_idx:
            tmp = []
            if valid_col[0, :, i].sum() > num_cls_col / 3:
                for k in range(valid_col.shape[1]):
                    if valid_col[0, k, i]:
                        all_ind = torch.tensor(list(range(max(0, max_indices_col[0, k, i] - local_width),
                                                          min(num_grid_col - 1, max_indices_col[0, k, i] + local_width) + 1)))

                        out_tmp = (pred['loc_col'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5

                        out_tmp = out_tmp / (num_grid_col - 1) * self.img_h
                        tmp = (int(self.cfg.col_anchor[k] * self.img_w), int(out_tmp))
                        if tmp[0] < middle:
                            left_tmp.append(tmp)
                        else:
                            right_tmp.append(tmp)
                left_coords.extend(left_tmp)
                right_coords.extend(right_tmp)

        return left_coords, right_coords

    def detect(self, img, draw=True):
        """
        检测单张图像的车道线
        :param image_path: 图像路径
        :param draw: 是否在图像上绘制车道线
        :return: 检测结果字典，包含坐标和拟合参数；如果draw=True，还返回绘制后的图像
        """
        # 图像预处理
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_tensor = self.img_transforms(img_pil).unsqueeze(0).cuda()

        with torch.no_grad():
            pred = self.net(img_tensor)

        left_coords, right_coords = self.pred2coords(pred)

        result = {
            'left_coords': np.array(left_coords),
            'right_coords': np.array(right_coords),
            'left_line': None,
            'right_line': None
        }

        if len(left_coords) > 0:
            result['left_line'] = ransac_polyfit(result['left_coords'][:, 0], result['left_coords'][:, 1],
                                                 degree=1, max_trials=100, residual_threshold=30)

        if len(right_coords) > 0:
            result['right_line'] = ransac_polyfit(result['right_coords'][:, 0], result['right_coords'][:, 1],
                                                  degree=1, max_trials=100, residual_threshold=30)

        if draw:
            img_copy = img.copy()
            if result['left_line'] is not None:
                draw_full_image_line(img_copy, result['left_line'][0], result['left_line'][1])
            if result['right_line'] is not None:
                draw_full_image_line(img_copy, result['right_line'][0], result['right_line'][1])
            result['image'] = img_copy
            

        return result

    def visualize(self, result, window_name="Lane Detection", window_size=(800, 450)):
        """
        可视化检测结果
        :param result: detect方法的返回值
        :param window_name: 窗口名称
        :param window_size: 窗口大小
        """
        if 'image' not in result:
            raise ValueError("结果中没有图像，请在detect时设置draw=True")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, window_size[0], window_size[1])
        cv2.imshow(window_name, result['image'])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 示例使用
    IMG_INPUT_PATH = r"D:\UFLD\Ultra-Fast-Lane-Detection-v2-master\UFLD_dataset\clips\train\FACT19_1_1LDImage1.jpg"
    IMG_OUTPUT_PATH = "test_img.jpg"

    detector = UFLDDetector()
    result = detector.detect(IMG_INPUT_PATH, draw=True)
    detector.visualize(result)

    # 保存结果图像
    if 'image' in result:
        cv2.imwrite(IMG_OUTPUT_PATH, result['image'])
        print(f"结果已保存到 {IMG_OUTPUT_PATH}")

    # 打印检测结果
    print("左车道坐标:", result['left_coords'])
    print("右车道坐标:", result['right_coords'])
    print("左车道拟合参数:", result['left_line'])
    print("右车道拟合参数:", result['right_line']) 

# python demo_img.py configs/tusimple_res18.py 