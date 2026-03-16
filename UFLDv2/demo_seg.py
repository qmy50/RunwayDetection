import torch, os, cv2
import sys
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utils.dist_utils import dist_print
from utils.common import merge_config
from model.model_tusimple import get_model
from data.torch_dataset import LaneTestDataset
from data.constant import tusimple_row_anchor, tusimple_col_anchor


def get_anchor(cfg):
    row_anchor = np.array(tusimple_row_anchor) / cfg.train_height
    col_anchor = np.array(tusimple_col_anchor) / cfg.train_width
    return row_anchor, col_anchor


def visualize_seg(seg_out, img_shape, num_lanes, colors=None):
    """
    将分割 logits 转换为彩色图像
    :param seg_out: (1, C, H, W) 分割 logits
    :param img_shape: (height, width) 目标图像尺寸
    :param num_lanes: 车道线数量（不包括背景）
    :param colors: 颜色列表，长度应为 num_lanes+1，若为 None 则自动生成
    :return: 彩色分割图 (H, W, 3) uint8
    """
    if colors is None:
        # 自动生成颜色（BGR格式），背景黑色，车道线使用 tab10 色图
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('tab10', num_lanes)
        colors = [(0, 0, 0)]  # 背景
        for i in range(num_lanes):
            rgba = cmap(i)
            bgr = (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))
            colors.append(bgr)

    # 上采样到原图尺寸
    seg_up = F.interpolate(seg_out, size=img_shape, mode='bilinear', align_corners=False)  # (1, C, H, W)
    seg_pred = seg_up.argmax(dim=1).squeeze(0)  # (H, W)
    seg_np = seg_pred.cpu().numpy().astype(np.uint8)

    # 转换为彩色图
    color_map = np.zeros((*img_shape, 3), dtype=np.uint8)
    for c in range(len(colors)):
        color_map[seg_np == c] = colors[c]
    return color_map


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()
    cfg.batch_size = 1
    print('setting batch_size to 1 for demo generation')

    dist_print('start testing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    # 根据数据集获取相关参数（这里仅用 cfg 中的信息，不重新定义）
    net = get_model(cfg)

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((int(cfg.train_height / cfg.crop_ratio), cfg.train_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if cfg.dataset == 'CULane':
        splits = ['test0_normal.txt', 'test1_crowd.txt', 'test2_hlight.txt',
                  'test3_shadow.txt', 'test4_noline.txt', 'test5_arrow.txt',
                  'test6_curve.txt', 'test7_cross.txt', 'test8_night.txt']
        datasets = [LaneTestDataset(cfg.data_root,
                                     os.path.join(cfg.data_root, 'list/test_split/'+split),
                                     img_transform=img_transforms,
                                     crop_size=cfg.train_height) for split in splits]
        img_w, img_h = 1640, 590
    elif cfg.dataset == 'Tusimple':
        splits = ['test.txt']
        datasets = [LaneTestDataset(cfg.data_root,
                                     os.path.join(cfg.data_root, split),
                                     img_transform=img_transforms,
                                     crop_size=cfg.train_height) for split in splits]
        img_w, img_h = 1920, 1080
    else:
        raise NotImplementedError

    for split, dataset in zip(splits, datasets):
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_name = split.replace('.txt', '.avi')
        vout = cv2.VideoWriter(video_name, fourcc, 30.0, (img_w, img_h))

        row_anchor, col_anchor = get_anchor(cfg)  # 此处不再使用，但保留以保持函数调用

        for i, data in enumerate(tqdm.tqdm(loader)):
            imgs, names = data
            imgs = imgs.cuda()
            with torch.no_grad():
                pred = net(imgs)

            # 读取原始图像
            vis = cv2.imread(os.path.join(cfg.data_root, names[0]))
            if vis is None:
                print(f"Warning: cannot read image {names[0]}, skipping...")
                continue

            # 如果模型输出了分割结果，则叠加显示
            if 'seg_out' in pred:
                seg_out = pred['seg_out']  # (1, C, H_seg, W_seg)
                # 自动生成颜色（车道线数 = cfg.num_lanes）
                num_lanes = cfg.num_lanes
                # 生成彩色分割图（与原图相同尺寸）
                seg_color = visualize_seg(seg_out, (img_h, img_w), num_lanes)
                # 与原图叠加（半透明）
                alpha = 0.5
                vis = cv2.addWeighted(vis, 1 - alpha, seg_color, alpha, 0)
            else:
                print("Warning: seg_out not found in model output, writing original image.")

            vout.write(vis)

        vout.release()
        print(f"Video saved to {video_name}")