import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import numpy as np
import random
import json
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# 忽略 PIL 的 DecompressionBombWarning
Image.MAX_IMAGE_PIXELS = None

def loader_func(path):
    return Image.open(path)

class LaneTestDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform=None, crop_size=None):
        super(LaneTestDataset, self).__init__()
        self.path = path
        self.img_transform = img_transform
        self.crop_size = crop_size
        with open(list_path, 'r') as f:
            self.list = f.readlines()
        self.list = [l[1:] if l[0] == '/' else l for l in self.list]  # exclude the incorrect path prefix '/' of CULane

    def __getitem__(self, index):
        name = self.list[index].split()[0]
        img_path = os.path.join(self.path, name)
        img = loader_func(img_path)

        if self.img_transform is not None:
            img = self.img_transform(img)
        img = img[:,-self.crop_size:,:]

        return img, name

    def __len__(self):
        return len(self.list)

def extend_coords(coords):
    """对行方向插值后的 x 坐标进行底部延伸（线性拟合后半段）"""
    num_lanes, num_cls = coords.shape
    # 兼容CPU/GPU张量
    if coords.is_cuda:
        coords_np = coords.cpu().numpy()
    else:
        coords_np = coords.numpy()
    coords_axis = np.arange(num_cls)
    fitted = coords.clone()

    for i in range(num_lanes):
        lane = coords_np[i]
        if lane[-1] > 0:
            continue

        valid = lane > 0
        num_valid = np.sum(valid)
        if num_valid < 6:
            continue

        half = num_valid // 2
        p = np.polyfit(coords_axis[valid][half:], lane[valid][half:], deg=1)
        start = coords_axis[valid][half]
        fitted_lane = np.polyval(p, np.arange(start, num_cls))
        fitted[i, start:] = torch.tensor(fitted_lane, dtype=fitted.dtype)
    # print(f"车道线 {i}: 有效点数 {num_valid}, 最后一个点 {lane[-1]}, 是否延伸 {num_valid>=6 and lane[-1]<=0}")
    return fitted.to(coords.device)

def apply_affine_to_points(points, M, img_size):
    """
    对车道线坐标应用仿射变换（适配crop/resize后的坐标映射）
    Args:
        points: [num_lanes, num_cls] 单维度坐标值（x或y）
        M: 仿射变换矩阵 (2,3)
        img_size: 原始图像尺寸 (w, h)
    Returns:
        transformed_points: 变换后的坐标
    """
    num_lanes, num_cls = points.shape
    transformed = points.clone()
    
    # 只处理有效点
    valid_mask = points > 0
    if not torch.any(valid_mask):
        return transformed
    
    # 构造完整的(x,y)坐标（行锚点对应固定y，列锚点对应固定x）
    if len(img_size) == 2:
        w, h = img_size
        # 对于行锚点（y固定，x变化），需要先恢复原始y坐标
        row_anchor_pix_original = (np.array(dataset.row_anchor) * dataset.orig_h).astype(int)
        for i in range(num_lanes):
            for j in range(num_cls):
                if valid_mask[i, j]:
                    x = points[i, j].item()
                    y = row_anchor_pix_original[j]
                    # 应用仿射变换
                    xy = np.array([[x, y]], dtype=np.float32)
                    xy_transformed = cv2.transform(xy.reshape(-1,1,2), M).reshape(-1,2)
                    transformed[i, j] = torch.tensor(xy_transformed[0, 0])
    return transformed

class TrainDataset(Dataset):
    def __init__(self, data_root, list_path,
                 row_anchor, col_anchor,
                 train_width, train_height,
                 num_cell_row, num_cell_col,
                 dataset_name, top_crop, num_lanes,use_aux = False,segment_transform = None):
        super().__init__()
        self.data_root = data_root
        self.train_width = train_width
        self.train_height = train_height
        self.top_crop = top_crop
        self.num_cell_row = num_cell_row
        self.num_cell_col = num_cell_col
        self.num_lanes = num_lanes

        self.use_aux = use_aux
        self.seg_transform = segment_transform

        # 保存锚点（归一化值）
        self.row_anchor = row_anchor          # 例如 [0.2, 0.3, ..., 1.0]
        self.col_anchor = col_anchor

        # 原始图像尺寸
        if dataset_name == 'CULane':
            self.orig_w, self.orig_h = 1640, 590
        elif dataset_name == 'Tusimple':
            self.orig_w, self.orig_h = 1920, 1080  
        elif dataset_name == 'CurveLanes':
            self.orig_w, self.orig_h = 2560, 1440
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # 读取 list.txt，过滤空行
        with open(list_path, 'r', encoding='utf-8') as f:
            self.list = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        l = self.list[idx]
        l_info = l.split()
        img_name, mask_name = l_info[0], l_info[1]
        
        # 处理路径前缀
        if img_name.startswith('/'):
            img_name = img_name[1:]
        if mask_name.startswith('/'):
            mask_name = mask_name[1:]

        # 构建完整路径
        img_path = os.path.join(self.data_root, img_name)
        seg_path = os.path.join(self.data_root, mask_name)
        img_path = img_path.replace('\\', '/')
        seg_path = seg_path.replace('\\', '/')

        # 读取图像和分割图
        try:
            img = Image.open(img_path).convert('RGB')
            seg = Image.open(seg_path).convert('RGB')   # 兼容彩色分割图
        except Exception as e:
            print(f"读取图像失败 {img_path}: {e}")
            # 返回空样本（实际训练中建议跳过）
            return self.__getitem__((idx + 1) % len(self))
        
        w, h = img.size

        # ========== 随机仿射变换 ==========
        scale_x = random.uniform(0.8, 1.2)
        scale_y = random.uniform(0.8, 1.2)
        angle = random.uniform(-6, 6)
        tx = random.uniform(-200, 200)
        ty = random.uniform(-100, 100)

        # 构建仿射变换矩阵（简化版，兼容OpenCV）
        center = (w / 2, h / 2)
        # 先旋转缩放
        M_rot = cv2.getRotationMatrix2D(center, angle, scale_x)
        # 再平移
        M_rot[0, 2] += tx
        M_rot[1, 2] += ty

        # 对图像应用仿射
        img_np = np.array(img)
        img_np = cv2.warpAffine(img_np, M_rot, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        img_t = Image.fromarray(img_np)

        # 对分割图应用相同变换（最近邻）
        seg_np = np.array(seg)
        seg_np = cv2.warpAffine(seg_np, M_rot, (w, h), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        seg_t = Image.fromarray(seg_np)

        # ========== resize 和 crop ==========
        mid_h = int(self.train_height / self.top_crop)
        img = img_t.resize((self.train_width, mid_h), Image.BILINEAR)
        seg = seg_t.resize((self.train_width, mid_h), Image.NEAREST)

        crop_y = mid_h - self.train_height
        img = img.crop((0, crop_y, self.train_width, mid_h))
        seg = seg.crop((0, crop_y, self.train_width, mid_h))

        # ========== 转换为张量 ==========
        # 图像归一化（用于模型输入）
        img = TF.to_tensor(img)  # [3, H, W], [0,1]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img = (img - mean) / std

        # 分割图转为灰度标签
        seg_np = np.array(seg)               # [H, W, 3] uint8
        seg_gray = seg_np[:, :, 0].copy()       # 取一个通道
        seg_tensor = torch.from_numpy(seg_gray).long()  # [H, W]
        # seg_tensor = torch.from_numpy(seg_np.transpose(2,0,1)).long()  # [3, H, W]
        # ========== 从分割图提取车道线坐标 ==========
        # 计算行锚点在crop/resize后图像上的 y 坐标（像素）
        row_anchor_pix = (np.array(self.row_anchor) * self.train_height).astype(int)
        row_anchor_pix = np.clip(row_anchor_pix, 0, self.train_height-1)
        num_row_cls = len(row_anchor_pix)

        lane_x = np.full((self.num_lanes, num_row_cls), -1, dtype=np.float32)

        for lane_idx in range(1, self.num_lanes+1):
            for i, y in enumerate(row_anchor_pix):
                # 防止索引越界
                if y >= self.train_height:
                    continue
                row_data = seg_gray[y, :]
                pos = np.where(row_data == lane_idx)[0]
                if len(pos) > 0:
                    x = np.mean(pos)
                    lane_x[lane_idx-1, i] = x

        x_row = torch.from_numpy(lane_x).float()  # [lanes, num_row_cls]

        # 可选：延伸底部
        x_row_ext = extend_coords(x_row)          # [lanes, num_row_cls]
        # 转置为 [num_row_cls, lanes]
        x_row_ext = x_row_ext.T

        # 计算列锚点在crop/resize后图像上的 x 坐标（像素)
        col_anchor_pix = (np.array(self.col_anchor) * self.train_width).astype(int)
        col_anchor_pix = np.clip(col_anchor_pix, 0, self.train_width-1)
        num_col_cls = len(col_anchor_pix)

        lane_y = np.full((self.num_lanes, num_col_cls), -1, dtype=np.float32)

        for lane_idx in range(1, self.num_lanes+1):
            for i, x in enumerate(col_anchor_pix):
                if x >= self.train_width:
                    continue
                col_data = seg_gray[:, x]
                pos = np.where(col_data == lane_idx)[0]
                if len(pos) > 0:
                    y = np.mean(pos)
                    lane_y[lane_idx-1, i] = y

        y_col = torch.from_numpy(lane_y).float()  # [lanes, num_col_cls]

        # 转置为 [num_col_cls, lanes]
        y_col = y_col.T

        # ========== 生成分类和回归标签 ==========
        # 行方向标签（基于resize/crop后的尺寸）
        labels_row = (x_row_ext / self.train_width * (self.num_cell_row - 1)).long()
        labels_row[x_row_ext < 0] = -1
        labels_row[x_row_ext > self.train_width] = -1
        labels_row = torch.clamp(labels_row, min=-1, max=self.num_cell_row - 1)

        # 行方向回归标签
        labels_row_float = x_row_ext / self.train_width
        labels_row_float = torch.clamp(labels_row_float, min=-1.0, max=1.0)
        labels_row_float[(x_row_ext < 0) | (x_row_ext > self.train_width)] = -1.0

        # 列方向标签
        labels_col = (y_col / self.train_height * (self.num_cell_col - 1)).long()
        labels_col[y_col < 0] = -1
        labels_col[y_col > self.train_height] = -1
        labels_col = torch.clamp(labels_col, min=-1, max=self.num_cell_col - 1)

        # 列方向回归标签
        labels_col_float = y_col / self.train_height
        labels_col_float = torch.clamp(labels_col_float, min=-1.0, max=1.0)
        labels_col_float[(y_col < 0) | (y_col > self.train_height)] = -1.0

        result = {
            'images': img,                         # [3, H, W]
            'labels_row': labels_row,               # [num_row_cls, lanes]
            'labels_col': labels_col,               # [num_col_cls, lanes]
            'labels_row_float': labels_row_float,
            'labels_col_float': labels_col_float
        }
        if self.use_aux:
            assert self.seg_transform is not None
            seg_label = self.seg_transform(seg_tensor)
            result['seg_images'] =  seg_label, 
        return result

# 使用示例（完整可运行）
if __name__ == '__main__':
    # 设置随机种子确保结果可复现
    # random.seed(42)
    # np.random.seed(42)
    # torch.manual_seed(42)

    # 配置参数
    num_row = 56
    num_col = 30
    # 针对Tusimple数据集的锚点设置（适配720高度）
    row_anchor = np.linspace(260, 1080, num_row) / 1080  # 从160开始避免顶部无效区域
    col_anchor = np.linspace(0, 1, num_col)
    
    # 创建数据集
    dataset = TrainDataset(
        data_root=r'D:\UFLD\Ultra-Fast-Lane-Detection-v2-master\UFLD_dataset',
        list_path=r'D:\UFLD\Ultra-Fast-Lane-Detection-v2-master\UFLD_dataset\train_gt.txt',
        row_anchor=row_anchor,
        col_anchor=col_anchor,
        train_width=800,
        train_height=320,
        num_cell_row=100,
        num_cell_col=100,
        dataset_name='Tusimple',
        top_crop=0.8,
        num_lanes=2
    )
    
    print(f"数据集总样本数: {len(dataset)}")
    
    # 加载一个样本测试
    try:
        sample = dataset[5]
        
        # 打印各张量形状
        print("\n=== 样本张量形状 ===")
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v.shape}")
        
        # 打印标签示例
        print("\n=== 行标签示例===")
        print(sample['labels_row'])
        print("\n=== 列标签示例===")
        print(sample['labels_col'])
        
        # 可视化图像
        print("\n=== 图像可视化 ===")
        img = sample['images']
        # 反归一化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)  # 限制到[0,1]范围
        img_np = img.permute(1, 2, 0).cpu().numpy()
        
        # 可视化原始图像和分割图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 显示图像
        ax1.imshow(img_np)
        ax1.set_title('Input Image')
        ax1.axis('off')
        
        # 显示分割图
        seg = sample['seg_images'].float() / sample['seg_images'].max()  # 归一化到[0,1]
        seg_np = seg.permute(1, 2, 0).cpu().numpy()  # [3,H,W] → [H,W,3]

        ax2.imshow(seg_np, cmap='jet')
        ax2.set_title('Segmentation Mask')
        ax2.axis('off')

        # ===== 新增：绘制行锚点 =====
        train_width = dataset.train_width
        train_height = dataset.train_height
        # 重新计算行锚点在最终图像上的 y 坐标（与 __getitem__ 中一致）
        row_anchor_pix = (np.array(dataset.row_anchor) * train_height).astype(int)
        row_anchor_pix = np.clip(row_anchor_pix, 0, train_height-1)

        # 获取行方向归一化浮点标签，形状 [num_row_cls, lanes]
        labels_row_float = sample['labels_row_float'].cpu().numpy()

        colors = ['red', 'blue']  # 左线红色，右线蓝色
        for lane_idx in range(dataset.num_lanes):
            x_vals = labels_row_float[:, lane_idx]          # 该车道线所有锚点的 x 归一化值
            valid = x_vals > 0                               # 有效点（>0）
            x_pix = x_vals[valid] * train_width             # 还原像素坐标
            y_pix = row_anchor_pix[valid]                   # 对应 y 坐标
            ax1.scatter(x_pix, y_pix, c=colors[lane_idx], s=10, label=f'Lane {lane_idx+1}')
        ax1.legend()
        
        plt.tight_layout()
        plt.show()
        
        # 打印标签统计信息
        print("\n=== 标签统计 ===")
        row_labels_valid = sample['labels_row'][sample['labels_row'] != -1]
        col_labels_valid = sample['labels_col'][sample['labels_col'] != -1]
        
        print(f"有效行标签数: {len(row_labels_valid)}/{sample['labels_row'].numel()}")
        print(f"有效列标签数: {len(col_labels_valid)}/{sample['labels_col'].numel()}")
        if len(row_labels_valid) > 0:
            print(f"行标签范围: {row_labels_valid.min()} ~ {row_labels_valid.max()}")
        if len(col_labels_valid) > 0:
            print(f"列标签范围: {col_labels_valid.min()} ~ {col_labels_valid.max()}")
            
    except Exception as e:
        print(f"处理样本时出错: {e}")
        import traceback
        traceback.print_exc()