import torch
import os
import sys
import numpy as np

# ========== 关键：将项目根目录加入Python路径 ==========
# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录（假设data文件夹在根目录下，向上退一级）
project_root = os.path.dirname(current_dir)
# 将根目录加入Python搜索路径
sys.path.append(project_root)

# ========== 正确导入configs模块 ==========
try:
    from configs.tusimple_res18 import cfg  # 导入配置（推荐方式）
except ImportError:
    # 兼容备选导入方式（防止命名问题）
    import configs.tusimple_res18 as cfg

from data.torch_dataset import TrainDataset  # 导入你的TrainDataset
from torchvision import transforms
import data.mytransforms as mytransforms

def get_train_loader_v1(batch_size, data_root, distributed=False, num_lanes=4):
    """
    仅针对Tusimple数据集构建训练数据加载器
    Args:
        batch_size: 批次大小
        data_root: Tusimple数据集根目录
        distributed: 是否分布式训练（默认False）
        num_lanes: 车道线数量（Tusimple默认4）
    Returns:
        train_loader: Tusimple训练数据加载器
        cls_num_per_lane: 每个车道线的锚点数量（固定56）
    """
    # ========== 1. 配置Tusimple专属锚点 ==========
    cfg.row_anchor = np.linspace(260, cfg.origin_height, cfg.num_row) / cfg.origin_height
    cfg.col_anchor = np.linspace(0, 1, cfg.num_col)

    segment_transform = transforms.Compose([
        mytransforms.FreeScaleMask((cfg.train_height // 8, cfg.train_width // 8)),
        mytransforms.MaskToTensor(),
    ])

    # ========== 2. 初始化Tusimple数据集 ==========
    train_dataset = TrainDataset(
        data_root=data_root,
        list_path=os.path.join(data_root, 'train_gt.txt'),
        row_anchor=cfg.row_anchor,
        col_anchor=cfg.col_anchor,
        train_width=cfg.train_width,
        train_height=cfg.train_height,
        num_cell_row=cfg.num_cell_row,
        num_cell_col=cfg.num_cell_col,
        dataset_name='Tusimple',
        top_crop=0.8,  # Tusimple专属top_crop值
        num_lanes=num_lanes,use_aux = cfg.use_aux,
        segment_transform = segment_transform
    )

    # ========== 3. 配置采样器（兼容分布式） ==========
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            shuffle=True  # 训练时打乱数据
        )
    else:
        sampler = torch.utils.data.RandomSampler(train_dataset)

    # ========== 4. 配置DataLoader（适配系统） ==========
    # Windows系统num_workers设为0，Linux/Mac设为4
    num_workers = 0 if os.name == 'nt' else 4
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,  # 加速GPU数据传输
        drop_last=True,   # 丢弃最后一个不完整批次
        persistent_workers=True if num_workers > 0 else False  # 提升加载效率
    )

    return train_loader


# ========== 测试代码（验证Tusimple加载器） ==========
if __name__ == '__main__':
    # 配置参数（根据你的实际路径修改）
    BATCH_SIZE = 2
    DATA_ROOT = r'D:\UFLD\Ultra-Fast-Lane-Detection-v2-master\UFLD_dataset'
    NUM_LANES = 2  # 测试用车道线数量，实际可改为4

    # 获取数据加载器
    train_loader= get_train_loader_v1(
        batch_size=BATCH_SIZE,
        data_root=DATA_ROOT,
        distributed=False,
        num_lanes=NUM_LANES
    )

    # 打印基础信息
    print(f"✅ Tusimple数据加载器初始化完成")
    print(f"📦 批次大小: {BATCH_SIZE}")
    print(f"🔢 总批次数量: {len(train_loader)}")

    # 迭代验证第一个批次
    for batch_idx, batch_data in enumerate(train_loader):
        print(f"\n=== 第 {batch_idx+1} 批次数据信息 ===")
        print(f"图像尺寸: {batch_data['images'].shape}")       # [batch, 3, H, W]
        print(f"分割图尺寸: {batch_data['seg_images'].shape}") # [batch, 1, H, W]
        print(f"行标签尺寸: {batch_data['labels_row'].shape}") # [batch, num_cls, num_lanes]
        print(f"列标签尺寸: {batch_data['labels_col'].shape}") # [batch, num_cls, num_lanes]
        break  # 仅验证第一个批次，避免耗时