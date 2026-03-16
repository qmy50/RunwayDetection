import cv2
import torch
import numpy as np
import os

def draw_points(img, points, color):
    """绘制点到图片上（复用原逻辑）"""
    points = points.view(-1,2).cpu().numpy()
    for (x,y) in points:
        if x < 0 or y < 0:
            continue
        img = cv2.circle(img, (int(x), int(y)), 5, color, -1)
    return img

def _linear_interp_torch(x, xp, fp, left=None, right=None):
    """
    手动实现线性插值（兼容旧版本 PyTorch）
    x: 目标位置 [M]
    xp: 原始位置 [N]
    fp: 原始值 [N]
    left: 左边界填充值
    right: 右边界填充值
    """
    if len(xp) == 0:
        return torch.full_like(x, fp[0] if len(fp) > 0 else 0.0)
    
    x = x.to(torch.float64)
    xp = xp.to(torch.float64)
    fp = fp.to(torch.float64)
    
    # 找到 x 在 xp 中的位置
    idx = torch.searchsorted(xp, x)
    
    # 处理边界情况
    idx = torch.clamp(idx, 1, len(xp) - 1)
    
    # 获取左右邻居
    x0 = xp[idx - 1]
    x1 = xp[idx]
    f0 = fp[idx - 1]
    f1 = fp[idx]
    
    # 线性插值
    t = (x - x0) / (x1 - x0 + 1e-10)
    result = f0 + t * (f1 - f0)
    
    # 处理边界填充
    if left is not None:
        result = torch.where(x < xp[0], torch.tensor(left, dtype=result.dtype, device=result.device), result)
    if right is not None:
        result = torch.where(x > xp[-1], torch.tensor(right, dtype=result.dtype, device=result.device), result)
    
    return result.to(fp.dtype)


def run(input_tensor, interp_loc, direction):
    """
    原CUDA插值的Python等价实现
    Args:
        input_tensor: 输入张量 [B, 4, 35, 2]，B=1（批次维度）
        interp_loc: 插值位置张量 [N]，N是插值后点数（如30）
        direction: 插值方向 0=水平（x轴，基于y插值），1=垂直（y轴，基于x插值）
    Returns:
        new_all_points: 插值后的坐标张量 [B, 4, N, 2]
    """
    # 分离x/y坐标 [B,4,35]
    x_coords = input_tensor[..., 0]  # [1,4,35]
    y_coords = input_tensor[..., 1]  # [1,4,35]
    
    B, num_lanes, num_anchors = x_coords.shape
    num_interp = interp_loc.shape[0]
    
    # 初始化插值结果
    new_x = torch.ones((B, num_lanes, num_interp), device=input_tensor.device) * -99999
    new_y = torch.ones((B, num_lanes, num_interp), device=input_tensor.device) * -99999

    if direction == 0:  # 水平插值（基于y轴位置插值x）
        # 遍历每个批次、每条车道
        for b in range(B):
            for lane in range(num_lanes):
                # 筛选有效坐标（x≠-99999）
                valid_mask = x_coords[b, lane] != -99999
                if not valid_mask.any():
                    continue  # 无有效点则跳过
                
                # 提取有效y和对应的x
                valid_y = y_coords[b, lane, valid_mask]
                valid_x = x_coords[b, lane, valid_mask]
                
                # 线性插值：基于interp_loc（目标y）插值x
                # torch.interp要求x坐标升序，先排序
                sorted_idx = torch.argsort(valid_y)
                sorted_y = valid_y[sorted_idx]
                sorted_x = valid_x[sorted_idx]
                
                # 执行插值（超出范围的用边界值填充）
                interp_x = _linear_interp_torch(
                    interp_loc,  # 目标y位置
                    sorted_y,    # 原始y位置
                    sorted_x,    # 原始x坐标
                    left=sorted_x[0].item() if sorted_x.numel() > 0 else 0,  # 左边界填充
                    right=sorted_x[-1].item() if sorted_x.numel() > 0 else 0 # 右边界填充
                )
                
                # 赋值到结果张量
                new_x[b, lane] = interp_x
                new_y[b, lane] = interp_loc  # 目标y位置
    
    elif direction == 1:  # 垂直插值（基于x轴位置插值y）
        # 逻辑与水平插值对称
        for b in range(B):
            for lane in range(num_lanes):
                valid_mask = y_coords[b, lane] != -99999
                if not valid_mask.any():
                    continue
                
                valid_x = x_coords[b, lane, valid_mask]
                valid_y = y_coords[b, lane, valid_mask]
                
                sorted_idx = torch.argsort(valid_x)
                sorted_x = valid_x[sorted_idx]
                sorted_y = valid_y[sorted_idx]
                
                interp_y = _linear_interp_torch(
                    interp_loc,
                    sorted_x,
                    sorted_y,
                    left=sorted_y[0].item() if sorted_y.numel() > 0 else 0,
                    right=sorted_y[-1].item() if sorted_y.numel() > 0 else 0
                )
                
                new_y[b, lane] = interp_y
                new_x[b, lane] = interp_loc

    # 合并x/y坐标 [B,4,N,2]
    new_all_points = torch.stack([new_x, new_y], dim=-1)
    return new_all_points

def test(culane_root):
    """测试函数（替换CUDA调用为Python实现）"""
    # 1. 加载CULane标注文件
    test_lines_txt_path = os.path.join(culane_root, 'driver_161_90frame/06031919_0929.MP4/00000.lines.txt')
    lanes = open(test_lines_txt_path, 'r').readlines()

    # 2. 初始化车道线坐标矩阵
    all_points = np.zeros((4,35,2), dtype=np.float)
    the_anno_row_anchor = np.array([250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 
                                    350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 
                                    450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 
                                    550, 560, 570, 580, 590])
    all_points[:,:,1] = np.tile(the_anno_row_anchor, (4,1))  # y轴赋值
    all_points[:,:,0] = -99999  # x轴初始化为无效值

    # 3. 加载标签图并填充坐标
    label_img_path = os.path.join(culane_root, 'laneseg_label_w16/driver_161_90frame/06031919_0929.MP4/00000.png')
    label_img = cv2.imread(label_img_path)[:,:,0]

    for lane_idx , lane in enumerate(lanes):
        ll = lane.strip().split(' ')
        point_x = ll[::2]
        point_y = ll[1::2]

        # 计算车道归属
        mid_idx = int(len(point_x)/2)
        mid_x = int(float(point_x[mid_idx]))
        mid_y = int(float(point_y[mid_idx]))
        lane_order = label_img[mid_y-1, mid_x - 1]
        if lane_order == 0:
            continue  # 无效车道跳过（替代原pdb调试）

        # 填充x坐标到对应锚点
        for i in range(len(point_x)):
            p1x = float(point_x[i])
            pos = (int(point_y[i]) - 250) / 10
            all_points[lane_order - 1, int(pos), 0] = p1x

    # 4. 转换为CUDA张量（也可CPU运行，去掉.cuda()即可）
    all_points = torch.tensor(all_points).cuda().view(1,4,35,2)
    new_interp_locations = torch.linspace(0,590,30).cuda()

    # 5. 调用Python实现的插值（替换原my_interp.run）
    new_all_points = run(all_points.float(), new_interp_locations.float(), 0)

    # 6. 可视化
    img_path = os.path.join(culane_root, 'driver_161_90frame/06031919_0929.MP4/00000.png')
    img = cv2.imread(img_path)
    if img is None:  # 备用：用标签图可视化
        img = cv2.imread(label_img_path) * 128
    img = draw_points(img, all_points, (0,255,0))   # 原始点（绿色）
    img = draw_points(img, new_all_points, (0,0,255)) # 插值点（红色）
    cv2.imwrite('test_python.png', img)

    # 打印结果验证
    torch.set_printoptions(sci_mode=False)
    print("原始坐标示例：", all_points[0,0,:5])
    print("插值后坐标示例：", new_all_points[0,0,:5])

if __name__ == "__main__":
    # 替换为你的CULane数据集根路径
    test('/path/to/your/culane')