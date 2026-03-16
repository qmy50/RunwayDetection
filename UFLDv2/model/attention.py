import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
class SENet(nn.Module):
    def __init__(self, channel, reduction_ratio=16):
        """
        初始化SENet模块。
        Args:
            channel: 输入特征图的通道数。
            reduction_ratio: 第一个全连接层的降维比例，默认16。
        """
        super(SENet, self).__init__()
        # 第一步：Squeeze，全局平均池化，将每个通道的HxW压缩为1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)     
 
        # 第二步：Excitation，两个全连接层构成的门控机制
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio, bias=False), # 降维，减少参数量
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel, bias=False), # 升维回原通道数
            nn.Sigmoid() # 将权重限制在0~1之间
        )
 
    def forward(self, x):
        """
        前向传播。
        Args:
            x: 输入张量，形状为 [batch_size, channel, height, width]
        Returns:
            加权后的特征图。
        """
        b, c, h, w = x.size()
        # Squeeze: [b, c, h, w] -> [b, c, 1, 1] -> [b, c],展平后传给全连接层
        avg = self.avg_pool(x).view(b, c) 
 
        # Excitation: [b, c] -> [b, c] (经过FC和Sigmoid)
        weights = self.fc(avg).view(b, c, 1, 1) # 调整形状为[b, c, 1, 1]便于广播相乘
 
        # Scale: 将权重乘回原特征图
        return x * weights
 

class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, channel, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        # 同时使用平均池化和最大池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
 
        # 共享的全连接层（多层感知机）
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        b, c, h, w = x.size()
        # 平均池化路径
        avg_out = self.mlp(self.avg_pool(x).view(b, c))  # 输入形状为(b,c)输出形状为（b，c）
        # 最大池化路径
        max_out = self.mlp(self.max_pool(x).view(b, c))  # 形状同上
        # 相加后激活
        channel_weights = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * channel_weights
 
class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 用卷积层生成空间权重图，输入通道为2（最大和平均），输出通道为1
        # 通常使用7x7的大卷积核来获取较大的感受野
        padding = kernel_size // 2  # 保持尺寸不变
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False) #（h - 7+ 2*3）/1 + 1 = h,w同理保持原尺寸
        self.sigmoid = nn.Sigmoid()  # 输出图片维度是（b，1，h，w）且所有值位于0-1之间
 
    def forward(self, x):
        b, c, h, w = x.size()
        # 沿通道维度做最大池化和平均池化
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [b, 1, h, w]
        max_out, _ = torch.max(x, dim=1, keepdim=True) # [b, 1, h, w]
        # 在通道维度上拼接
        concat = torch.cat([avg_out, max_out], dim=1)  # [b, 2, h, w]
        # 卷积生成空间权重图
        spatial_weights = self.sigmoid(self.conv(concat))  # [b, 1, h, w]
        return x * spatial_weights
 
class CBAM(nn.Module):
    """完整的CBAM模块：先通道，后空间"""
    def __init__(self, channel, reduction_ratio=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channel, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=spatial_kernel)
 
    def forward(self, x):
        # 顺序应用通道注意力和空间注意力
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x
    
def draw_heatmap(img, attention_map):
    """
    img: 原图 (H, W, 3) 0~255
    attention_map: 注意力图 (H', W') 数值任意
    """
    # 1. 缩放到原图大小
    heatmap = cv2.resize(attention_map, (img.shape[1], img.shape[0]))
    
    # 2. 归一化 0~1
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # 3. 转成彩色热力图
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # # 4. 与原图融合
    # overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    
    return heatmap


import torch
import torch.nn as nn

class EMAttention(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMAttention, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

if __name__ == '__main__':
    test_input = torch.randn(2, 64, 32, 32)
    model = EMAttention(channels=64, factor=16)
    output = model(test_input)
    print("输入 shape:", test_input.shape)
    print("输出 shape:", output.shape)
    print("✅ 运行成功！")

# # 测试
# if __name__ == '__main__':
#     test_input = torch.randn(2, 512, 26, 26)
#     cbam_block = CBAM(channel=512)
#     output = cbam_block(test_input)
#     print(f"CBAM输入形状: {test_input.shape}")
#     print(f"CBAM输出形状: {output.shape}")
 
#     # 可视化一下权重（这里用打印部分值代替）
#     # 我们可以单独运行通道注意力，看看权重
#     ca = ChannelAttention(512)
#     ca_output = ca(test_input)
#     # 模拟查看空间注意力权重图的一个切片
#     sa = SpatialAttention()
#     # 需要先有个输入，这里用个全1矩阵简单看一下
#     img = torch.randint(0,255,size = (1,3,200,200)).float()
#     avg_out = torch.mean(img, dim=1, keepdim=True)  # [b, 1, h, w]
#     max_out, _ = torch.max(img, dim=1, keepdim=True)# [b, 1, h, w]

#     dummy_for_sa = torch.cat([avg_out, max_out], dim=1)
#     print(dummy_for_sa)
#     sa_weight = sa.conv(dummy_for_sa)

#     print(f"空间注意力卷积核大小: {sa.conv.weight.shape}")
#     print(f'空间权重矩阵维度:{sa_weight.shape}')
#     sa_weight = sa_weight.squeeze().detach().numpy()
#     img = img.squeeze().permute(1,2,0)
#     img = img.detach().numpy().astype(np.uint8) 
#     print(img.shape)
#     cv2.imshow('Origin',img)
#     heapmap = draw_heatmap(img, sa_weight)
#     cv2.imshow('heap',heapmap)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


