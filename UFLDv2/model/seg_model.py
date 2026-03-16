import torch
from UFLDv2.utils.common import initialize_weights

class conv_bn_relu(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=False):
        super(conv_bn_relu,self).__init__()
        self.conv = torch.nn.Conv2d(in_channels,out_channels, kernel_size, 
            stride = stride, padding = padding, dilation = dilation,bias = bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    

class SegHead(torch.nn.Module):
    def __init__(self,backbone, num_lanes):
        super(SegHead, self).__init__()

        self.aux_header2 = torch.nn.Sequential(
            conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(128,128,3,padding=1),
            conv_bn_relu(128,128,3,padding=1),
            conv_bn_relu(128,128,3,padding=1),      #这里的输出为128通道，尺寸和输入一致
        )
        self.aux_header3 = torch.nn.Sequential(
            conv_bn_relu(256, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(1024, 128, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(128,128,3,padding=1),
            conv_bn_relu(128,128,3,padding=1),      #这里输出为128通道，尺寸和输入一致
        )
        self.aux_header4 = torch.nn.Sequential(
            conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(2048, 128, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(128,128,3,padding=1),      #输出128通道，尺寸和输入一致
        )
        self.aux_combine = torch.nn.Sequential(     #多尺度融合并输出最终预测
            conv_bn_relu(384, 256, 3,padding=2,dilation=2),  # 膨胀卷积：扩大感受野
            conv_bn_relu(256, 128, 3,padding=2,dilation=2),  # 更大膨胀率：覆盖更多上下文
            conv_bn_relu(128, 128, 3,padding=2,dilation=2),
            conv_bn_relu(128, 128, 3,padding=4,dilation=4), 
            torch.nn.Conv2d(128, num_lanes+1, 1)             # 1×1卷积：通道压缩到类别数（车道线+背景）
            # output : n, num_of_lanes+1, h, w
        )   #输入时拼接后的128 * 3通道的特征

        initialize_weights(self.aux_header2,self.aux_header3,self.aux_header4,self.aux_combine)

        # self.droput = torch.nn.Dropout(0.1)
    def forward(self,x2,x3,fea):
        # 1. 处理layer2特征：尺寸不变（H/8, W/8）
        x2 = self.aux_header2(x2)

        # 2. 处理layer3特征：先卷积，再上采样2倍（从H/16→H/8，和x2对齐）
        x3 = self.aux_header3(x3)
        x3 = torch.nn.functional.interpolate(x3,scale_factor = 2,mode='bilinear')
        # interpolate 默认只对空间维度（即高度和宽度）进行插值，而不能选择其他维度（如通道或批量）

        # 3. 处理layer4特征：先卷积，再上采样4倍（从H/32→H/8，和x2对齐）
        x4 = self.aux_header4(fea)
        x4 = torch.nn.functional.interpolate(x4,scale_factor = 4,mode='bilinear')
        aux_seg = torch.cat([x2,x3,x4],dim=1) # 按通道维度拼接
        aux_seg = self.aux_combine(aux_seg) #输出维度时[B, num_lanes+1, H/8, W/8]
        return aux_seg