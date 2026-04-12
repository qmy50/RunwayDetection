import torch
import torch.nn as nn
from model.backbone import resnet
import numpy as np
from UFLDv2.utils.common import initialize_weights
from UFLDv2.model.seg_model import SegHead
from UFLDv2.model.layer import CoordConv

# from utils.common import initialize_weights
# from model.seg_model import SegHead
# from model.layer import CoordConv

class parsingNet(nn.Module):
    def __init__(self, pretrained=True, backbone='50', num_grid_row = None, num_cls_row = None, 
                 num_grid_col = None, num_cls_col = None, num_lane_on_row = None, num_lane_on_col = None, 
                 use_aux=False,input_height = None, input_width = None, fc_norm = False, use_ema= True):
        super(parsingNet, self).__init__()
        self.num_grid_row = num_grid_row
        self.num_cls_row = num_cls_row
        self.num_grid_col = num_grid_col
        self.num_cls_col = num_cls_col
        self.num_lane_on_row = num_lane_on_row
        self.num_lane_on_col = num_lane_on_col
        self.use_aux = use_aux
        self.dim1 = self.num_grid_row * self.num_cls_row * self.num_lane_on_row
        self.dim2 = self.num_grid_col * self.num_cls_col * self.num_lane_on_col
        self.dim3 = 2 * self.num_cls_row * self.num_lane_on_row         #2：表示每个行锚点-车道线对的“存在/不存在”二分类（对应softmax的2个维度）
        self.dim4 = 2 * self.num_cls_col * self.num_lane_on_col         #2：表示每个行锚点-车道线对的“存在/不存在”二分类（对应softmax的2个维度）
        self.total_dim = self.dim1 + self.dim2 + self.dim3 + self.dim4
        mlp_mid_dim = 2048
        self.input_dim = input_height // 32 * input_width // 32 * 8 #2是核心，代表模型对每个 “行锚点 - 车道线对” 预测二分类概率（存在 / 不存在）
        # 如果是320*800的话计算得到2000，288*800计算得到1800
        feat_channels = 512 if backbone  in ['34','18', '34fca'] else 2048
        if use_ema:
            self.em_attention = EMAttention(channels = feat_channels, factor = 8)   # factor 可调
        else:
            self.em_attention = nn.Identity()
        self.model = resnet(backbone, pretrained=pretrained)

        # for avg pool experiment
        # self.pool = torch.nn.AdaptiveAvgPool2d(1)
        # self.pool = torch.nn.AdaptiveMaxPool2d(1)

        # self.register_buffer('coord', torch.stack([torch.linspace(0.5,9.5,10).view(-1,1).repeat(1,50), torch.linspace(0.5,49.5,50).repeat(10,1)]).view(1,2,10,50))

        self.cls = torch.nn.Sequential(
            torch.nn.LayerNorm(self.input_dim) if fc_norm else torch.nn.Identity(),
            torch.nn.Linear(self.input_dim, mlp_mid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_mid_dim, self.total_dim),
        )
        self.pool = torch.nn.Conv2d(512,8,1) if backbone in ['34','18', '34fca'] else torch.nn.Conv2d(2048,8,1)
        if self.use_aux:
            self.seg_head = SegHead(backbone, num_lane_on_row + num_lane_on_col)
            print('辅助分割维度验证======')
        initialize_weights(self.cls)
    def forward(self, x):

        x2,x3,fea = self.model(x)
        if self.use_aux:
            seg_out = self.seg_head(x2, x3,fea)

        fea = self.em_attention(fea)                      # 注意力增强
        fea = self.pool(fea)
        # print(fea.shape)
        # print(self.coord.shape)
        # fea = torch.cat([fea, self.coord.repeat(fea.shape[0],1,1,1)], dim = 1)
        
        fea = fea.view(-1, self.input_dim)
        out = self.cls(fea)

        # 拆分输出为4个预测分支，封装成字典
        pred_dict = {
            # 行方向位置预测：[B, num_grid_row, num_cls_row, num_lane_on_row] → [B,37,56,2]
            'loc_row': out[:,:self.dim1].view(-1,self.num_grid_row, self.num_cls_row, self.num_lane_on_row), 
            # 列方向位置预测：[B, num_grid_col, num_cls_col, num_lane_on_col] → [B,37,10,2]
            'loc_col': out[:,self.dim1:self.dim1+self.dim2].view(-1, self.num_grid_col, self.num_cls_col, self.num_lane_on_col),
            # 行方向存在性预测：[B,2,num_cls_row,num_lane_on_row] → [B,2,56,2]（2=存在/不存在）
            'exist_row': out[:,self.dim1+self.dim2:self.dim1+self.dim2+self.dim3].view(-1, 2, self.num_cls_row, self.num_lane_on_row), 
            # 列方向存在性预测：[B,2,num_cls_col,num_lane_on_col] → [B,2,10,2]
            'exist_col': out[:,-self.dim4:].view(-1, 2, self.num_cls_col, self.num_lane_on_col)
        }
        if self.use_aux:
            pred_dict['seg_out'] = seg_out    #上边的exist_row的最终维度是(batch_size, 2, num_cls_row, num_lane_on_row)
        
        return pred_dict            #loc_row是预测每个行锚点 - 车道线对在 num_grid_row 个网格中的位置概率
                                    #exit_row是预测每个行锚点 - 车道线对 “存在 / 不存在” 的二分类概率
    def forward_tta(self, x):
        x2,x3,fea = self.model(x)

        pooled_fea = self.pool(fea)
        n,c,h,w = pooled_fea.shape

        left_pooled_fea = torch.zeros_like(pooled_fea)
        right_pooled_fea = torch.zeros_like(pooled_fea)
        up_pooled_fea = torch.zeros_like(pooled_fea)
        down_pooled_fea = torch.zeros_like(pooled_fea)

        left_pooled_fea[:,:,:,:w-1] = pooled_fea[:,:,:,1:]
        left_pooled_fea[:,:,:,-1] = pooled_fea.mean(-1)
        
        right_pooled_fea[:,:,:,1:] = pooled_fea[:,:,:,:w-1]
        right_pooled_fea[:,:,:,0] = pooled_fea.mean(-1)

        up_pooled_fea[:,:,:h-1,:] = pooled_fea[:,:,1:,:]
        up_pooled_fea[:,:,-1,:] = pooled_fea.mean(-2)

        down_pooled_fea[:,:,1:,:] = pooled_fea[:,:,:h-1,:]
        down_pooled_fea[:,:,0,:] = pooled_fea.mean(-2)
        # 10 x 25
        fea = torch.cat([pooled_fea, left_pooled_fea, right_pooled_fea, up_pooled_fea, down_pooled_fea], dim = 0)
        fea = fea.view(-1, self.input_dim)

        out = self.cls(fea)

        return {'loc_row': out[:,:self.dim1].view(-1,self.num_grid_row, self.num_cls_row, self.num_lane_on_row), 
                'loc_col': out[:,self.dim1:self.dim1+self.dim2].view(-1, self.num_grid_col, self.num_cls_col, self.num_lane_on_col),
                'exist_row': out[:,self.dim1+self.dim2:self.dim1+self.dim2+self.dim3].view(-1, 2, self.num_cls_row, self.num_lane_on_row), 
                'exist_col': out[:,-self.dim4:].view(-1, 2, self.num_cls_col, self.num_lane_on_col)}


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


def get_model(cfg):
    return parsingNet(pretrained = True, backbone=cfg.backbone, 
                      num_grid_row = cfg.num_cell_row, num_cls_row = cfg.num_row, 
                      num_grid_col = cfg.num_cell_col, num_cls_col = cfg.num_col, 
                      num_lane_on_row = cfg.num_lanes, num_lane_on_col = cfg.num_lanes, 
                      use_aux = cfg.use_aux, input_height = cfg.train_height, 
                      input_width = cfg.train_width, fc_norm = cfg.fc_norm,use_ema = cfg.use_ema).cuda()