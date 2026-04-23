#pragma once
#include <vector>
#include <utility>

// 处理 ONNX 输出的后处理，返回左右车道线坐标点
// 参数说明：
//   loc_row, loc_col, exist_row, exist_col : 模型输出的四个张量 (batch=1, 已经展平为一维数组)
//   shape_row : [num_grid_row, num_cls_row, num_lane_row]
//   shape_col : [num_grid_col, num_cls_col, num_lane_col]
//   img_w, img_h : 原始图像宽高
//   row_anchor, col_anchor : 锚点数组 (归一化坐标，长度分别为 num_cls_row, num_cls_col)
//   local_width : 局部搜索半径 (默认1)
// 返回值: 两个 vector，每个元素为 (x, y) 整数坐标
std::pair<std::vector<std::pair<int, int>>, std::vector<std::pair<int, int>>>
pred2coords_cpp(
    const std::vector<float>& loc_row,
    const std::vector<float>& loc_col,
    const std::vector<float>& exist_row,
    const std::vector<float>& exist_col,
    const std::vector<int>& shape_row,   // [num_grid_row, num_cls_row, num_lane_row]
    const std::vector<int>& shape_col,   // [num_grid_col, num_cls_col, num_lane_col]
    float img_w, float img_h,
    const std::vector<float>& row_anchor,
    const std::vector<float>& col_anchor,
    int local_width = 1
);