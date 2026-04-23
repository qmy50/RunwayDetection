#include "lane_postprocess.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>
#include <iostream>

static inline float softmax_sum(const std::vector<float>& x, const std::vector<int>& indices) {
    // x 和 indices 长度相同
    float max_val = x[0];
    for (size_t i = 1; i < x.size(); ++i) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum_exp = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        sum_exp += std::exp(x[i] - max_val);
    }
    float weighted = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        float prob = std::exp(x[i] - max_val) / sum_exp;
        weighted += prob * indices[i];   // 用 indices[i] 作为位置权重
    }
    return weighted;

}

std::pair<std::vector<std::pair<int, int>>, std::vector<std::pair<int, int>>>
pred2coords_cpp(
    const std::vector<float>& loc_row,
    const std::vector<float>& loc_col,
    const std::vector<float>& exist_row,
    const std::vector<float>& exist_col,
    const std::vector<int>& shape_row,
    const std::vector<int>& shape_col,
    float img_w, float img_h,
    const std::vector<float>& row_anchor,
    const std::vector<float>& col_anchor,
    int local_width)
{
    // int start = 10;
    // int end = 20;
    // std::vector<int> indices(end - start + 1);
    // std::iota(indices.begin(), indices.end(), start);
    // for(auto item:indices){
    //     std::cout << item << std::endl;
    // }

    int num_grid_row = shape_row[0];
    int num_cls_row  = shape_row[1];
    int num_lane_row = shape_row[2];
    int num_grid_col = shape_col[0];
    int num_cls_col  = shape_col[1];
    int num_lane_col = shape_col[2];


    int step_row_cls    = num_lane_row;
    int step_row_grid   = num_cls_row * num_lane_row;
    int step_col_cls    = num_lane_col;
    int step_col_grid   = num_cls_col * num_lane_col;
    int chan_offset_row = num_cls_row * num_lane_row;
    int chan_offset_col = num_cls_col * num_lane_col;

    std::vector<int> max_indices_row(num_cls_row * num_lane_row, 0);
    std::vector<int> valid_row(num_cls_row * num_lane_row, 0);

    for (int k = 0; k < num_cls_row; ++k) {
        for (int i = 0; i < num_lane_row; ++i) {
            int base = k * step_row_cls + i;
            int max_idx = 0;
            float max_val = loc_row[base];
            for (int g = 1; g < num_grid_row; ++g) {
                float v = loc_row[g * step_row_grid + base];
                if (v > max_val) {
                    max_val = v;
                    max_idx = g;
                }
            }
            max_indices_row[base] = max_idx;

            float exist_max_val = exist_row[base];
            int exist_max_idx = 0;
            float v1 = exist_row[chan_offset_row + base];
            if (v1 > exist_max_val) {
                exist_max_idx = 1;
            }
            valid_row[base] = exist_max_idx;
        }
    }
    std::vector<int> max_indices_col(num_cls_col * num_lane_col, 0);
    std::vector<int> valid_col(num_cls_col * num_lane_col, 0);

    for (int k = 0; k < num_cls_col; ++k) {
        for (int i = 0; i < num_lane_col; ++i) {
            int base = k * step_col_cls + i;
            int max_idx = 0;
            float max_val = loc_col[base];
            for (int g = 1; g < num_grid_col; ++g) {
                float v = loc_col[g * step_col_grid + base];
                if (v > max_val) {
                    max_val = v;
                    max_idx = g;
                }
            }
            max_indices_col[base] = max_idx;
            float exist_max_val = exist_col[base];
            int exist_max_idx = 0;
            float v1 = exist_col[chan_offset_col + base];
            if (v1 > exist_max_val) {
                exist_max_idx = 1;
            }
            valid_col[base] = exist_max_idx;
        }
    }

    // for(auto& item:max_indices_col){
    //     std::cout << item << ',';
    // }
    // std::cout <<std::endl;
    // for(auto& item:valid_col){
    //     std::cout << item <<',';
    // }
    // std::cout<<std::endl;

    std::vector<std::pair<int, int>> left_coords, right_coords;
    std::vector<int> lane_indices = {0, 1};

    for (int i : lane_indices) {
        int valid_count = 0;
        for (int k = 0; k < num_cls_row; ++k) {
            int base = k * step_row_cls + i;
            if (valid_row[base]) ++valid_count;
        }
        if (valid_count <= 5) continue;

        for (int k = 0; k < num_cls_row; ++k) {
            int base = k * step_row_cls + i;
            if (!valid_row[base]) continue;

            int center = max_indices_row[base];
            int start = std::max(0, center - local_width);
            int end = std::min(num_grid_row - 1, center + local_width) + 1;
            std::vector<int> indices(end - start + 1);
            std::iota(indices.begin(), indices.end(), start);

            
            std::vector<float> vals(indices.size());
            for (size_t t = 0; t < indices.size(); ++t) {
                int g = indices[t];
                vals[t] = loc_row[g * step_row_grid + base];
            }
            float pos_f = softmax_sum(vals, indices);
            float x = ((pos_f + 0.5f) / (num_grid_row - 1)) * img_w;
            float y = row_anchor[k] * img_h;
            // std::cout << "x:" << x << ",y:" << y <<std::endl;
            if (i == 0)
                left_coords.emplace_back((int)std::round(x), (int)std::round(y));
            else
                right_coords.emplace_back((int)std::round(x), (int)std::round(y));
        }
    }

    for (int i : lane_indices) {
        int valid_count = 0;
        for (int k = 0; k < num_cls_col; ++k) {
            int base = k * step_col_cls + i;
            if (valid_col[base]) ++valid_count;
        }
        if (valid_count <= 5) continue;

        for (int k = 0; k < num_cls_col; ++k) {
            int base = k * step_col_cls + i;
            if (!valid_col[base]) continue;

            int center = max_indices_col[base];
            int start = std::max(0, center - local_width);
            int end = std::min(num_grid_col - 1, center + local_width) + 1;
            std::vector<int> indices(end - start + 1);
            std::iota(indices.begin(), indices.end(), start);
            std::vector<float> vals(indices.size());
            for (size_t t = 0; t < indices.size(); ++t) {
                int g = indices[t];
                vals[t] = loc_col[g * step_col_grid + base];
            }
            float pos_f = softmax_sum(vals, indices);
            float y = ((pos_f + 0.5f) / (num_grid_col - 1)) * img_h;
            float x = col_anchor[k] * img_w;

            if (i == 0)
                left_coords.emplace_back((int)std::round(x), (int)std::round(y));
            else
                right_coords.emplace_back((int)std::round(x), (int)std::round(y));
        }
    }

    return {left_coords, right_coords};
}
