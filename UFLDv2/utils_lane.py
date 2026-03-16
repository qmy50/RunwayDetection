import numpy as np
import cv2

def draw_full_image_line(img, k, b, color=(0, 0, 255), thickness=2):
    """
    在图像上绘制直线 x = k*y + b。
    
    参数：
        img (numpy.ndarray): 要绘制的图像（会直接修改）
        k (float): 斜率 (dx/dy)
        b (float): 截距
        color (tuple): BGR颜色，默认为绿色 (0,255,0)
        thickness (int): 线条粗细，默认为2
    """
    h, w = img.shape[:2]          # 获取图像高度和宽度

    # 选择图像顶部 (y=0) 和底部 (y=h-1) 两个点
    y1 = 0
    y2 = h - 1
    x1 = int(k * y1 + b)          # 对应 x 坐标
    x2 = int(k * y2 + b)

    # 用 cv2.line 绘制直线，超出图像的部分会自动裁剪
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def ransac_polyfit(x_data, y_data, degree=1, max_trials=100, residual_threshold=30):
        best_coeffs = None  # 保存最优的多项式系数（最终车道曲线）
        best_inlier_count = 0  # 最优模型的内点数量

        # 数据量不足时，退化为普通拟合
        if len(x_data) < degree + 2:
            return np.polyfit(x_data, y_data, degree) if len(x_data) > degree else None

        n_samples = len(x_data)
        sample_size = degree + 1  # 拟合2次多项式需要至少3个点（degree=2 → 3个点）

        # 核心：多次随机抽样+拟合+验证，找到最优模型
        for _ in range(max_trials):  # 最多试100次
            # 步骤1：随机选少量样本（3个点）作为“候选内点”
            sample_indices = np.random.choice(n_samples, sample_size, replace=False)
            sample_x = x_data[sample_indices]
            sample_y = y_data[sample_indices]

            # 步骤2：用候选内点拟合一个临时多项式模型
            try:
                coeffs = np.polyfit(sample_y, sample_x, degree)  # 拟合y→x的2次多项式（车道曲线）

                # 步骤3：用临时模型验证所有点，区分“内点”和“外点”
                y_pred = np.polyval(coeffs, y_data)  # 用模型预测所有点的x值
                residuals = np.abs(x_data - y_pred)  # 计算真实x与预测x的误差（残差）
                inlier_mask = residuals < residual_threshold  # 误差<30像素 → 内点（真实车道点）
                inlier_count = np.sum(inlier_mask)  # 统计内点数量

                # 步骤4：保留内点最多的模型（最优模型）
                if inlier_count > best_inlier_count:
                    best_coeffs = coeffs
                    best_inlier_count = inlier_count
                    # 内点足够多时提前终止（加速）
                    if inlier_count > 0.7 * n_samples:
                        break
            except:
                continue
        # 最终：用所有内点重新拟合，得到更精准的模型
        if best_coeffs is not None and best_inlier_count > sample_size:
            y_pred = np.polyval(best_coeffs, y_data)
            residuals = np.abs(x_data - y_pred)
            inlier_mask = residuals < residual_threshold
            if np.sum(inlier_mask) > sample_size:
                try:
                    return np.polyfit(y_data[inlier_mask], x_data[inlier_mask], degree)
                except:
                    return best_coeffs 
                
        return best_coeffs