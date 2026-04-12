import numpy as np


# ---------------------- 1. 欧拉角转旋转矩阵 ----------------------
def euler_to_rot(alpha, beta, gamma):
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    R_x = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    R_y = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    R_z = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
    return R_z @ R_y @ R_x

# ---------------------- 2. 旋转矩阵转欧拉角 ----------------------
def rot_to_euler(R): 
    """
    旋转矩阵转欧拉角（Z-Y-X外旋，输出rpy角：alpha滚转、beta俯仰、gamma偏航）
    输入：R -> 3x3旋转矩阵,注意这里使用的是机体系到地面系
    输出：alpha, beta, gamma -> 弧度制
    """
    # 提取旋转矩阵元素（论文公式2-14~2-16定义）
    R11, R12, R13 = R[0,0], R[0,1], R[0,2]
    R21, R22, R23 = R[1,0], R[1,1], R[1,2]
    R31, R32, R33 = R[2,0], R[2,1], R[2,2]
    
    # 计算俯仰角beta（避免sqrt负数，用clip限制范围）
    beta = np.arctan2(-R31, np.sqrt(R11**2 + R12**2).clip(1e-8))  # 1e-8防止数值不稳定
    
    # 计算滚转角alpha和偏航角gamma（处理奇异值）
    cos_beta = np.cos(beta)
    if abs(cos_beta) > 1e-8:  # 非奇异情况
        alpha = np.arctan2(R32 / cos_beta, R33 / cos_beta)
        gamma = np.arctan2(R21 / cos_beta, R11 / cos_beta)
    else:  # 奇异情况（beta≈±90°），滚转角设0，仅解偏航角
        alpha = 0.0
        gamma = np.arctan2(-R12, R11)
    
    return alpha, beta, gamma

# ---------------------- 3. 相机系转机体系旋转矩阵 ----------------------
def cam_to_body_rot(R_c):
    R_cb = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    return R_cb @ R_c

# ---------------------- 4. 基于灭点的姿态估计 ----------------------
def vanishing_point_pose(p_inf, K):
    p_inf_hom = np.array([p_inf[0], p_inf[1], 1.0])
    r3 = np.linalg.inv(K) @ p_inf_hom
    r3 = r3 / np.linalg.norm(r3)
    beta = -np.arcsin(r3[1])
    gamma = np.arctan2(r3[0], r3[2])
    return beta, gamma

def calculate_pose_from_runway(W = 60, k1 = None, k2 = None, beta = 0, gama = 0):
    """
    论文公式4-21：基于跑道灭点的无人机高度+横向偏移解算
    输入：
        W: 跑道实际宽度 (米，论文固定参数)
        K: 相机内参矩阵 3x3 (标定得到，对应论文2.4.3节)
        u_m: 图像中跑道中线 像素X坐标
        v1: 图像中跑道左边线 像素Y坐标
        v2: 图像中跑道右边线 像素Y坐标
    输出：
        H: 无人机相对跑道高度 (米)
        delta_y: 无人机相对跑道中心横向偏移 (米，正=偏右，负=偏左)
    """
    H,delta_x = None,None
    H = (k1 * k1 *np.cos(beta)*W)/(k1 - k2)*np.cos(gama)
    delta_x = W / (2*(k1 - k2)) * (2*k1*k2*np.tan(gama)*np.sin(beta) + k1 + k2)
    return H, delta_x

# ---------------------- 示例验证 ----------------------
if __name__ == "__main__":
    # 示例1：欧拉角-旋转矩阵互转验证
    alpha, beta, gamma = 0.1, 0.2, 0.3
    R = euler_to_rot(alpha, beta, gamma)
    alpha_rec, beta_rec, gamma_rec = rot_to_euler(R)
    print("=== 欧拉角-旋转矩阵互转验证 ===")
    print(f"原始欧拉角：{[alpha, beta, gamma]}")
    print(f"恢复欧拉角：{[alpha_rec, beta_rec, gamma_rec]}")
    print(f"误差：{[alpha-alpha_rec, beta-beta_rec, gamma-gamma_rec]}\n")
    
    # 示例2：相机系转机体系验证
    R_c = euler_to_rot(0.05, 0.15, 0.25)
    R_b = cam_to_body_rot(R_c)
    print("=== 相机系转机体系验证 ===")
    print("相机系旋转矩阵：\n", R_c.round(4))
    print("机体系旋转矩阵：\n", R_b.round(4), "\n")
    
    # 示例3：灭点姿态估计验证
    K_true = np.array([[915.855, 0, 647.461], [0, 918.629, 379.504], [0, 0, 1]])
    p_inf = (640, 360)
    beta_vp, gamma_vp = vanishing_point_pose(p_inf, K_true)
    print("=== 灭点姿态估计验证 ===")
    print(f"灭点坐标：{p_inf}")
    print(f"俯仰角：{np.degrees(beta_vp):.2f}°，偏航角：{np.degrees(gamma_vp):.2f}°")

