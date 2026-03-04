import numpy as np

class YOLO_KF:
    def __init__(self,dt,R = [100, 20, 120, 20]):
        self.dt = dt
        self.Q = np.diag([1e-2, 1e-4, 1e-4, 1e-3, 1e-1, 1e-4, 1e-4, 1e-3]).astype(np.float32)
        self.R = np.diag(R).astype(np.float32)     
        self.H = np.array([[1,0,0,0,0,0,0,0],
                           [0,0,1,0,0,0,0,0],
                           [0,0,0,0,1,0,0,0],
                           [0,0,0,0,0,0,1,0]])
        self.A = np.array([[1, dt, 0,  0,  0,  0,  0,  0],
                            [0,  1, 0,  0,  0,  0,  0,  0],
                            [0,  0, 1, dt,  0,  0,  0,  0],
                            [0,  0, 0,  1,  0,  0,  0,  0],
                            [0,  0, 0,  0,  1, dt,  0,  0],
                            [0,  0, 0,  0,  0,  1,  0,  0],
                            [0,  0, 0,  0,  0,  0,  1, dt],
                            [0,  0, 0,  0,  0,  0,  0,  1]])
        self.x_hat = None
        self.p_hat = np.eye(8)
        self.initialized = False

    def predict(self):
        if not self.initialized:
            return
        self.x_hat = np.dot(self.A, self.x_hat)
        self.p_hat = self.A @ self.p_hat @ self.A.T + self.Q

    def update(self, z):
        if not self.initialized:
            # 初始化状态：位置分量为观测值，速度分量为0
            self.x_hat = np.array([[z[0,0]], [0], [z[1,0]], [0], [z[2,0]], [0], [z[3,0]], [0]])
            self.initialized = True
            # p_hat 保持为初始的 np.eye(8)（或可根据需要调整）
            cx = self.x_hat[0,0]
            cy = self.x_hat[2,0]
            w = self.x_hat[4,0]
            h = self.x_hat[6,0]
            return cx, cy, w, h
        # 正常卡尔曼更新
        temp = np.linalg.inv(self.H @ self.p_hat @ self.H.T + self.R)
        K = self.p_hat @ self.H.T @ temp
        self.x_hat = self.x_hat + K @ (z - self.H @ self.x_hat)
        self.p_hat = (np.eye(8) - K @ self.H) @ self.p_hat
        cx = self.x_hat[0,0]
        cy = self.x_hat[2,0]
        w = self.x_hat[4,0]
        h = self.x_hat[6,0]
        return cx, cy, w, h

class right_line_KF():
    def __init__(self,dt,R):
        self.dt = dt
        self.Q = np.diag([1e-1, 1,1e-1 ,1]).astype(np.float32)  
        self.R = np.diag(R).astype(np.float32)  
        self.H = np.array([[1, 0, 0, 0],[0, 0, 1, 0]])  
        self.A = np.array([[1, dt, 0, 0],
                           [0, 1,  0, 0],
                           [0, 0,  1,dt],
                           [0, 0,  0, 1]]) 
        # 状态向量 [k, k_dot, b, b_dot] 对应直线 x = k*y + b
#  np.array([[0.3817],[0],[378],[0]])  
        self.x_hat = None 
        self.p_hat = np.eye(4)  
        self.initialized = False

    def predict(self):
        if not self.initialized:
            return
        self.x_hat = np.dot(self.A, self.x_hat)
        self.p_hat = self.A @ self.p_hat @ self.A.T + self.Q

    def update(self, right_poly):
        # right_poly = [k, b]
        if not self.initialized:
            self.x_hat = np.array([[right_poly[0]], [0], [right_poly[1]], [0]])
            self.initialized = True
            k_right = self.x_hat[0, 0]
            b_right = self.x_hat[2, 0]
            return [k_right, b_right]
        z = np.array([[right_poly[0]], [right_poly[1]]])
        S = self.H @ self.p_hat @ self.H.T + self.R
        K = self.p_hat @ self.H.T @ np.linalg.inv(S)
        self.x_hat = self.x_hat + K @ (z - self.H @ self.x_hat)
        self.p_hat = (np.eye(4) - K @ self.H) @ self.p_hat
        right_k = self.x_hat[0, 0]
        right_b = self.x_hat[2, 0]
        return [right_k, right_b]

class left_line_KF():
    def __init__(self,dt,R):
        self.dt = dt
        self.Q = np.diag([1e-1, 1e-1,1e-1 ,1e-1]).astype(np.float32)  
        self.R = np.diag(R).astype(np.float32) 
        self.H = np.array([[1,0,0,0],[0,0,1,0]]) 
        self.A = np.array([[1,dt,0,0],
                           [0,1,0,0],
                           [0,0,1,dt],
                           [0,0,0,1]]) 
        # 原 y = -2.67*x +1618 转换为 x = (-1/2.67)*y + 1618/2.67 ≈ -0.3745*y + 606
        #np.array([[-0.3745],[0],[606],[0]]) 
        self.x_hat = None
        self.p_hat = np.eye(4)  
        self.initialized = False

    def predict(self):
        if not self.initialized:
            return
        self.x_hat = np.dot(self.A, self.x_hat)
        self.p_hat = self.A @ self.p_hat @ self.A.T + self.Q

    def update(self, left_poly):
        if not self.initialized:
            self.x_hat = np.array([[left_poly[0]], [0], [left_poly[1]], [0]])
            self.initialized = True
            k_left = self.x_hat[0, 0]
            b_left = self.x_hat[2, 0]
            return [k_left, b_left]
        z = np.array([[left_poly[0]], [left_poly[1]]])
        S = self.H @ self.p_hat @ self.H.T + self.R
        K = self.p_hat @ self.H.T @ np.linalg.inv(S)
        self.x_hat = self.x_hat + K @ (z - self.H @ self.x_hat)
        self.p_hat = (np.eye(4) - K @ self.H) @ self.p_hat
        left_k = self.x_hat[0, 0]
        left_b = self.x_hat[2, 0]
        return [left_k, left_b]

class HorizonKF():
    def __init__(self, dt,R):
        self.dt = dt
        self.Q = np.diag([1e-1, 1e-1, 1e-1, 1e-1]).astype(np.float32)
        self.R = np.diag(R).astype(np.float32)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0]])
        self.A = np.array([[1, dt, 0,  0],
                           [0,  1, 0,  0],
                           [0,  0, 1, dt],
                           [0,  0, 0,  1]])
        self.x_hat = None
        self.p_hat = np.eye(4)
        self.initialized = False

    def predict(self):
        if not self.initialized:
            return
        self.x_hat = np.dot(self.A, self.x_hat)
        self.p_hat = self.A @ self.p_hat @ self.A.T + self.Q

    def update(self, horizon_poly):
        """
        参数 horizon_poly: 长度为2的列表或数组 [k, b]，对应地平线直线 y = k*x + b
        """
        if not self.initialized:
            # 初始化：k 和 b 分量取观测值，速度分量为 0
            self.x_hat = np.array([[horizon_poly[0]], [0],
                                    [horizon_poly[1]], [0]])
            self.initialized = True
            k = self.x_hat[0, 0]
            b = self.x_hat[2, 0]
            return [k, b]

        z = np.array([[horizon_poly[0]], [horizon_poly[1]]])
        S = self.H @ self.p_hat @ self.H.T + self.R
        K = self.p_hat @ self.H.T @ np.linalg.inv(S)
        self.x_hat = self.x_hat + K @ (z - self.H @ self.x_hat)
        self.p_hat = (np.eye(4) - K @ self.H) @ self.p_hat
        k = self.x_hat[0, 0]
        b = self.x_hat[2, 0]
        return [k, b]