import cv2
import numpy as np
from utils import draw_full_image_line,calculate_midline,mirror_line_by_vp_x,ransac_polyfit

class BirdViewConverter:
    def __init__(self,img,IPM_points):
        # self.img = cv2.imread(img_path)
        self.img = img
        src_h, src_w = self.img.shape[:2]
        dst_h, dst_w = src_h, src_w  # 鸟瞰图尺寸与原图一致

        # self.src_points = np.float32([
        #     (int(src_w*0.01), src_h-230),    # 左下（近点左）
        #     (int(src_w*0.78), src_h-230),    # 右下（近点右）
        # (int(src_w*0.48), int(src_h*0.53)),   # 右上（远点右）
        #     (int(src_w*0.35), int(src_h*0.53))    # 左上（远点左）

        self.src_points = np.float32([IPM_points[0], IPM_points[1], IPM_points[2], IPM_points[3]])
        self.dst_points = np.float32([
            (int(src_w*0.3), src_h-20),    # 左下
            (int(src_w*0.7), src_h-20),    # 右下
            (int(src_w*0.7), 200),          # 右上
            (int(src_w*0.3), 200)       # 左上
        ])
        self.H, _ = cv2.findHomography(self.src_points, self.dst_points, cv2.RANSAC, 5.0)

        self.birdview = cv2.warpPerspective(
            self.img,                # 输入图像
            self.H,                  # 单应性矩阵
            (dst_w, dst_h),     # 输出图像尺寸
            flags=cv2.INTER_CUBIC  # 插值方式（立方插值更清晰）
        )
        self.gray_birdview = cv2.cvtColor(self.birdview,cv2.COLOR_BGR2GRAY)
        # cv2.imshow("birdview",self.gray_birdview)
        self.left_lines = []
        self.right_lines = []
        self.left_lines_ori = []
        self.right_lines_ori = []
        self.bird_left_lines_x = []
        self.bird_left_lines_y = []
        self.bird_right_lines_x = []
        self.bird_right_lines_y = []
        self.H_inv = np.linalg.inv(self.H)

    def detect_vertical_lines_in_birdview(self):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))    #自适应直方图均衡化
        clahe_gray = clahe.apply(self.gray_birdview)
        blur = cv2.GaussianBlur(clahe_gray, (5, 5), 0)
        edges = cv2.Canny(blur, 10, 50)
        h, w = self.gray_birdview.shape[:2]

        center_x = w // 2  # 图像中心x坐标
        mask_width = int(w * 0.3)  # 10%宽度
        mask_left = center_x - mask_width // 2  # 掩码左边界
        mask_right = center_x + mask_width // 2  # 掩码右边界
        edges[:,mask_left:mask_right] = 0
        sobel_x = cv2.Sobel(blur, cv2.CV_64F, dx=1, dy=0, ksize=3)
        sobel_y = cv2.Sobel(blur, cv2.CV_64F, dx=0, dy=1, ksize=3)
        grad_dir = np.arctan2(sobel_y, sobel_x) * 180 / np.pi

        mask1 = (np.abs(grad_dir) <= 10)
        mask2 = (np.abs(grad_dir) >= 170)
        mask3 = (edges == 255)
        vertical_edges = np.zeros_like(edges, dtype=np.uint8)
        vertical_edges[(mask1 | mask2) & mask3] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        vertical_edges = cv2.morphologyEx(vertical_edges, cv2.MORPH_CLOSE, kernel)
        lines = cv2.HoughLinesP(
            vertical_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=10,
            minLineLength=5,
            maxLineGap=10
        )
        filtered_lines = lines
        if filtered_lines is not None:
            for line in filtered_lines:
                x1, y1, x2, y2 = line[0]
                centor_x = (x1 + x2)/2  # 笔误：center_x
                if centor_x < w/2:
                    self.left_lines.append(line[0])
                else:
                    self.right_lines.append(line[0])

    @staticmethod
    def bird2ori(bird_x, bird_y, H_inv):
        bird_point = np.array([[bird_x, bird_y]], dtype=np.float32)
        bird_point = np.expand_dims(bird_point, axis=0)
        ori_point = cv2.perspectiveTransform(bird_point, H_inv)
        ori_x = int(ori_point[0][0][0])
        ori_y = int(ori_point[0][0][1])
        return ori_x, ori_y
    
    @staticmethod
    def ori2bird(ori_x, ori_y, H):
        """
        原始图像坐标转换为鸟瞰图坐标
        :param ori_x: 原图x坐标
        :param ori_y: 原图y坐标
        :param H: 正向透视变换矩阵（不是逆矩阵）
        :return: 鸟瞰图的x、y坐标（整数）
        """
        # 步骤1：构造原图坐标点（格式和bird2ori保持一致）
        ori_point = np.array([[ori_x, ori_y]], dtype=np.float32)
        # 步骤2：扩展维度（满足cv2.perspectiveTransform的输入要求）
        ori_point = np.expand_dims(ori_point, axis=0)
        # 步骤3：执行透视变换（使用正向矩阵H）
        bird_point = cv2.perspectiveTransform(ori_point, H)
        # 步骤4：提取并转换为整数坐标
        bird_x = int(bird_point[0][0][0])
        bird_y = int(bird_point[0][0][1])
        return bird_x, bird_y

    def convert_bird_kb_to_ori(self, bird_k, bird_b):
        """
        将鸟瞰图的直线参数 (k, b → x=ky+b) 转换为原始空间的直线参数
        :param bird_k: 鸟瞰图直线斜率（x=ky+b）
        :param bird_b: 鸟瞰图直线截距
        :return: (ori_k, ori_b) 原始空间直线参数（x=ky+b），失败返回None
        """
        h_bird, w_bird = self.gray_birdview.shape[:2]
        
        # 步骤1：在鸟瞰图中取直线上的2个特征点（覆盖主要区域）
        y1_bird = 250          # 鸟瞰图上部点
        x1_bird = bird_k * y1_bird + bird_b
        y2_bird = h_bird - 70  # 鸟瞰图下部点
        x2_bird = bird_k * y2_bird + bird_b
        
        # 步骤2：裁剪点到鸟瞰图范围内
        x1_bird = np.clip(x1_bird, 0, w_bird-1)
        x2_bird = np.clip(x2_bird, 0, w_bird-1)
        y1_bird = np.clip(y1_bird, 0, h_bird-1)
        y2_bird = np.clip(y2_bird, 0, h_bird-1)
        
        x1_ori, y1_ori = self.bird2ori(x1_bird, y1_bird, self.H_inv)
        x2_ori, y2_ori = self.bird2ori(x2_bird, y2_bird, self.H_inv)
        
        # 步骤4：用原始空间的2个点拟合直线（x=ky+b）
        dy = y2_ori - y1_ori
        if abs(dy) < 1e-6:  # 水平线，避免除零
            return None
        
        ori_k = (x2_ori - x1_ori) / dy
        ori_b = x1_ori - ori_k * y1_ori
        draw_full_image_line(self.img, ori_k, ori_b)
        
        return ori_k, ori_b

    def birdview_to_origin(self):
        self.detect_vertical_lines_in_birdview()

        for line in self.left_lines:
            bx1,by1,bx2,by2 = line
            x1,y1 = self.bird2ori(bx1,by1,self.H_inv)
            x2,y2 = self.bird2ori(bx2,by2,self.H_inv)
            self.left_lines_ori.append([x1,y1,x2,y2])
            cv2.circle(self.img,(x1, y1),radius=3,color=(0,255,0), thickness=-1)
            cv2.circle(self.img,(x2, y2),radius=3,color=(0,255,0), thickness=-1)
            self.bird_left_lines_x.append([bx1,bx2])
            self.bird_left_lines_y.append([by1,by2])
        for line in self.right_lines:
            bx1,by1,bx2,by2 = line
            x1,y1 = self.bird2ori(bx1,by1,self.H_inv)
            x2,y2 = self.bird2ori(bx2,by2,self.H_inv)
            self.right_lines_ori.append([x1,y1,x2,y2])
            cv2.circle(self.img,(x1, y1),radius=3,color=(255,255,0), thickness=-1)
            cv2.circle(self.img,(x2, y2),radius=3,color=(255,255,0), thickness=-1)
            self.bird_right_lines_x.append([bx1,bx2])
            self.bird_right_lines_y.append([by1,by2])

        # cv2.imshow("Dot image",self.img)

    def draw_line(self, lines_x:list, lines_y:list):
        if len(lines_x) < 10 or len(lines_y) < 10:
            # print("直线点数据不足")
            return [], []   #
        # if len(vp) == 2: 
        #     lines_x.append(vp[0])
        #     lines_y.append(vp[1])
        
        coeff = ransac_polyfit(np.array(lines_x).flatten(), np.array(lines_y).flatten())
        if coeff is None:
            return [], []
        
        k_bird, b_bird = coeff  # x = k_bird * y + b_bird (鸟瞰图)
        # 转换为原图直线参数
        ori_k, ori_b = self.convert_bird_kb_to_ori(k_bird, b_bird)  # 此函数已正确返回 x = k*y + b
        
        # 返回鸟瞰图直线参数和原图直线参数（均为 x = k*y + b 形式）
        return (k_bird, b_bird), [(ori_k, ori_b)]


if __name__ == "__main__":
    img_path =r"D:\program\image01.png"
    img = cv2.imread(img_path)
    cv2.imshow("Origin Image",img)
    IPM_points = np.float32([
                (200, 504),    # 左下（近点左）
                (1100, 504),    # 右下（近点右）
                (690, 290),   # 右上（远点右）
                (530, 290)    # 左上（远点左）
            ])
    birdViewConverter = BirdViewConverter(img,IPM_points)
    birdViewConverter.birdview_to_origin()
    # 调用时处理返回值
    left_line = birdViewConverter.draw_line(birdViewConverter.bird_left_lines_x,birdViewConverter.bird_left_lines_y)
    right_line = birdViewConverter.draw_line(birdViewConverter.bird_right_lines_x,birdViewConverter.bird_right_lines_y)
    h,w = img.shape[:2]
    print(left_line)
    # middle_slope,middle_int,_ = calculate_midline(left_line[0][0],left_line[0][1],right_line[0][0],right_line[0][1],h)
    # draw_full_image_line(img, 1/middle_slope, -middle_int/middle_slope)
    # 显示最终绘制结果
    cv2.imshow("Lane Line Result", birdViewConverter.img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()