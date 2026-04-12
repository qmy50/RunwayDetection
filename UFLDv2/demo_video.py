import torch, os, cv2
from utils.dist_utils import dist_print
import torch
from utils.common import merge_config, get_model
import tqdm
import torchvision.transforms as transforms
from PIL import Image
from utils_lane import draw_full_image_line, ransac_polyfit
import numpy as np

def pred2coords(pred, row_anchor, col_anchor, local_width = 1, original_image_width = 1640, original_image_height = 590):
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
    batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

    max_indices_row = pred['loc_row'].argmax(1).cpu()
    valid_row = pred['exist_row'].argmax(1).cpu()

    max_indices_col = pred['loc_col'].argmax(1).cpu()
    valid_col = pred['exist_col'].argmax(1).cpu()

    pred['loc_row'] = pred['loc_row'].cpu()
    pred['loc_col'] = pred['loc_col'].cpu()

    right_coords = []
    left_coords = []
    middle = original_image_width / 2
    coords = []

    row_lane_idx = [0,1]
    col_lane_idx = [0,1]

    for i in row_lane_idx:
        left_tmp = []
        right_tmp = []
        tmp = []
        if valid_row[0,:,i].sum() > num_cls_row / 4:
            for k in range(valid_row.shape[1]):
                if valid_row[0,k,i]:
                    all_ind = torch.tensor(list(range(max(0,max_indices_row[0,k,i] - local_width), min(num_grid_row-1, max_indices_row[0,k,i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_row'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row-1) * original_image_width
                    tmp = (int(out_tmp), int(row_anchor[k] * original_image_height))
                    if tmp[0] < middle:
                        left_tmp.append(tmp)
                    else:
                        right_tmp.append(tmp)
            left_coords.extend(left_tmp)
            right_coords.extend(right_tmp)

    for i in col_lane_idx:
        left_tmp = []
        right_tmp = []
        tmp = []
        if valid_col[0,:,i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0,k,i]:
                    all_ind = torch.tensor(list(range(max(0,max_indices_col[0,k,i] - local_width), min(num_grid_col-1, max_indices_col[0,k,i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_col'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_col-1) * original_image_height
                    tmp = (int(col_anchor[k] * original_image_width), int(out_tmp))
                    if tmp[0] < middle:
                        left_tmp.append(tmp)
                    else:
                        right_tmp.append(tmp)
            left_coords.extend(left_tmp)
            right_coords.extend(right_tmp)

    return left_coords, right_coords

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    # ===================== 视频配置项 =====================
    VIDEO_INPUT_PATH  = "test_video_2.mp4"    # 输入视频
    VIDEO_OUTPUT_PATH = "output_lane.mp4"    # 输出视频
    # ======================================================

    args, cfg = merge_config()
    cfg.batch_size = 1
    dist_print('start testing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
        img_w, img_h = 1640, 590
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
        img_w, img_h = 1920, 1080
    else:
        raise NotImplementedError

    # 加载模型
    net = get_model(cfg)
    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        compatible_state_dict[k[7:] if 'module.' in k else k] = v
    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval().cuda()

    # 图像预处理
    img_transforms = transforms.Compose([
        transforms.Resize((int(cfg.train_height / cfg.crop_ratio), cfg.train_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # ===================== 视频读取 =====================
    cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (img_w, img_h))
    # =====================================================

    # 限制窗口大小
    cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Lane Detection", 800, 450)

    print(f"视频总帧数：{total_frames}, FPS：{fps}")
    pbar = tqdm.tqdm(total=total_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 预处理
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_tensor = img_transforms(img_pil).unsqueeze(0).cuda()

        # 推理
        with torch.no_grad():
            pred = net(img_tensor)

        # 获取左右车道点
        left_coords, right_coords = pred2coords(
            pred, cfg.row_anchor, cfg.col_anchor,
            original_image_width=img_w,
            original_image_height=img_h
        )

        # 拟合直线
        vis = cv2.resize(frame, (img_w, img_h))
        if len(left_coords) > 3:
            left_coords = np.array(left_coords)
            left_line = ransac_polyfit(left_coords[:,0], left_coords[:,1], degree=1)
            draw_full_image_line(vis, left_line[0], left_line[1], color=(0,255,0), thickness=3)

        if len(right_coords) > 3:
            right_coords = np.array(right_coords)
            right_line = ransac_polyfit(right_coords[:,0], right_coords[:,1], degree=1)
            draw_full_image_line(vis, right_line[0], right_line[1], color=(0,255,0), thickness=3)

        # 显示 + 保存
        # cv2.imshow("Lane Detection", vis)
        out.write(vis)
        pbar.update(1)

        # 按Q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放
    pbar.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("视频处理完成！")

    # python demo_video.py configs/tusimple_res18.py 
