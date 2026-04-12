import torch
from utils.common import merge_config, get_model

# ===================== 读取配置 =====================
args, cfg = merge_config()
cfg.batch_size = 1
cfg.use_attention = False   # 【关键】关闭EMA注意力，解决ONNX报错
cfg.aux_head = False        # 关闭辅助头

print('setting batch_size to 1 for demo generation')

# ===================== 模型初始化 =====================
assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

if cfg.dataset == 'CULane':
    cls_num_per_lane = 18
elif cfg.dataset == 'Tusimple':
    cls_num_per_lane = 56
else:
    raise NotImplementedError

model = get_model(cfg)

# ===================== 加载权重（自动过滤多余层） =====================
checkpoint = torch.load(cfg.test_model, map_location="cpu")
state_dict = checkpoint['model']

# 【核心】自动跳过 em_attention 等训练层，不报错！
model_dict = model.state_dict()
filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict}
model.load_state_dict(filtered_dict)

model.cpu()
model.eval()

# ===================== 导出 ONNX =====================
dummy_input = torch.randn(1, 3, 320, 800)
onnx_file = "model_best.onnx"

# ===================== 【最终正确】导出 =====================
torch.onnx.export(
    model,
    dummy_input,
    onnx_file,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=["loc_row", "loc_col", "exist_row", "exist_col"]
)

print(f"✅ ONNX 模型已保存至: {onnx_file}")

# ===================== 验证模型 =====================
import onnx
onnx_model = onnx.load(onnx_file)
onnx.checker.check_model(onnx_model)
print("✅ ONNX 模型检查通过！")

# python -m utils.convert_onnx configs/tusimple_res18.py