from ultralytics import YOLO
model = YOLO(r"D:\yolo\runs\detect\train4\weights\best.pt")
model.export(
    format="onnx",
    simplify=True,    # 必须开
    opset=12,
    batch=1,         # 固定单帧，最快
    dynamic=False,    # 关闭动态尺寸
    imgsz=640
)

