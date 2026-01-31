from ultralytics import YOLO

MODEL_PATH = "runs/detect/yolo26n/polyp_yolo/yolo_det/weights/last.pt"

model = YOLO(MODEL_PATH)

path = model.export(
    format='engine', 
    device=0, 
    half=True, 
    dynamic=False
)

print(f"Success! TensorRT model saved at: {path}")
