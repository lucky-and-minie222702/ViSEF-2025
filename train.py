from ultralytics import YOLO

def main():
    model = YOLO("models/yolov11s.pt")  

    # Train
    model.train(
        data="yolo_dataset/data.yaml",
        epochs = 100,
        imgsz = 640,
        batch = 16,
        device = 0,
        workers = 4,
        pretrained=True,

        optimizer="AdamW",
        lr0 = 1e-3,
        lrf = 1e-2,
        weight_decay=5e-4,

        hsv_h = 0.005,
        hsv_s = 0.005,
        hsv_v = 0.005,
        degrees = 3.6, 
        translate = 0.05,
        scale = 0.1,
        perspective = 0.0003,
        flipud = 0.0, # no vertical flip
        fliplr = 0.5, # 50% horizontal flip
        mosaic = 0.25,

        # Val
        val = True,
        save = True,
        save_period = 10,
        project = "runs/polyp_yolo",
        name = "yolov8s_det",

        patience = 20,
        deterministic = True,
        seed = 27022009,
    )

    # Final evaluation
    metrics = model.val()
    print(metrics)

if __name__ == "__main__":
    main()
