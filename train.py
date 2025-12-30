from ultralytics import YOLO

def main():
    model = YOLO("yolo11m.pt")  

    # Train
    model.train(
        data="yolo_dataset/data.yaml",
        epochs = 200,
        imgsz = 640,
        batch = 32,
        device = 0,
        workers = 2,
        pretrained = True,

        optimizer = "Adam",
        lr0 = 0.001,
        lrf = 0.01,
        weight_decay = 0.00025,
        warmup_epochs = 3,

        multi_scale = True,
        hsv_h = 0.005,
        hsv_s = 0.005,
        hsv_v = 0.005,
        degrees = 3.6, 
        translate = 0.05,
        scale = 0.75,
        perspective = 0.00001,
        flipud = 0.0, # no vertical flip
        fliplr = 0.5, # 50% horizontal flip
        mosaic = 0.3,

        # Val
        val = True,
        save = True,
        save_period = 10,
        project = "runs/polyp_yolo",
        name = "yolov8s_det",

        patience = 25,
        deterministic = True,
        seed = 27022009,
        
        cls = 0.0,
    )

    # Final evaluation
    metrics = model.val()
    print(metrics)

if __name__ == "__main__":
    main()
