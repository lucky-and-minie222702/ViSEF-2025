from ultralytics import YOLO

config = {
    "conf": 0.2,
    "iou": 0.6,
}

def main():
    model = YOLO("yolo11x.pt")  

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

        # multi_scale = True,
        hsv_h = 0.01,
        hsv_s = 0.9,
        hsv_v = 0.6,
        degrees = 3.6, 
        translate = 0.05,
        scale = 0.75,
        perspective = 0.00001,
        flipud = 0.0, # no vertical flip
        fliplr = 0.5, # 50% horizontal flip
        mosaic = 0.5,
        copy_paste = 0.5,
        mixup = 0.5,

        # Val
        val = True,
        save = True,
        save_period = 10,
        project = "runs/polyp_yolo",
        name = "yolo_det",

        patience = 25,
        deterministic = True,
        seed = 27022009,
        
        dfl = 2,
        
        **config
    )

    # Final evaluation
    metrics = model.val(**config)
    print(metrics)

if __name__ == "__main__":
    main()
