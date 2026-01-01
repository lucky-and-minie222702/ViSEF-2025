from ultralytics import YOLO

config = {
    "conf": 0.1,
    "iou": 0.45 
}

def main():
    model = YOLO("yolo11s.pt")  

    # Train
    model.train(
        data = "yolo_dataset/data.yaml",
        epochs = 300 + 1,
        imgsz = 640,
        batch = 64,
        nbs = 64,           
        pretrained = True,
        
        device = [0, 1],
        box = 10.0,

        optimizer = "Adam",
        lr0 = 0.001,
        lrf = 0.01,
        weight_decay = 0.0005,
        warmup_epochs = 3,

        multi_scale = True,
        hsv_h = 0.015,
        hsv_s = 0.8,
        hsv_v = 0.5,
        degrees = 3.6, 
        translate = 0.05,   
        scale = 0.5,
        perspective = 0.00001,
        flipud = 0.0, # no vertical flip
        fliplr = 0.5, # 50% horizontal flip
        copy_paste = 0.5,
        mosaic = 0.2,
        close_mosaic = 10,

        # Val
        val = True,
        save = True,
        save_period = 30,
        project = "yolo11s/polyp_yolo",
        name = "yolo_det",

        patience = 1_000_000,
        deterministic = True,
        seed = 27022009,
        
        save_conf = True,
        
        
        **config
    )

    # Final evaluation
    metrics = model.val(**config)
    print(metrics)

if __name__ == "__main__":
    main()
