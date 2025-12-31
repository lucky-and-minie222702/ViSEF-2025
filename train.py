from ultralytics import YOLO

config = {
    "conf": 0.005,
}

def main():
    model = YOLO("yolo11x.pt")  

    # Train
    model.train(
        data="yolo_dataset/data.yaml",
        epochs = 100 + 1,
        imgsz = 1280,
        batch = -1,
        nbs = 64,           
        pretrained = True,
        
        # multiple GPUs
        device = [0, 1],

        optimizer = "Adam",
        lr0 = 0.001,
        lrf = 0.01,
        weight_decay = 0.00025,
        warmup_epochs = 3,
        dfl = 2,

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
        save_period = 20,
        project = "runs/polyp_yolo",
        name = "yolo_det",

        patience = 15,
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
