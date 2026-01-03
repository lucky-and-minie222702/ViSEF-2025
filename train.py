from ultralytics import YOLO
import sys

epoch_map =  {
    "s": 300,
    "m": 400,
    "l": 500,
}

variant = sys.argv[1]

def main(): 
    model = YOLO(f"yolo11{variant}.pt")  

    # Train
    model.train(
        data = "yolo_dataset/data.yaml",
        epochs = epoch_map[variant],
        imgsz = 640,
        batch = 32,
        nbs = 32,           
        pretrained = True,
        
        device = 0,
        box = 10.0,
        dfl = 2.0,
        cls = 0.3,

        optimizer = "Adam",
        lr0 = 0.001,
        lrf = 0.01,
        weight_decay = 0.0005,
        warmup_epochs = 3,

        hsv_h = 0.015,
        hsv_s = 0.8,
        hsv_v = 0.5,
        degrees = 180, 
        translate = 0.05,   
        scale = 0.7,
        perspective = 0.00001,
        flipud = 0.0, # no vertical flip
        fliplr = 0.5, # 50% horizontal flip
        copy_paste = 0.5,
        mosaic = 0.2,
        close_mosaic = 100,

        # Val
        val = True,
        save = True,
        save_period = 50,
        project = f"yolo11{variant}/polyp_yolo",
        name = "yolo_det",

        patience = 1_000_000,
        deterministic = True,
        seed = 27022009,
        
        save_conf = True,
    )

if __name__ == "__main__":
    main()
