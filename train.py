from ultralytics import YOLO
import sys

epoch_map =  {
    "s": 500,
    "m": 700,
    "l": 900,
}

variant = sys.argv[1]

def main(): 
    model = YOLO(f"yolo11{variant}.pt")  

    # Train
    model.train(
        data = "yolo_dataset/data.yaml",
        epochs = epoch_map[variant],
        imgsz = 640,
        batch = 64,
        nbs = 64,           
        pretrained = True,
        
        device = 0,
        box = 10.0,
        dfl = 2.0,

        optimizer = "Adam",
        lr0 = 0.00085,
        lrf = 0.01,
        weight_decay = 0.0005,
        warmup_epochs = 3,

        hsv_h = 0.02,
        hsv_s = 0.75,
        hsv_v = 0.45,
        degrees = 45, 
        translate = 0.05,   
        scale = 0.6,
        perspective = 0.0001,
        flipud = 0.4, # vertical flip
        fliplr = 0.4, # orizontal flip
        copy_paste = 0.4,
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
