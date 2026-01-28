from ultralytics import YOLO
import sys


version = sys.argv[1]
variant = sys.argv[2]

def main(): 
    model = YOLO(f"yolo{version}{variant}.pt") 

    # Train
    model.train(
        data = "yolo_dataset/data.yaml",
        epochs = 300,
        imgsz = 640,
        batch = 64,
        nbs = 64,           
        pretrained = True,
        
        device = 0,
        freeze = 4,

        optimizer = "MuSGD",
        lr0 = 0.01,
        warmup_epochs = 3,

        hsv_h = 0.006,
        hsv_s = 0.28,
        hsv_v = 0.16,

        degrees = 9, 
        translate = 0.1,   
        scale = 0.4,
        perspective = 0.0001,   
        flipud = 0.5, # vertical flip
        fliplr = 0.5, # horizontal flip
        copy_paste = 0.0,
        mosaic = 0.3,
        close_mosaic = 50,

        # Val
        val = True,
        save = True,
        save_period = 50,
        project = f"yolo{version}{variant}/polyp_yolo",
        name = "yolo_det",

        patience = 1_000_000,
        deterministic = True,
        seed = 27022009,
        
        save_conf = True,
        # resume = True,
    )

if __name__ == "__main__":
    main()
