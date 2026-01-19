from ultralytics import YOLO
import sys

version = sys.argv[1]
variant = sys.argv[2]


def main(): 
    model = YOLO(f"yolo{version}{variant}.pt")  

    # Train
    model.train(
        data = "yolo_dataset/data.yaml",
        epochs = 600,
        imgsz = 640,
        batch = 64,
        nbs = 64,           
        pretrained = True,
        
        device = 0,
        box = 10.0,
        dfl = 2.0,

        optimizer = "Adam",
        lr0 = 0.0009,
        lrf = 0.01,
        weight_decay = 0.0005,
        warmup_epochs = 3,

        hsv_h = 0.02,
        hsv_s = 0.75,
        hsv_v = 0.45,
        degrees = 1.8, 
        translate = 0.01,   
        scale = 0.2,
        perspective = 0.0001,   
        flipud = 0.5, # vertical flip
        fliplr = 0.5, # horizontal flip
        copy_paste = 0.4,
        mosaic = 0.3,
        close_mosaic = 100,

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
    )

if __name__ == "__main__":
    main()
