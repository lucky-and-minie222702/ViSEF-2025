import os
import cv2
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm


PADDING = 0.2

def generate_stage2_data(model_path, dataset_root, output_root):
    model = YOLO(model_path)
    
    tasks = [
        {'src': 'train', 'dst': 'stage2_train'},
        {'src': 'remain', 'dst': 'stage2_train'},
        {'src': 'val', 'dst': 'stage2_val'}
    ]

    for task in tasks:
        img_dir = Path(dataset_root) / 'images' / task['src']
        
        img_files = list(img_dir.glob('*.*'))
        results = model.predict(source=str(img_dir), conf=0.5, save=False, stream=True)

        for r in tqdm(results, desc = f"Processing {task['src']}", total = len(img_files), ncols = 100):
            img_name = Path(r.path).stem
            img_orig = r.orig_img
            h_orig, w_orig, _ = img_orig.shape

            for i, box in enumerate(r.boxes):
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                w_box = x2 - x1
                h_box = y2 - y1
                
                pad_w = int(w_box * PADDING)
                pad_h = int(h_box * PADDING)
                
                x1_safe = max(0, x1 - pad_w)
                y1_safe = max(0, y1 - pad_h)
                x2_safe = min(w_orig, x2 + pad_w)
                y2_safe = min(h_orig, y2 + pad_h)

                crop = img_orig[y1_safe:y2_safe, x1_safe:x2_safe]
                save_dir = Path(output_root) / task['dst'] / label
                save_dir.mkdir(parents=True, exist_ok=True)
                
                save_path = save_dir / f"{img_name}_crop{i}.jpg"
                cv2.imwrite(str(save_path), crop)

    print(f"Done! Stage 2 data is ready in: {output_root}")


generate_stage2_data(
    model_path = "runs/detect/yolo26n/polyp_yolo/yolo_det/weights/last.engine",
    dataset_root = 'yolo_dataset',
    output_root = 'stage2_data'
)