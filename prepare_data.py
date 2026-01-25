import os
import cv2
import yaml
import shutil
import random
import numpy as np
from tqdm import tqdm
from glob import glob

RAW_ROOT = "dataset"
OUT_ROOT = "yolo_dataset"

TEST_NEGATIVE_COUNT = 1000
TRAIN_NEGATIVE_COUNT = 1000
VAL_NEGATIVE_COUNT = 600

TRAIN_RATIO = 0.9
MIN_AREA_RATIO = 0.005
MAX_AREA_RATIO = 0.35
BOX_TIGHTEN_RATIO = 1 

CLASS_ID = 0 

random.seed(22022009)


def ensure_dirs():
    for split in ["train", "val", "test"]:
        os.makedirs(f"{OUT_ROOT}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUT_ROOT}/labels/{split}", exist_ok=True)


def mask_to_bboxes(mask_path): 
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    h, w = mask.shape 
    
    _, bin_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY) 
    contours, _ = cv2.findContours( bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE ) 
    
    total_area = max(h, w) ** 2
    boxes = [] 
    
    for cnt in contours: 
        x, y, bw, bh = cv2.boundingRect(cnt) 
        
        area = bw * bh
        if area < total_area * MIN_AREA_RATIO or area > total_area * MAX_AREA_RATIO:
            continue

        cx = (x + bw / 2) / w 
        cy = (y + bh / 2) / h 

        bw /= w 
        bh /= h 
        
        bw *= BOX_TIGHTEN_RATIO**(1/2)
        bh *= BOX_TIGHTEN_RATIO**(1/2)
        
        boxes.append((CLASS_ID, cx, cy, bw, bh))

    return boxes, len(boxes) > 0

def write_label(label_path, boxes):
    with open(label_path, "w") as f:
        for b in boxes:
            f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")


def copy_sample(img_path, label_path, split):
    shutil.copy(img_path, f"{OUT_ROOT}/images/{split}/{os.path.basename(img_path)}")
    shutil.copy(label_path, f"{OUT_ROOT}/labels/{split}/{os.path.basename(label_path)}")


def process_positive_dataset(dataset_name, split_map):
    img_dir = f"{RAW_ROOT}/{dataset_name}/images"
    mask_dir = f"{RAW_ROOT}/{dataset_name}/masks"

    images = sorted(glob(f"{img_dir}/*"))

    for img_path in tqdm(images, desc=dataset_name):
        name = os.path.splitext(os.path.basename(img_path))
        basename = name[0]
        name = ''.join(name)
        mask_path = f"{mask_dir}/{basename}.jpg"
        mask_path = f"{mask_dir}/{basename}.png"
        
        if not os.path.exists(mask_path):
            continue

        boxes, qualified = mask_to_bboxes(mask_path)
        if not qualified:
            continue
        
        label_tmp = f"/tmp/{basename}.txt"
        write_label(label_tmp, boxes)

        split = split_map[dataset_name]
        copy_sample(img_path, label_tmp, split)

        os.remove(label_tmp)


def process_negative_images(image_paths, split):
    for img_path in tqdm(image_paths, desc=f"negative-{split}"):
        name = os.path.splitext(os.path.basename(img_path))
        basename = name[0]
        name = ''.join(name)

        label_tmp = f"/tmp/{basename}.txt"
        open(label_tmp, "w").close()  # empty label

        copy_sample(img_path, label_tmp, split)
        os.remove(label_tmp)


def main():
    ensure_dirs()

    # Test set
    split_map = {"BKAI": "test"}

    process_positive_dataset("BKAI", split_map)

    neg_images = glob(f"{RAW_ROOT}/polypgen_negative/images/*")
    neg_test = neg_images[:TEST_NEGATIVE_COUNT:]
    neg_remaining = neg_images[TEST_NEGATIVE_COUNT::]

    process_negative_images(neg_test, "test")

    # Train / Val
    positive_datasets = [
        "CVC-ClinicDB",
        "CVC-ColonDB",
        "Kvasir-SEG",
        "polypgen_positive",
    ]

    all_pos = []
    for d in positive_datasets:
        imgs = glob(f"{RAW_ROOT}/{d}/images/*")
        all_pos.extend([(d, p) for p in imgs])

    random.shuffle(all_pos)
    split_idx = int(len(all_pos) * TRAIN_RATIO)

    train_pos = all_pos[:split_idx]
    val_pos = all_pos[split_idx:]

    for split_name, subset in [("train", train_pos), ("val", val_pos)]:
        for dataset, img_path in tqdm(subset, desc=split_name):
            name = os.path.splitext(os.path.basename(img_path))
            basename = name[0]
            name = ''.join(name)
            mask_path = f"{RAW_ROOT}/{dataset}/masks/{name}"

            boxes, qualified = mask_to_bboxes(mask_path)
            if not qualified:
                continue

            label_tmp = f"/tmp/{basename}.txt"
            write_label(label_tmp, boxes)

            copy_sample(img_path, label_tmp, split_name)
            os.remove(label_tmp)

    process_negative_images(neg_remaining[:TRAIN_NEGATIVE_COUNT:], "train")
    process_negative_images(neg_remaining[TRAIN_NEGATIVE_COUNT:TRAIN_NEGATIVE_COUNT + VAL_NEGATIVE_COUNT:], "val")

    # data.yaml file
    data_yaml = {
        "path": OUT_ROOT,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "polyp"},
    }

    with open(f"{OUT_ROOT}/data.yaml", "w") as f:
        yaml.dump(data_yaml, f)

    print("Complete.")


if __name__ == "__main__":
    main()
