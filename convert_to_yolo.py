import os
import cv2
import yaml
import shutil
import random
import numpy as np
from tqdm import tqdm
from glob import glob
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


RAW_ROOT = "dataset"
OUT_ROOT = "yolo_dataset"

TRAIN_NEGATIVE_COUNT = 2500
VAL_NEGATIVE_COUNT = 300
TEST_NEGATIVE_COUNT = 600

TRAIN_POSITIVE_COUNT = 1500
VAL_POSITIVE_COUNT = 1000
TEST_POSITIVE_COUNT = 1800


MIN_AREA_RATIO = 0.005
MAX_AREA_RATIO = 0.5    
BOX_TIGHTEN_RATIO = 1 

CLASS_ID = 0 

random.seed(27022009)

box_area = []

def ensure_dirs():
    for split in ["train", "val", "test"]:
        os.makedirs(f"{OUT_ROOT}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUT_ROOT}/labels/{split}", exist_ok=True)


def mask_to_bboxes(mask_path): 
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    h, w = mask.shape 
    
    _, bin_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY) 
    contours, _ = cv2.findContours( bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    
    total_area = max(h, w) ** 2
    boxes = [] 
    
    for cnt in contours: 
        x, y, bw, bh = cv2.boundingRect(cnt) 
        
        area = bw * bh
        if not area < total_area * MIN_AREA_RATIO:
            box_area.append(area / total_area)
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
    
pos_count = {
    "train": TRAIN_POSITIVE_COUNT,
    "val": VAL_POSITIVE_COUNT,
    "test": TEST_POSITIVE_COUNT,
}

neg_count = {
    "train": TRAIN_NEGATIVE_COUNT,
    "val": VAL_NEGATIVE_COUNT,
    "test": TEST_NEGATIVE_COUNT
}


def main():
    ensure_dirs()
    
    # POSITIVE

    positive_datasets = [
        "polypgen",
        "Kvasir-SEG",
        "PolypDB",
        "CVC-ColonDB",
        "CVC-ClinicDB"
        
    ]

    all_pos = []
    for d in positive_datasets:
        imgs = glob(f"{RAW_ROOT}/positive/{d}/images/*")
        all_pos.extend([(f"positive/{d}", p) for p in imgs])
        
    random.shuffle(all_pos)
    idx = 0
    train_pos = all_pos[idx: idx + TRAIN_POSITIVE_COUNT:]
    idx += TRAIN_POSITIVE_COUNT
    val_pos = all_pos[idx:idx + VAL_POSITIVE_COUNT:]
    idx += VAL_POSITIVE_COUNT
    test_pos = all_pos[idx:idx + TEST_POSITIVE_COUNT:]

    for split_name, subset in [("train", train_pos), ("val", val_pos), ("test", test_pos)]:
        for dataset, img_path in tqdm(subset, desc = f"{split_name}-positive"):
            name = os.path.splitext(os.path.basename(img_path))
            basename = name[0]
            name = ''.join(name)
            mask_path = f"{RAW_ROOT}/{dataset}/masks/{basename}.jpg"
        
            if not os.path.exists(mask_path):
                mask_path = f"{RAW_ROOT}/{dataset}/masks/{basename}.png"
        
            if not os.path.exists(mask_path):
                print(mask_path)
                exit()
                continue

            boxes, qualified = mask_to_bboxes(mask_path)
            if not qualified:
                pos_count[split_name] -= 1
                continue

            label_tmp = f"/tmp/{basename}.txt"
            write_label(label_tmp, boxes)

            copy_sample(img_path, label_tmp, split_name)
            os.remove(label_tmp)


    # NEGATIVE

    neg_images = glob(f"{RAW_ROOT}/negative/hyperkvasir/*")
    
    # hard negative
    df = pd.read_csv("dataset/negative_labels.csv")        
    train_include_id = df[df["Finding"] == "bbps-0-1"]["Video file"].to_list()
    train_include_id = set(train_include_id)
    train_include_neg = []
    other_neg = []
    for p in neg_images :
        if Path(p).stem in train_include_id:
            train_include_neg.append(p)
        else:
            other_neg.append(p)
    
    random.shuffle(other_neg)
    neg_images = train_include_neg + other_neg
    
    idx = 0
    train_neg = neg_images[idx: idx + TRAIN_NEGATIVE_COUNT:]
    idx += TRAIN_NEGATIVE_COUNT
    val_neg = neg_images[idx:idx + VAL_NEGATIVE_COUNT:]
    idx += VAL_NEGATIVE_COUNT
    test_neg = neg_images[idx:idx + TEST_NEGATIVE_COUNT:]
    
    # add polypgen negative (after filter)
    train_neg.extend(glob(f"{RAW_ROOT}/negative/polypgen/*"))
    neg_count["train"] = len(train_neg)

    for split_name, subset in [("train", train_neg), ("val", val_neg), ("test", test_neg)]:
        for img_path in tqdm(subset, desc=f"{split_name}-negative"):
            name = os.path.splitext(os.path.basename(img_path))
            basename = name[0]
            name = ''.join(name)

            label_tmp = f"/tmp/{basename}.txt"
            open(label_tmp, "w").close()  # empty label

            copy_sample(img_path, label_tmp, split_name)
            os.remove(label_tmp)


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

    print(f"Postive images: {pos_count}")
    print(f"Negative images: {neg_count}")
    
    plt.hist(box_area, bins = 30)
    plt.title("Box area distribution")
    plt.xlabel("Area (normalized)")
    plt.ylabel("Count")
    plt.savefig("box_area.png", dpi = 200)
    
    print("Complete.")


if __name__ == "__main__":
    main()
