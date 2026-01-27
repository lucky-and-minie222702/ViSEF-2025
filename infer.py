from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import time
import torch
import os
import math

MODEL_PATH = "runs/detect/new_yolo26s/polyp_yolo/yolo_det/weights/last.pt"

IMG_SIZE = 960,
CONF_THRES = 0.9
IOU_THRES = 0.7
DEVICE = 0

model = YOLO(MODEL_PATH)

start = int(sys.argv[1])
end = int(sys.argv[2])

for i in range(start, end+1):
    INPUT_VIDEO = f"videos/{i}.mp4"
    OUTPUT_VIDEO = f"videos_pred/{i}.mp4"
    
    cap = cv2.VideoCapture(INPUT_VIDEO)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1.0 / fps
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    
    # Warm-up
    for _ in range(20):
        ret, frame = cap.read()
        if not ret:
            break
        r = model(frame, verbose=False)
        _ = r[0].plot()
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(total_frames), ncols = 100, desc = f"Video {i}"):
        ret, frame = cap.read()
        if not ret:
            break

        r = model.predict(
            source = frame,
            imgsz = IMG_SIZE,
            conf = CONF_THRES,
            iou = IOU_THRES,
            max_det = 5,
            workers = 2,
            verbose = False,
        )
        annotated = r[0].plot(
            labels = True,
            conf = True,
        )
        out.write(annotated)
    
    cap.release()
    out.release()

torch.cuda.empty_cache()