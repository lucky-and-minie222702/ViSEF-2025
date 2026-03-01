import os
import glob
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from tqdm import tqdm
import shutil
import yaml
import random
import cv2
import av
import torch
from rfdetr import RFDETRNano
import sys
import joblib

def blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score

CONF_THRES = 0.9
det_model = YOLO("det_best.pt")
cls_model  = YOLO("cls_best.pt")


start = int(sys.argv[1])
end = int(sys.argv[2])

polyp_frame_map = {}

for video_id in range(start, end+1):
    INPUT_VIDEO = f"videos/{video_id}.mp4"
    OUTPUT_VIDEO = f"videos_pred/{video_id}.mp4"
    
    cap = cv2.VideoCapture(INPUT_VIDEO)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1.0 / fps
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    
    # Warm-up
    for _ in range(20):
        r = det_model(np.zeros((640, 640, 3)), verbose=False)
        r[0].plot()
        r = cls_model(np.zeros((224, 224, 3)), verbose=False)
        r[0].plot()

    polyp_frame = []
    fr = 0    

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(range(total_frames), ncols = 100, desc = f"Video {video_id}")
    for _ in pbar:
        ret, frame = cap.read()
        if not ret:
            break

    
        det_result = det_model.predict(
            source=frame,
            imgsz=800,
            conf=CONF_THRES,
            verbose=False
        )[0]

        pred_boxes = det_result.boxes.xyxy.cpu().numpy()
        pred_confs = det_result.boxes.conf.cpu().numpy()

        valid_boxes = []
        last_valid_boxes = None
        last_valid_frame = -1
        persist_frames = 5
        
        annotated = frame

        if len(pred_boxes) > 0:
            valid_boxes = []
            valid_cls_confs = []

            if len(polyp_frame) >= 1:
                if polyp_frame[-1][0] == fr - 1:
                    for i, box in enumerate(pred_boxes):
                        x1, y1, x2, y2 = map(int, box)

                        crop = frame[
                            max(0, int(y1 * 0.9)) : min(height, int(y2 * 1.1)),
                            max(0, int(x1 * 0.9)) : min(width, int(x2 * 1.1))
                        ]

                        cls_result = cls_model.predict(
                            source=crop,
                            imgsz=96,
                            verbose=False
                        )[0]

                        cls_conf = cls_result.probs.top1conf.item()
                        cls_label = cls_result.probs.top1

                        if cls_label == 1 and cls_conf > 0.95:
                            valid_boxes.append(i)
                            valid_cls_confs.append(cls_conf)

                    if len(valid_boxes) > 0:
                        annotated = frame.copy()
                        current_boxes = []

                        for idx, cls_conf in zip(valid_boxes, valid_cls_confs):
                            x1, y1, x2, y2 = map(int, pred_boxes[idx])
                            det_conf = pred_confs[idx]

                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)

                            current_boxes.append((x1, y1, x2, y2, det_conf, cls_conf))

                        # save last valid detection
                        last_valid_boxes = current_boxes
                        last_valid_frame = fr

                        polyp_frame.append((fr, det_conf, cls_conf))

        # persistence logic
        elif last_valid_boxes is not None:
            # If within next 5 frames
            if fr - last_valid_frame <= persist_frames:
                annotated = frame.copy()

                for (x1, y1, x2, y2, det_conf, cls_conf) in last_valid_boxes:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        
        out.write(annotated)
        pbar.set_postfix(polyp_fr = len(polyp_frame))
            
    polyp_frame_map[i] = polyp_frame
    cap.release()
    out.release()

torch.cuda.empty_cache()
    
joblib.dump(polyp_frame_map, "polyp_frame_map.joblib")