import cv2
import numpy as np


class DataUtils:
    @staticmethod
    def mask_to_yolo(mask):
        H, W = mask.shape
        yolo_boxes = []

        # Ensure binary
        mask = (mask > 0).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            bw = w / W
            bh = h / H

            yolo_boxes.append([0, cx, cy, bw, bh])

        return yolo_boxes
