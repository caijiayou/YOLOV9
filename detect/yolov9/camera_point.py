#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import torch
import numpy as np
from pathlib import Path

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device
from utils.plots import colors
from utils.augmentations import letterbox

# =========================
# 初始化模型（只做一次）
# =========================
device = select_device('')  # 自動選 CPU / GPU
weights = Path(r"C:\Users\ab881\OneDrive\桌面\練習程式碼\weights\face&phone\best.pt")
data = Path('datasets.yaml')

model = DetectMultiBackend(weights, device=device, data=data, fp16=False)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size((640, 640), s=stride)

# =========================
# 開啟攝影機
# =========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

frame_id = 0
last_pred = None
last_im_shape = None

# =========================
# 主迴圈
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    # 每 2 幀推論一次
    if frame_id % 2 == 0:

        # --- YOLO 標準 letterbox ---
        im, ratio, pad = letterbox(
            frame,
            imgsz,
            stride=stride,
            auto=True
        )

        im = im.transpose((2, 0, 1))[::-1]  # BGR → RGB, HWC → CHW
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(device)
        im = im.float() / 255.0

        if len(im.shape) == 3:
            im = im[None]

        # --- 推論 ---
        pred = model(im)
        pred = pred[0][1]
        pred = non_max_suppression(
            pred,
            conf_thres=0.25,
            iou_thres=0.45,
            max_det=50
        )

        last_pred = pred
        last_im_shape = im.shape

    else:
        pred = last_pred

    # =========================
    # 畫框
    # =========================
    if pred is not None:
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(
                    last_im_shape[2:],
                    det[:, :4],
                    frame.shape
                ).round()

                h, w = frame.shape[:2]

                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)

                    # 中心點
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # ---------- 左右判斷 ----------
                    margin = 40  # 中央容許誤差（px）
                    center_x = w // 2

                    if cx < center_x - margin:
                        side = "LEFT"
                        side_color = (255, 0, 0)   # 藍
                    elif cx > center_x + margin:
                        side = "RIGHT"
                        side_color = (0, 0, 255)   # 紅
                    else:
                        side = "CENTER"
                        side_color = (0, 255, 0)   # 綠

                    # ---------- 畫 YOLO 框 ----------
                    color = colors(int(cls), True)
                    label = f"{names[int(cls)]} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # ---------- 畫中心點 ----------
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

                    # ---------- 顯示左右提示 ----------
                    if side == "LEFT":
                        cv2.putText(frame, "<<LEFT",
                                    (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.2,
                                    side_color,
                                    3)

                    elif side == "RIGHT":
                        cv2.putText(frame, "RIGHT >>",
                                    (w - 220, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.2,
                                    side_color,
                                    3)

                    else:
                        cv2.putText(frame, "CENTER",
                                    (w // 2 - 80, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.2,
                                    side_color,
                                    3)



    # =========================
    # 顯示畫面
    # =========================
    cv2.imshow('YOLOv9 Live Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
