#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import torch
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device
from utils.plots import colors
import numpy as np

# 初始化模型（只做一次）
device = select_device('')  # CPU 或 GPU 自動選擇
weights = Path(r'C:\Users\ab881\OneDrive\桌面\YOLOV9\detect\yolov9\best.pt')
data = Path('datasets.yaml')
model = DetectMultiBackend(weights, device=device, data=data, fp16=False)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size((640, 640), s=stride)

# 打開攝影機
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

frame_id = 0
last_pred = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break
    
    # 每2幀推論一次
    if frame_id % 2 == 0:
        # 預處理圖片
        im = cv2.resize(frame, imgsz)
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(device)
        im = im.float() / 255.0
        if len(im.shape) == 3:
            im = im[None]
        
        # 模型推論
        pred = model(im)
        pred = pred[0][1]
        pred = non_max_suppression(pred, 0.25, 0.45, max_det=50)  # 降低 NMS 負擔
        last_pred = pred
    else:
        pred = last_pred  # 使用上一幀結果
    
    # 畫框（簡單 OpenCV）
    if pred is not None:
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = map(int, xyxy)
                    color = colors(int(cls), True)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f'{names[int(cls)]} {conf:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 顯示結果
    cv2.imshow('YOLOv9 Live Detection', frame)
    
    # 按 q 離開
    if cv2.waitKey(1) == ord('q'):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
