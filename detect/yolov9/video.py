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
weights = Path(r'C:\Users\ab881\OneDrive\桌面\練習程式碼\weights\crack\best.pt')
data = Path('datasets.yaml')
model = DetectMultiBackend(weights, device=device, data=data, fp16=False)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size((640, 640), s=stride)

# 打開影片
input_path = input("你的輸入影片路徑: ")
output_path = input("輸出影片路徑: ")

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print(f"無法打開影片: {input_path}")
    exit()

# 影片的寬高與 FPS
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 編碼
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  # 影片讀完

    frame_idx += 1
    # --- YOLOv9 預處理 ---
    im = cv2.resize(frame, imgsz)
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device)
    im = im.float() / 255.0
    if len(im.shape) == 3:
        im = im[None]

    # --- 模型推論 ---
    pred = model(im)
    pred = pred[0][1]
    pred = non_max_suppression(pred, 0.25, 0.45, max_det=1000)

    # --- 畫框與標籤 ---
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                color = colors(int(cls), True)
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])),
                              (int(xyxy[2]), int(xyxy[3])), color, 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 寫入影片
    out.write(frame)

    # 可選：每 10 幀打印進度
    if frame_idx % 10 == 0:
        print(f"處理第 {frame_idx} 幀")

# 釋放資源
cap.release()
out.release()
print(f"影片已輸出至 {output_path}")
