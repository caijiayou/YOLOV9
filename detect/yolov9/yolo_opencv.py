#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import torch
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
import numpy as np

# 初始化模型（只做一次）
device = select_device('')
weights = Path(r'C:\Users\ab881\OneDrive\桌面\YOLOV9教材\YOLOV9\detect\yolov9\best.pt')
data = Path('datasets.yaml')
model = DetectMultiBackend(weights, device=device, data=data, fp16=False)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size((640, 640), s=stride)

# 打開攝影機
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

ret, result_frame = cap.read()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break
    
    ret1, frame1 = cap.read()
    output = cv2.subtract(frame, frame1)  # 相減
    print(output.mean())

    if output.mean()>2:
        print('YOLO辨識中...')
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
        pred = non_max_suppression(pred, 0.25, 0.45, max_det=1000)
        
        # 畫框與標籤
        annotator = Annotator(frame, line_width=2, example=str(names))
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    print('label: ', label)
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))
        
        # 顯示結果畫面
        result_frame = annotator.result()
    
    cv2.imshow('frame', frame)
    cv2.imshow('相減畫面', output)
    cv2.imshow('YOLOV9 Libe Detection', result_frame)
    
    # 按 q 離開
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()