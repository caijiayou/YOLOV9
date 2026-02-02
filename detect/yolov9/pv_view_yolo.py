#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import torch
import numpy as np
import requests
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device
from utils.plots import colors

# ----------------------------
# 設定 Server IP
# ----------------------------
RPI_IP = "10.1.1.36"          # 改成你的 Pi IP
url = f"http://{RPI_IP}:5000/video"

# ----------------------------
# 初始化 YOLOv9 模型
# ----------------------------
device = select_device('')  # CPU 或 GPU
weights = Path(r'C:\Users\ab881\OneDrive\桌面\YOLOV9\detect\yolov9\best.pt')
data = Path('datasets.yaml')

model = DetectMultiBackend(weights, device=device, data=data, fp16=False)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size((320, 320), s=stride)  # Client YOLO input

# ----------------------------
# 初始化串流
# ----------------------------
stream = requests.get(url, stream=True)
bytes_data = b''

frame_id = 0
last_pred = None

# ----------------------------
# 主迴圈：接收串流 + YOLO
# ----------------------------
for chunk in stream.iter_content(chunk_size=1024):
    bytes_data += chunk

    # 防止 buffer 爆掉
    if len(bytes_data) > 1_000_000:
        bytes_data = b''
        continue

    a = bytes_data.find(b'\xff\xd8')  # JPEG start
    b = bytes_data.find(b'\xff\xd9')  # JPEG end

    if a != -1 and b != -1:
        jpg = bytes_data[a:b+2]
        bytes_data = b''  # 只保留最新一張 frame

        frame = cv2.imdecode(
            np.frombuffer(jpg, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        if frame is None:
            continue

        # ----------------------------
        # YOLO 每 2 幀推論一次
        # ----------------------------
        if frame_id % 2 == 0:
            im = cv2.resize(frame, imgsz)
            im = im.transpose((2, 0, 1))[::-1]  # HWC -> CHW, BGR -> RGB
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im).to(device)
            im = im.float() / 255.0
            if len(im.shape) == 3:
                im = im[None]

            pred = model(im)
            pred = pred[0][1]
            pred = non_max_suppression(pred, 0.25, 0.45, max_det=50)
            last_pred = pred
        else:
            pred = last_pred

        # ----------------------------
        # 畫框
        # ----------------------------
        if pred is not None:
            for det in pred:
                if len(det):
                    det[:, :4] = scale_boxes(
                        im.shape[2:], det[:, :4], frame.shape
                    ).round()

                    for *xyxy, conf, cls in det:
                        x1, y1, x2, y2 = map(int, xyxy)
                        color = colors(int(cls), True)
                        label = f"{names[int(cls)]} {conf:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ----------------------------
        # 顯示畫面
        # ----------------------------
        cv2.imshow("Client YOLO Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

cv2.destroyAllWindows()
