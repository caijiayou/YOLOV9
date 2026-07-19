#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
import cv2
import requests
import queue
import threading
import torch
import numpy as np
from pathlib import Path
from aiortc import (RTCPeerConnection, RTCSessionDescription)
from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, scale_boxes, check_img_size)
from utils.torch_utils import select_device
from utils.plots import (Annotator, colors)

# ==========================
# Raspberry Pi
# ==========================
RPI_IP = "192.168.0.8"

# ==========================
# YOLOv9 Setting
# ==========================
device = select_device('')

weights = Path(r"C:\Users\Administrator\Desktop\work\YOLOV9\detect\yolov9\best.pt")

data = Path('datasets.yaml')

model = DetectMultiBackend(weights, device=device, data=data, fp16=False)

stride = model.stride
names = model.names
pt = model.pt

imgsz = check_img_size((320,320), s=stride)

print("YOLOv9 Ready")

# ==========================
# Frame Queue
# ==========================
frame_queue = queue.Queue(maxsize=1)
stop = False

# ==========================
# YOLO Thread
# ==========================
def yolo_worker():
    global stop
    result_frame = None
    while not stop:
        try:
            frame = frame_queue.get(timeout=1)
        except:
            continue
        im0 = frame.copy()

        # ======================
        # YOLO preprocessing
        # ======================
        im = cv2.resize(im0, (320,320))
        im = im.transpose((2,0,1))[::-1]
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(device)
        im = im.float()/255.0
        if len(im.shape)==3:
            im = im[None]

        # ======================
        # inference
        # ======================
        with torch.no_grad():
            pred = model(im)
        pred = pred[0]
        pred = non_max_suppression(pred, 0.25, 0.45, max_det=1000)

        annotator = Annotator(im0, line_width=2, example=str(names))

        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = (f"{names[int(cls)]}"f" {conf:.2f}")
                    print(label)
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))

        result_frame = annotator.result()
        cv2.imshow("YOLOv9 WebRTC", cv2.resize(result_frame, (640,480)))
        if cv2.waitKey(1)&0xff == ord('q'):
            stop=True
            break

# ==========================
# WebRTC
# ==========================
async def start():
    global stop
    pc = RTCPeerConnection()
    pc.addTransceiver("video", direction="recvonly")
    @pc.on("track")
    async def on_track(track):
        print("Receive:", track.kind)
        if track.kind=="video":
            while not stop:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except:
                        pass
                frame_queue.put(img)

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    r = requests.post(
        f"http://{RPI_IP}:8080/offer",
        json={
            "sdp":
            pc.localDescription.sdp,
            "type":
            pc.localDescription.type
        }
    )

    answer = r.json()
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))

    print("WebRTC connected")
    t = threading.Thread(target=yolo_worker, daemon=True)
    t.start()

    while not stop:
        await asyncio.sleep(0.1)

    await pc.close()

    cv2.destroyAllWindows()

if __name__=="__main__":
    asyncio.run(start())