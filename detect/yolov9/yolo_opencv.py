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

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), stride=32, auto=True):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r, r

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = dw % stride, dh % stride

    dw /= 2
    dh /= 2

    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im

def main():
    device = select_device('cpu')
    weights = Path(r'C:\Users\ab881\OneDrive\桌面\YOLOV9\detect\yolov9\best.pt')
    data = Path('datasets.yaml')

    model = DetectMultiBackend(weights, device=device, data=data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((320, 320), s=stride)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("Camera read failed")
        return

    result_frame = prev_frame.copy()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        output = cv2.absdiff(prev_frame, frame)
        diff_score = output.mean()
        prev_frame = frame.copy()
        print(f"Diff score: {diff_score:.2f}")

        if diff_score > 10:
            print("YOLO辨識中...")

            im0 = frame.copy()
            im = letterbox(im0, imgsz, stride=stride, auto=True)
            im = im.transpose((2, 0, 1))[::-1]
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im).to(device)
            im = im.float() / 255.0
            if len(im.shape) == 3:
                im = im[None]

            try:
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    pred = model(im)
                    pred = pred[0][1]
                    pred = non_max_suppression(pred, 0.25, 0.45, max_det=1000)
            except RuntimeError as e:
                print(f"Inference error: {e}")
                continue

            annotator = Annotator(frame, line_width=2, example=str(names))
            for det in pred:
                if len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]} {conf:.2f}'
                        print('label:', label)
                        annotator.box_label(xyxy, label, color=colors(int(cls), True))

            result_frame = annotator.result()

        cv2.imshow('frame', cv2.resize(frame, (320, 240)))
        cv2.imshow('相減畫面', cv2.resize(output, (320, 240)))
        cv2.imshow('YOLOV9 Live Detection', cv2.resize(result_frame, (320, 240)))

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()