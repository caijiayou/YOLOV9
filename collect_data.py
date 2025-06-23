#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import os

# 開啟攝影機
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

name_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break
    
    cv2.imshow('frame', frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        name_count += 1
        img_name = f'img{name_count}.jpg'
        path = './dataset/images/' + img_name
        cv2.imwrite(path, frame)
        print(f'Saved: {path}')

cap.release()
cv2.destroyAllWindows()