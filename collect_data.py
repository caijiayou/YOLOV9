#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows 強烈建議加 CAP_DSHOW
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

save_dir = input("請輸入存檔資料夾路徑：")
os.makedirs(save_dir, exist_ok=True)

print("按 c 拍照儲存，按 q 離開")

count = 0

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("❌ 讀取影像失敗")
        continue

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('c'):
        count += 1
        filename = f"img{count}.jpg"
        save_path = os.path.join(save_dir, filename)

        print("嘗試存檔路徑：", save_path)
        print("影像 shape：", frame.shape)

        success = cv2.imwrite(save_path, frame)

        if not success:
            success, buf = cv2.imencode(".jpg", frame)
            if success:
                with open(save_path, "wb") as f:
                    f.write(buf)
                print(f"✅ Saved (encode): {save_path}")
            else:
                print("❌ 失敗")
        else:
            print(f"✅ Saved: {save_path}")

cap.release()
cv2.destroyAllWindows()
