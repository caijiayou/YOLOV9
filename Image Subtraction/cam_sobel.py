import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("無法開啟攝影機")
    exit()

threshold_value = 40  # 可自行調整

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 灰階
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Sobel X、Y
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # 梯度幅值
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # 轉成 uint8
    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # 二值化
    _, binary = cv2.threshold(
        magnitude,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    cv2.imshow("Original", frame)
    cv2.imshow("Sobel", magnitude)
    cv2.imshow("Binary Edge", binary)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()