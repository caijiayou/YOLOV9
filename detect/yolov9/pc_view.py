import requests
import cv2
import numpy as np

RPI_IP = "10.1.1.26"
url = f"http://{RPI_IP}:5000/video"

# 開啟串流
stream = requests.get(url, stream=True)
bytes_data = b''

for chunk in stream.iter_content(chunk_size=1024):
    bytes_data += chunk

    a = bytes_data.find(b'\xff\xd8')  # JPEG start
    b = bytes_data.find(b'\xff\xd9')  # JPEG end

    if a != -1 and b != -1:
        jpg = bytes_data[a:b+2]
        bytes_data = bytes_data[b+2:]

        img = cv2.imdecode(
            np.frombuffer(jpg, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        if img is not None:
            cv2.imshow("Raspberry Pi Stream", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
