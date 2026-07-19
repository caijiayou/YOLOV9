import asyncio
import cv2
import requests
import queue
import threading
from aiortc import (RTCPeerConnection, RTCSessionDescription)

RPI_IP = "192.168.0.8"

frame_queue = queue.Queue(maxsize=1)

stop = False

def display():
    global stop
    while not stop:
        try:
            frame = frame_queue.get(
                timeout=1
            )
            cv2.imshow("WebRTC" ,frame)
            if cv2.waitKey(1)&0xff == ord("q"):
                stop=True
                break
        except:
            pass

async def start():
    global stop
    pc = RTCPeerConnection()
    pc.addTransceiver("video", direction="recvonly")
    @pc.on("track")
    async def on_track(track):
        print("Track:", track.kind)
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

    r = requests.post(f"http://{RPI_IP}:8080/offer",
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

    t = threading.Thread(target=display, daemon=True)
    t.start()

    while not stop:
        await asyncio.sleep(0.1)
    await pc.close()
    cv2.destroyAllWindows()

if __name__=="__main__":
    asyncio.run(start())