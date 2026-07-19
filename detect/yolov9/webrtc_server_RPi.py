  GNU nano 8.4                                                                                                                                                                                                                                                                                                       webrtc_server.py *
import cv2
import asyncio
import threading
import queue
from aiohttp import web
from aiortc import (RTCPeerConnection, RTCSessionDescription, VideoStreamTrack)
from av import VideoFrame

pcs = set()
camera = None
# ==========================
# Camera Thread
# ==========================
class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.queue = queue.Queue(maxsize=1)
        self.running = True
        print("Camera:", self.cap.isOpened())
        self.thread = threading.Thread(target=self.capture, daemon=True)
        self.thread.start()

    def capture(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            # 永遠只保留最新一張
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except:
                    pass
            self.queue.put(frame)

    def get_frame(self):
        try:
            return self.queue.get(timeout=1)
        except:
            return None

    def stop(self):
        self.running = False
        self.cap.release()

# ==========================
# WebRTC Track
# ==========================
class CameraTrack(VideoStreamTrack):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera
    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = self.camera.get_frame()
        while frame is None:
            await asyncio.sleep(0.01)
            frame = self.camera.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video = VideoFrame.from_ndarray(frame, format="rgb24")
        video.pts = pts
        video.time_base = time_base
        return video

# ==========================
# WebRTC Offer
# ==========================
async def offer(request):
    global camera
    params = await request.json()
    offer = RTCSessionDescription(
        sdp=params["sdp"],
        type=params["type"]
    )
    pc = RTCPeerConnection()
    pcs.add(pc)
    print("New Client")

    if camera is None:
        camera = Camera()

    track = CameraTrack(camera)
    pc.addTrack(track)

    @pc.on("connectionstatechange")
    async def state():
        print(
            "State:",
            pc.connectionState
        )
        if pc.connectionState in [
            "failed",
            "closed"
        ]:
            pcs.discard(pc)
            await pc.close()

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.json_response({
        "sdp":
        pc.localDescription.sdp,
        "type":
        pc.localDescription.type
    })

# ==========================
# Shutdown
# ==========================
async def shutdown(app):
    global camera
    for pc in pcs:
        await pc.close()
    pcs.clear()
    if camera:
        camera.stop()

app = web.Application()
app.router.add_post("/offer", offer)
app.on_shutdown.append(shutdown)

web.run_app(
    app,
    host="0.0.0.0",
    port=8080
)
