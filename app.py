import os
import re
import cv2
import numpy as np
import face_recognition
import urllib.parse
from datetime import datetime
import threading
import asyncio
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ✅ FastAPI Setup
app = FastAPI()

# ✅ Static Pages
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    return FileResponse(os.path.join("static", "index.html"))

@app.get("/register", response_class=HTMLResponse)
async def serve_register_page():
    return FileResponse(os.path.join("static", "register.html"))

@app.get("/detect", response_class=HTMLResponse)
async def serve_detect_page():
    return FileResponse(os.path.join("static", "detect.html"))

# ✅ Face Encodings
output_dir = "Known_faces"
os.makedirs(output_dir, exist_ok=True)
encoding_file = os.path.join(output_dir, "face_encodings.npy")

def load_encodings():
    if os.path.exists(encoding_file):
        try:
            data = np.load(encoding_file, allow_pickle=True).item()
            return list(data['encodings']), list(data['names']), list(data['regnos'])
        except Exception as e:
            print("⚠️ Failed to load encodings:", e)
    return [], [], []

def save_encodings():
    np.save(encoding_file, {
        "encodings": np.array(known_encodings, dtype=object),
        "names": np.array(known_names, dtype=object),
        "regnos": np.array(known_regnos, dtype=object)
    }, allow_pickle=True)

# ✅ Global Data
known_encodings, known_names, known_regnos = load_encodings()
recognized_today = set()

# ✅ Video Stream
class VideoStream:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            success, frame = self.cap.read()
            if success:
                self.frame = frame

    def get_frame(self):
        return self.frame

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

video_stream = VideoStream()

def mark_attendance(name, regno):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("attendance.csv", "a") as f:
        f.write(f"{regno},{name},{timestamp}\n")
    recognized_today.add(name)

# ✅ Frame Generator
def generate_frames():
    frame_skip = 2
    frame_count = 0

    while True:
        frame = video_stream.get_frame()
        if frame is None:
            continue

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            encoding = np.array(encoding, dtype=np.float32)
            if encoding.shape != (128,):
                continue

            name = "Unknown"
            regno = "000000000"

            if known_encodings:
                known_encodings_array = np.array(known_encodings, dtype=np.float32)
                distances = np.linalg.norm(known_encodings_array - encoding, axis=1)
                min_distance_index = np.argmin(distances)
                min_distance = distances[min_distance_index]

                if min_distance < 0.5:
                    name = known_names[min_distance_index]
                    if min_distance_index < len(known_regnos):
                        regno = known_regnos[min_distance_index]

                    if name not in recognized_today:
                        mark_attendance(name, regno)

            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ✅ Video Stream Endpoint
@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# ✅ Face Registration via WebSocket
@app.websocket("/register/{username}/{regno}")
async def register_face(websocket: WebSocket, username: str, regno: str):
    await websocket.accept()
    decoded_username = urllib.parse.unquote(username).strip()
    cleaned_username = re.sub(r'\s+', ' ', decoded_username)

    if not re.match(r"^\d{9}$", regno):
        await websocket.send_text("❌ Invalid registration number! Must be 9 digits.")
        await websocket.close()
        return

    if not re.match(r"^[a-zA-Z0-9 ]+$", cleaned_username) or "  " in cleaned_username:
        await websocket.send_text("❌ Invalid username! Use only letters, numbers, and single spaces.")
        await websocket.close()
        return

    global known_encodings, known_names, known_regnos
    if cleaned_username in known_names:
        indices = [i for i, name in enumerate(known_names) if name == cleaned_username]
        known_encodings = [enc for i, enc in enumerate(known_encodings) if i not in indices]
        known_names = [name for i, name in enumerate(known_names) if i not in indices]
        known_regnos = [r for i, r in enumerate(known_regnos) if i not in indices]
        await websocket.send_text(f"🔄 Old data for {cleaned_username} removed.")

    images_saved = 0

    while images_saved < 5:
        frame = video_stream.get_frame()
        if frame is None:
            await websocket.send_text("⚠️ Waiting for camera...")
            await asyncio.sleep(0.5)
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if not face_encodings:
            await websocket.send_text("📸 No face detected! Please stay still.")
            await asyncio.sleep(1)
            continue

        for encoding in face_encodings:
            filename = os.path.join(output_dir, f"{cleaned_username}_{regno}_{images_saved + 1}.jpg")
            cv2.imwrite(filename, frame)
            known_encodings.append(encoding.tolist())
            known_names.append(cleaned_username)
            known_regnos.append(regno)
            images_saved += 1
            await websocket.send_text(f"✅ Saved image {images_saved}/5")
            await asyncio.sleep(1)

        if images_saved >= 5:
            break

    save_encodings()
    await websocket.send_text("🎉 Face registration completed!")
    await websocket.close()

# ✅ Run App
if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    finally:
        video_stream.stop()