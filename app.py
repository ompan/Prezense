import os
import cv2
import numpy as np
import face_recognition
import threading
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

# Create storage directory
output_dir = "Known_faces"
os.makedirs(output_dir, exist_ok=True)
encoding_file = os.path.join(output_dir, "face_encodings.npy")

# Load or initialize known face encodings
if os.path.exists(encoding_file):
    data = np.load(encoding_file, allow_pickle=True).item()
    known_encodings = list(data.get('encodings', []))
    known_names = list(data.get('names', []))
else:
    known_encodings = []
    known_names = []

# Initialize webcam
cap = cv2.VideoCapture(0)
frame_skip = 2  # Process every 2nd frame
resize_factor = 0.5  # Reduce resolution by half
lock = threading.Lock()

@app.websocket("/register/{username}")
async def register_face(websocket: WebSocket, username: str):
    await websocket.accept()
    username = username.strip().lower()

    saved_images = len([f for f in os.listdir(output_dir) if f.startswith(username)])
    if saved_images >= 5:
        await websocket.send_text("Error: Already stored 5 images for this user.")
        returna

    frame_count = 0
    while saved_images < 5:
        ret, frame = cap.read()
        frame_count += 1
        if not ret or frame_count % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            filename = os.path.join(output_dir, f"{username}_{saved_images + 1}.jpg")
            cv2.imwrite(filename, frame)

            if not any(np.linalg.norm(enc - encoding) < 0.6 for enc in known_encodings):
                known_encodings.append(encoding)
                known_names.append(username)

            saved_images += 1
            await websocket.send_text(f"Saved image {saved_images}/5")

        await asyncio.sleep(1)

    np.save(encoding_file, {'encodings': np.array(known_encodings), 'names': np.array(known_names)})
    await websocket.send_text("Face registration completed!")


def generate_frames():
    frame_count = 0
    while True:
        ret, frame = cap.read()
        frame_count += 1
        if not ret or frame_count % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            if len(known_encodings) > 0:
                distances = np.linalg.norm(known_encodings - encoding, axis=1)
                min_distance = np.min(distances)
                if min_distance < 0.6:
                    name = known_names[np.argmin(distances)]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")