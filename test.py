import cv2
import face_recognition
import numpy as np
import os
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import threading
from sklearn.neighbors import KDTree

# Load known face encodings and names
encoding_file = "Known_faces/face_encodings.npy"

if os.path.exists(encoding_file):
    data = np.load(encoding_file, allow_pickle=True).item()
    known_encodings = np.array(data['encodings'])  # Convert to NumPy array for efficiency
    known_names = data['names']
    if known_encodings.size > 0:
        tree = KDTree(known_encodings)  # Use KDTree for efficient nearest neighbor search
else:
    known_encodings = np.array([])
    known_names = []
    tree = None

# Initialize webcam
cap = cv2.VideoCapture(0)

# Process every nth frame for efficiency
FRAME_PROCESSING_INTERVAL = 5
frame_count = 0

def recognize_faces(frame):
    """Detects and recognizes faces in the given frame."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)  # Resize for faster processing

    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    face_names = []
    for encoding in face_encodings:
        if tree is not None:
            distances, indices = tree.query([encoding], k=1)
            if distances[0][0] < 0.5:  # Set threshold for match
                name = known_names[indices[0][0]]
            else:
                name = "Unknown"
        else:
            name = "Unknown"
        face_names.append(name)

    # Scale back face locations
    face_locations = [(top * 4, right * 4, bottom * 4, left * 4) for top, right, bottom, left in face_locations]

    return face_locations, face_names

def update_frame():
    """Captures video frames and updates the UI."""
    global frame_count
    ret, frame = cap.read()
    if not ret:
        return

    frame_count += 1
    if frame_count % FRAME_PROCESSING_INTERVAL == 0:
        face_locations, face_names = recognize_faces(frame)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    lbl_video.imgtk = imgtk
    lbl_video.configure(image=imgtk)

    root.after(10, update_frame)  # Repeat

def start_video():
    """Runs video processing in a separate thread for UI responsiveness."""
    threading.Thread(target=update_frame, daemon=True).start()

def close_app():
    """Releases resources and closes the app."""
    cap.release()
    root.destroy()

# Initialize Tkinter UI
root = tk.Tk()
root.title("Optimized Face Recognition System")

lbl_video = Label(root)
lbl_video.pack()

btn_start = Button(root, text="Start", command=start_video, font=("Arial", 14), bg="green", fg="white")
btn_start.pack(pady=5)

btn_quit = Button(root, text="Exit", command=close_app, font=("Arial", 14), bg="red", fg="white")
btn_quit.pack(pady=10)

root.mainloop()
