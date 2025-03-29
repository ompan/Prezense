import cv2
import face_recognition
import numpy as np
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from sklearn.neighbors import KDTree
import time

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            self.root.destroy()
            return

        self.encoding_file = "face_data.npy"
        self.known_encodings = np.array([])
        self.known_names = []
        self.tree = None
        self.load_known_faces()        
        self.setup_ui()
        self.frame_count = 0
        self.processing_times = []
        self.frame_skip = 5
        self.threshold = 0.5
        self.running = True
        self.update_frame()

    def setup_ui(self):
        self.video_label = tk.Label(self.root)
        self.video_label.pack(pady=10)
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        self.start_btn = tk.Button(btn_frame, text="Start", command=self.start, width=10)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = tk.Button(btn_frame, text="Stop", command=self.stop, width=10)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.exit_btn = tk.Button(btn_frame, text="Exit", command=self.close, width=10)
        self.exit_btn.pack(side=tk.LEFT, padx=5)
        self.status_label = tk.Label(self.root, text="Ready")
        self.status_label.pack(pady=5)
        self.stats_label = tk.Label(self.root, text="")
        self.stats_label.pack(pady=5)

    def load_known_faces(self):
        if os.path.exists(self.encoding_file):
            try:
                data = np.load(self.encoding_file, allow_pickle=True).item()
                self.known_encodings = np.array(data['encodings'])
                self.known_names = data['names']
                if len(self.known_encodings) > 0:
                    self.tree = KDTree(self.known_encodings)
                    self.update_status(f"Loaded {len(self.known_names)} faces")
            except Exception as e:
                messagebox.showwarning("Warning", f"Load error: {str(e)}")
                self.known_encodings = np.array([])

    def process_frame(self, frame):
        start = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (0, 0), fx=0.25, fy=0.25)
        locations = face_recognition.face_locations(small)
        encodings = face_recognition.face_encodings(small, locations)
        
        names = []
        for encoding in encodings:
            if self.tree and len(self.known_encodings) > 0:
                dist, idx = self.tree.query([encoding], k=1)
                name = self.known_names[idx[0][0]] if dist[0][0] < self.threshold else "Unknown"
            else:
                name = "Unknown"
            names.append(name)
        
        locations = [(t*4, r*4, b*4, l*4) for t, r, b, l in locations]
        self.processing_times.append(time.time() - start)
        return locations, names

    def update_frame(self):
        if not self.running: return
        ret, frame = self.cap.read()
        if not ret:
            self.update_status("Camera error")
            return
        
        self.frame_count += 1
        if self.frame_count % self.frame_skip == 0:
            locations, names = self.process_frame(frame)
            for (t, r, b, l), name in zip(locations, names):
                cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
                cv2.putText(frame, name, (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.video_label.img = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=self.video_label.img)
        if self.frame_count % 30 == 0 and self.processing_times:
            avg = sum(self.processing_times)/len(self.processing_times)
            self.stats_label.config(text=f"Avg: {avg*1000:.1f}ms | FPS: {1/avg:.1f}" if avg > 0 else "")
            self.processing_times = []
        self.root.after(10, self.update_frame)

    def start(self):
        self.running = True
        self.update_status("Running")
        self.update_frame()

    def stop(self):
        self.running = False
        self.update_status("Paused")

    def update_status(self, msg):
        self.status_label.config(text=msg)

    def close(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()