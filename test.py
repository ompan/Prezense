import cv2
import face_recognition
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import time
import hnswlib
import torch

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition")
        self.root.geometry("800x600")
        
        # Video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            tk.messagebox.showerror("Error", "Could not open webcam")
            self.root.destroy()
            return
        
        # Face recognition data
        self.face_data = {"encodings": np.array([]), "names": []}
        self.index = None
        self.load_face_data()
        
        # UI Elements
        self.video_label = tk.Label(self.root)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        self.running = True
        self.update_frame()

    def update_frame(self):
        if not self.running:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        start_time = time.time()
        face_count = self.process_faces(frame)
        processing_time = (time.time() - start_time) * 1000
        
        # Display frame
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.video_label.img = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=self.video_label.img)
        
        # Schedule next update
        self.root.after(10, self.update_frame())
    
    def process_faces(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to YUV and apply histogram equalization on the Y channel
        yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # Apply hist. eq. to Y channel
        equalized_rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

        # Resize for speed
        small_frame = cv2.resize(equalized_rgb, (0, 0), fx=0.5, fy=0.5)
        
        # Use CUDA if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor_frame = torch.tensor(small_frame).to(device)
        
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            
            if self.index is not None and len(self.face_data["encodings"]) > 0:
                labels, distances = self.index.knn_query(np.array([encoding]), k=1)
                if distances[0][0] < 0.6:  # Threshold for recognition
                    name = self.face_data["names"][labels[0][0]]
            
            # Scale back up the face locations
            top *= 2; right *= 2; bottom *= 2; left *= 2
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return len(face_locations)
    
    def load_face_data(self):
        try:
            data = np.load("Known_face/face_encodings.npy", allow_pickle=True).item()
            encodings = np.array(data.get("encodings", []))
            names = data.get("names", [])
            
            if len(encodings) > 0:
                self.index = hnswlib.Index(space='l2', dim=encodings.shape[1])
                self.index.init_index(max_elements=len(encodings), ef_construction=200, M=16)
                self.index.add_items(encodings, np.arange(len(names)))
                self.face_data["encodings"] = encodings
                self.face_data["names"] = names
                print(f"✅ Loaded {len(names)} face encodings.")
            else:
                self.index = None  # Ensure index is None if no encodings exist
                print("⚠️ No encodings found. Skipping face matching.")
        except Exception as e:
            print(f"❌ Error loading face data: {e}")
            self.index = None  # Prevent the app from breaking
    
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
