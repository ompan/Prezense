import cv2
import face_recognition
import os
import numpy as np

output_dir = "Known_faces"
os.makedirs(output_dir, exist_ok=True)
encoding_file = os.path.join(output_dir, "face_encodings.npy")

# Load existing face encodings
if os.path.exists(encoding_file):
    data = np.load(encoding_file, allow_pickle=True).item()
    known_encodings = list(data['encodings'])
    known_names = list(data['names'])
else:
    known_encodings = []
    known_names = []

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

name = input("Enter your name: ").strip().lower()
if not name:
    print("Error: Name cannot be empty.")
    cap.release()
    exit()

existing_images = len([f for f in os.listdir(output_dir) if f.startswith(name)])
if existing_images >= 5:
    print(f"Already stored 5 images for {name}.")
    cap.release()
    exit()

print("Press 'q' to quit or wait until 5 images are captured...")

FRAME_PROCESSING_INTERVAL = 5
frame_count = 0

while cap.isOpened() and existing_images < 5:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_PROCESSING_INTERVAL != 0:
        continue

    rgb_frame = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=0.5, fy=0.5), cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")

    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            face = frame[max(0, top):min(frame.shape[0], bottom), max(0, left):min(frame.shape[1], right)]

            if face.size > 0:
                filename = os.path.join(output_dir, f"{name}_{existing_images + 1}.jpg")
                cv2.imwrite(filename, face)
                print(f"Saved: {filename}")
                existing_images += 1

                if not any(np.linalg.norm(enc - encoding) < 1e-4 for enc in known_encodings):
                    known_encodings.append(encoding)
                    known_names.append(name)

    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or existing_images >= 5:
        break

cap.release()
cv2.destroyAllWindows()

if known_encodings:
    np.save(encoding_file, {'encodings': np.array(known_encodings), 'names': np.array(known_names)})
    print(f"Saved {len(known_encodings)} unique face encodings to {encoding_file}")

print("Face capture ended.")
