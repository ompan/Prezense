import cv2
import face_recognition
import os
import time
import numpy as np

# Create directory for storing face images and encodings
output_dir = "Known_faces"
os.makedirs(output_dir, exist_ok=True)
encoding_file = os.path.join(output_dir, "face_encodings.npy")

# Load existing encodings if available
if os.path.exists(encoding_file):
    data = np.load(encoding_file, allow_pickle=True).item()
    known_encodings = list(data['encodings'])
    known_names = list(data['names'])
else:
    known_encodings = []
    known_names = []

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get user input
name = input("Enter your name: ").strip()
if not name:
    print("Error: Name cannot be empty.")
    cap.release()
    exit()

# Count existing images for the person
existing_images = len([f for f in os.listdir(output_dir) if f.startswith(name)])
if existing_images >= 5:
    print("Already stored 5 images for this person.")
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
        continue  # Skip frames to optimize processing

    # Convert to RGB and resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    
    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            # Scale face location back to original size
            top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2
            
            # Draw rectangle around face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Extract face region and check bounds
            face = frame[max(0, top):max(0, bottom), max(0, left):max(0, right)]
            if face.size > 0:
                filename = os.path.join(output_dir, "{}_{}.jpg".format(name, existing_images + 1))
                cv2.imwrite(filename, face)
                print("Saved:", filename)
                existing_images += 1

                # Store encoding if it's unique
                if not any(np.linalg.norm(enc - encoding) < 1e-4 for enc in known_encodings):
                    known_encodings.append(encoding)
                    known_names.append(name)

    # Display the frame
    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or existing_images >= 5:
        break

cap.release()
cv2.destroyAllWindows()

# Save updated encodings
if known_encodings:
    np.save(encoding_file, {'encodings': np.array(known_encodings), 'names': np.array(known_names)})
    print("Saved {} unique face encodings to {}".format(len(known_encodings), encoding_file))

print("Face capture ended.")
