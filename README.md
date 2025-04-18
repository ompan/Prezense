# 👁️‍🗨️ Prezense

**Prezense** is a real-time facial recognition-based attendance management system designed to automate and streamline attendance tracking in classrooms, offices, or other organizational environments. It eliminates manual entry, reduces redundancy, and ensures secure and efficient attendance marking using live face detection.

## 🚀 Features

- 🔍 Real-time face recognition using `face_recognition` and OpenCV
- 🧠 FastAPI backend for lightweight performance
- 📸 Webcam-based face registration and attendance detection
- 💾 Persistent face encoding storage with `.npy` files
- ✅ Accurate matching with threshold and distance-based validation
- ⚡ Optimized performance with frame skipping and low-res detection
- 📊 Attendance logging to CSV files
- 🌐 Frontend served using static HTML (can be extended to mobile or React)

## 🛠️ Tech Stack

- Python 3.9+
- FastAPI
- face_recognition (dlib-based)
- OpenCV
- NumPy
- HTML/CSS (static frontend)
- Render (for cloud deployment)

## 🌐 Deployment (Render.com)

Prezense is deployable on [Render.com](https://render.com). Follow these steps:

1. Add `requirements.txt`, `start.sh`, and optionally `render.yaml` to your repo
2. Push the project to GitHub
3. Connect the repo to Render
4. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `./start.sh`
   - **Python Environment**
5. Deploy and test at your Render public URL

## 📸 Disclaimer

> Webcam-based features work best locally or on devices with camera access. For cloud deployment, consider integrating a mobile frontend (e.g., using Kotlin Multiplatform or React Native) to stream camera data.

---

Feel free to suggest enhancements or report issues!
