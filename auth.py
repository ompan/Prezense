from fastapi import FastAPI
from auth import login, decode_token, get_current_user
from face_recognition_module import register_face, video_feed

app = FastAPI()

app.post("/token")(login)
app.websocket("/register/{username}")(register_face)
app.get("/video_feed")(video_feed)
