import os
import numpy as np
import cv2
from flask import Blueprint, Flask, request, render_template, url_for, redirect, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from keras.models import load_model
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = "supersecretkey"

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


meso = None
yolo_model = None


def load_models():
    global meso, yolo_model
    if meso is None:
        meso = load_model("meso_model.h5")
    if yolo_model is None:
        yolo_model = YOLO("yolov8n-face.pt")


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)


def preprocess_face(face, target_size=(256, 256)):
    face = cv2.resize(face, target_size)
    face = face.astype("float32") / 255.0
    return np.expand_dims(face, axis=0)

def predict_video(video_path, frame_skip=10):
    load_models()
    cap = cv2.VideoCapture(video_path)
    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_skip != 0:
            continue

        results = yolo_model(frame)
        for result in results:
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                face_input = preprocess_face(face)
                pred = meso.predict(face_input, verbose=0)[0][0]
                predictions.append(pred)

    cap.release()
    if predictions:
        avg = np.mean(predictions)
        return ("Real" if avg >= 0.5 else "DeepFake", avg)

    return "No face detected", 0.0


@app.route("/")
def home():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", username=session.get("username"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

from main import upload_bp
app.register_blueprint(upload_bp)


if __name__ == "__main__":
    if not os.path.exists("users.db"):
        with app.app_context():
            db.create_all()

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
