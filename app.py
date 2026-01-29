import os
import cv2
import numpy as np
from flask import Blueprint, request, render_template, redirect, url_for, session
from keras.models import load_model
from ultralytics import YOLO

upload_bp = Blueprint("upload", __name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---- MODEL PLACEHOLDERS ----
meso = None
yolo_model = None

def load_models():
    global meso, yolo_model
    if meso is None:
        meso = load_model("meso_model.h5")
    if yolo_model is None:
        yolo_model = YOLO("yolov8n-face.pt")

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

@upload_bp.route("/upload")
def upload():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("upload.html")

@upload_bp.route("/predict", methods=["POST"])
def predict():
    if "user_id" not in session:
        return redirect(url_for("login"))

    file = request.files["file"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    label, score = predict_video(path)

    return render_template(
        "predict.html",
        label=label,
        score=f"{score:.4f}",
        file_url=f"/static/uploads/{file.filename}",
        is_video=True
    )
