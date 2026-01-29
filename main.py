from flask import Flask, Blueprint, request, render_template, url_for, redirect, session
from keras.models import load_model
import numpy as np
import cv2
import os
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = "render-secret"   

upload_bp = Blueprint("upload", __name__)

meso = load_model("meso_model.h5")
yolo_model = YOLO("yolov8n-face.pt")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.register_blueprint(upload_bp)


def load_models():
    global meso, yolo_model
    if meso is None or yolo_model is None:
        from keras.models import load_model
        from ultralytics import YOLO
        meso = load_model("meso_model.h5")
        yolo_model = YOLO("yolov8n-face.pt")
    return meso, yolo_model

def preprocess_face(face, target_size=(256, 256)):
    face = cv2.resize(face, target_size)
    face = face.astype("float32") / 255.0
    return np.expand_dims(face, axis=0)


def predict_video(video_path, meso_model, yolo_model=None, frame_skip=10):
    cap = cv2.VideoCapture(video_path)
    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_skip != 0:
            continue

        if yolo_model:
            results = yolo_model(frame)
            for result in results:
                for box in result.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    face = frame[y1:y2, x1:x2]
                    if face.size == 0:
                        continue
                    face_input = preprocess_face(face)
                    pred = meso_model.predict(face_input, verbose=0)[0][0]
                    predictions.append(pred)
        else:
            frame_input = preprocess_face(frame)
            pred = meso_model.predict(frame_input, verbose=0)[0][0]
            predictions.append(pred)

    cap.release()

    if predictions:
        avg_pred = np.mean(predictions)
        label = "Real" if avg_pred >= 0.5 else "DeepFake"
        return label, avg_pred

    return "No face detected", 0.0



@upload_bp.route("/upload", methods=["GET"])
def upload():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("upload.html")


@upload_bp.route("/predict", methods=["POST"])
def predict():
    if "user_id" not in session:
        return redirect(url_for("login"))

    file = request.files.get("file")
    if not file:
        return "No file uploaded."

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    ext = os.path.splitext(file.filename)[1].lower()
    file_url = f"/static/uploads/{file.filename}"

    meso_model, yolo = load_models()

    if ext in [".mp4", ".avi", ".mov", ".mkv"]:
        label, score = predict_video(path, meso_model, yolo)
        return render_template(
            "predict.html",
            label=label,
            score=f"{score:.4f}",
            file_url=file_url,
            is_video=True
        )
    else:
        import keras.preprocessing.image as kimg
        img = kimg.load_img(path, target_size=(256, 256))
        img_array = kimg.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = meso_model.predict(img_array)[0][0]
        label = "Real" if pred >= 0.5 else "DeepFake"

        return render_template(
            "predict.html",
            label=label,
            score=f"{pred:.4f}",
            file_url=file_url,
            is_video=False
        )


app.register_blueprint(upload_bp)

if __name__ == "__main__":
    app.secret_key = "dev"
    app.run(debug=True)
