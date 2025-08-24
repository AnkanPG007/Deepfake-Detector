from flask import Flask, request, render_template
from keras.models import load_model
import numpy as np
import cv2
import os
from ultralytics import YOLO

app = Flask(__name__)

# Load your pretrained MesoNet
meso = load_model("meso_model.h5")

# Optional: YOLOv8 face detector
yolo_model = YOLO("yolov8n-face.pt")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Preprocessing function (matches training)
def preprocess_face(face, target_size=(256, 256)):
    face = cv2.resize(face, target_size)
    face = face.astype("float32") / 255.0
    return np.expand_dims(face, axis=0)

# Video prediction function
def predict_video(video_path, meso_model, yolo_model=None, frame_skip=10):
    cap = cv2.VideoCapture(video_path)
    predictions = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to speed up
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_skip != 0:
            continue

        # Face detection
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
            # Use full frame if no YOLO
            frame_input = preprocess_face(frame)
            pred = meso_model.predict(frame_input, verbose=0)[0][0]
            predictions.append(pred)

    cap.release()
    if predictions:
        avg_pred = np.mean(predictions)
        label = "Real" if avg_pred >= 0.5 else "DeepFake"
        return label, avg_pred
    return "No face detected", 0.0

# Flask routes
@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)

            ext = os.path.splitext(file.filename)[1].lower()
            if ext in [".mp4", ".avi", ".mov", ".mkv"]:
                label, score = predict_video(path, meso, yolo_model)
                return f"""
                <h1>Prediction: {label}</h1>
                <p>Confidence (Real class): {score:.4f}</p>
                <video width="400" controls>
                    <source src='/static/uploads/{file.filename}' type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                """
            else:
                # Image prediction (existing pipeline)
                import keras.preprocessing.image as kimg
                img = kimg.load_img(path, target_size=(256, 256))
                img_array = kimg.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                pred = meso.predict(img_array)[0][0]
                label = "Real" if pred >= 0.5 else "DeepFake"
                return f"""
                <h1>Prediction: {label}</h1>
                <p>Confidence (Real class): {pred:.4f}</p>
                <img src='/static/uploads/{file.filename}' width='300'>
                """
    return """
    <h1>Upload Image or Video</h1>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit">
    </form>
    """

if __name__ == "__main__":
    app.run(debug=True)
