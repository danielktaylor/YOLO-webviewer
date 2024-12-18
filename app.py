from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load YOLO model
#yolo_model = YOLO('yolov8n.pt')
yolo_model = YOLO('model.pt')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(image_path):
    image = cv2.imread(image_path)
    results = yolo_model(image)
    for result in results:
        boxes = result.boxes.xyxy.numpy()
        classes = result.boxes.cls.numpy()
        confidences = result.boxes.conf.numpy()
        for box, cls, conf in zip(boxes, classes, confidences):
            class_name = yolo_model.names[int(cls)]  # Get the class name
            print(f"Class detected: {class_name}, Confidence: {conf}")  # Log class and confidence
            x1, y1, x2, y2 = map(int, box[:4])
            color = (0, 255, 0) if class_name == "no_prey" else (0, 0, 255) if class_name == "prey" else (255, 255, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(image, label, (max(5,x1), max(15,y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    return output_path

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = 0.5  # Process every 0.5 seconds
    output_frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = int(fps * frame_rate)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            results = yolo_model(frame)
            for result in results:
                boxes = result.boxes.xyxy.numpy()
                classes = result.boxes.cls.numpy()
                confidences = result.boxes.conf.numpy()
                for box, cls, conf in zip(boxes, classes, confidences):
                    class_name = yolo_model.names[int(cls)]  # Get the class name
                    print(f"Class detected: {class_name}, Confidence: {conf}")  # Log class and confidence
                    x1, y1, x2, y2 = map(int, box[:4])
                    color = (0, 255, 0) if class_name == "no_prey" else (0, 0, 255) if class_name == "prey" else (255, 255, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(frame, label, (max(5,x1), max(15,y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            output_frame_path = os.path.join(app.config['OUTPUT_FOLDER'], f'frame_{frame_count}.jpg')
            cv2.imwrite(output_frame_path, frame)
            output_frames.append(output_frame_path)

        frame_count += 1

    cap.release()
    return output_frames

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        if filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}:
            output_path = process_image(file_path)
            return jsonify({'output': [output_path]}), 200
        elif filename.rsplit('.', 1)[1].lower() == 'mp4':
            output_frames = process_video(file_path)
            return jsonify({'output': output_frames}), 200

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
