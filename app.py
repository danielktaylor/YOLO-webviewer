from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import secrets
import string
import cv2
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load YOLO models
detect_model = YOLO('detect.pt')
classify_model = YOLO('classify.pt')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_random_str(length=16):
    characters = string.ascii_letters + string.digits
    return ''.join(secrets.choice(characters) for _ in range(length))

def pad_image_to_square(image):
    h, w = image.shape[:2]
    if h == w:
        return image
    
    size = max(h, w)
    delta_w = size - w
    delta_h = size - h

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]  # Black padding
    new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_image

def draw_boxes(result, image):
    boxes = result.boxes.xyxy.numpy()
    classes = result.boxes.cls.numpy()
    confidences = result.boxes.conf.numpy()
    for box, cls, conf in zip(boxes, classes, confidences):
        class_name = detect_model.names[int(cls)]  # Get the class name
        print(f"Class detected: {class_name}, Confidence: {conf}")  # Log class and confidence
        x1, y1, x2, y2 = map(int, box[:4])
        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {conf:.2f}"
        cv2.putText(image, label, (max(5,x1), max(15,y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def process_image_file(full_image_path):
    image = cv2.imread(full_image_path)
    results = detect_model(image)
    for result in results:
        class_label = None
        class_conf = None

        if result and result[0].boxes.xyxy.numel() != 0:  # Is a cat detected?
            int_tensor = result[0].boxes.xyxy.int()  # Cast to integers

            int_list = int_tensor.tolist()[0]  # Convert to list and access the inner list
            x1, y1, x2, y2 = int_list  # Unpack into four integers
            frame_cropped = image[int(y1):int(y2), int(x1):int(x2)]  # TODO save this and make sure this crops correctly
            frame_cropped_sq = pad_image_to_square(frame_cropped)

            classification = classify_model.predict(frame_cropped_sq)
            class_label = str(classification[0].names[classification[0].probs.top1])
            class_conf = float(classification[0].probs.top1conf)

        new_filename = generate_random_str() + os.path.splitext(full_image_path)[1]
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], new_filename)
        image = draw_boxes(result, image)
        cv2.imwrite(output_path, image)
        return jsonify({'output': [{'file': new_filename, 'label': class_label, 'confidence': class_conf}]})

def process_video_file(video_path):
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
            results = detect_model(frame)
            
            class_label = None
            class_conf = None

            if results[0].boxes.xyxy.numel() != 0: # Is a cat detected?
                int_tensor = results[0].boxes.xyxy.int()  # Cast to integers

                int_list = int_tensor.tolist()[0]  # Convert to list and access the inner list
                x1, y1, x2, y2 = int_list  # Unpack into four integers
                frame_cropped = frame[int(y1):int(y2), int(x1):int(x2)] # TODO save this and make sure this crops correctly
                frame_cropped_sq = pad_image_to_square(frame_cropped)

                classification = classify_model.predict(frame_cropped_sq)
                class_label = str(classification[0].names[classification[0].probs.top1])
                class_conf = float(classification[0].probs.top1conf)

            output_filename = os.path.join(app.config['OUTPUT_FOLDER'], f'frame_{frame_count}.jpg')
            for result in results:
                frame = draw_boxes(result, frame)
            cv2.imwrite(output_filename, frame)
            output_frames.append({'file': f'frame_{frame_count}.jpg', 'label': class_label, 'confidence': class_conf})

        frame_count += 1

    cap.release()
    return jsonify({'output': output_frames})

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
            return process_image_file(file_path), 200
        elif filename.rsplit('.', 1)[1].lower() == 'mp4':
            return process_video_file(file_path), 200

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
