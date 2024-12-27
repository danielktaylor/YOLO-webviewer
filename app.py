from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image
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

RED = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

# Load trained models
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

def draw_box(image, label, confidence, color, x1, y1, x2, y2):
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {confidence:.2f}"
    cv2.putText(image, text, (max(5,x1), max(15,y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def run_models(image_sq, classify_only=False):
    '''
    Process a single frame of an image or video file. 
    If classify_only is True, only classifictiona is done. Otherwise object detection is performed, then classification on each object.
    Returns a list of tuples, one per object: (detection_label, detection_confidence, x1, y1, x2, y2, predicted_class_label, class_confidence)
    '''
    if classify_only:
        classification = classify_model.predict(image_sq)
        label = str(classification[0].names[classification[0].probs.top1])
        confidence = float(classification[0].probs.top1conf)
        return [(None, None, None, None, None, None, label, confidence)]

    results = detect_model(image_sq)

    all_objects = []
    for result in results:
        for object in result:
            if object.boxes.xyxy.numel() != 0:  # Was anything detected?
                int_tensor = object.boxes.xyxy.int()
                int_list = int_tensor.tolist()[0]
                x1, y1, x2, y2 = int_list

                detection_confidence = float(object.boxes.conf)
                detection_label = object.names[int(object.boxes.cls)]
                image_cropped = image_sq[int(y1):int(y2), int(x1):int(x2)]
                image_cropped_sq = pad_image_to_square(image_cropped)

                classification = classify_model.predict(image_cropped_sq)
                cls_label = str(classification[0].names[classification[0].probs.top1])
                cls_confidence = float(classification[0].probs.top1conf)

                all_objects.append((detection_label, detection_confidence, x1, y1, x2, y2, cls_label, cls_confidence))
    return all_objects

def process_frame(image, objects):
    best_classification = None
    best_confidence = 0.0

    for object in objects:
        detection_label, detection_confidence, x1, y1, x2, y2, cls_label, cls_confidence = object
        if detection_label:
            color = GREEN if cls_label == 'no_prey' else RED if cls_label == 'prey' else WHITE
            image = draw_box(image, detection_label, detection_confidence, color, x1, y1, x2, y2)

            if detection_confidence > best_confidence:
                best_classification = cls_label
                best_confidence = cls_confidence
        else: # Didn't do object detection, only classification
            best_classification = cls_label
            best_confidence = cls_confidence
    
    return image, best_classification, best_confidence

def process_image_file(full_image_path, classify_only):
    image = cv2.imread(full_image_path)
    image = pad_image_to_square(image) # Does make a difference!
    objects = run_models(image, classify_only)
    image, classification, confidence = process_frame(image, objects)

    new_filename = generate_random_str() + os.path.splitext(full_image_path)[1]
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], new_filename)
    cv2.imwrite(output_path, image)
    return jsonify({'output': [{'file': new_filename, 'label': classification, 'confidence': confidence}]})

def process_video_file(video_path, classify_only):
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
            frame = pad_image_to_square(frame) # Does make a difference!
            objects = run_models(frame, classify_only)
            image, classification, confidence = process_frame(frame, objects)
            filename = f'frame_{frame_count}.jpg'
            output_filename = os.path.join(app.config['OUTPUT_FOLDER'], filename)
            cv2.imwrite(output_filename, image)
            output_frames.append({'file': filename, 'label': classification, 'confidence': confidence})

        frame_count += 1

    cap.release()
    return jsonify({'output': output_frames})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    classify_only = request.form.get('classify_only') == 'true'
    convert_to_greyscale = request.form.get('convert_to_greyscale') == 'true'

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        if filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}:

            if convert_to_greyscale:
                with Image.open(file_path) as img:
                    img = img.convert("RGB")
                    pixels = img.load()
                    for y in range(img.height):
                        for x in range(img.width):
                            r, g, b = pixels[x, y]
                            gray = int(0.299 * r + 0.587 * g + 0.114 * b)  # Standard greyscale formula
                            pixels[x, y] = (gray, gray, gray)  # Set RGB channels to the same value
                    img.save(file_path)

            return process_image_file(file_path, classify_only), 200
        elif filename.rsplit('.', 1)[1].lower() == 'mp4':
            return process_video_file(file_path, classify_only), 200

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
