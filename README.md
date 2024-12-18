# Flask inference app

Drag-and-drop object detection UI for image and video files with custom YOLO models.

## Setup

* Place your YOLO `model.pt` file in root project directory
* `brew install python@3.12`
* `python3.12 -m venv env`  # ultralytics doesn't support python 3.13
* `source env/bin/activate`
* `pip install -r requirements.txt`
* `python app.py`
