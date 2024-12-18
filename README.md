# Flask inference app

Allows for drag-and-drop classification of image and video files.

## Setup

* Place YOLO `model.pt` file in root project directory
* `brew install python@3.12`
* `python3.12 -m venv env`  # ultralytics doesn't support python 3.13
* `source env/bin/activate`
* `pip install flask opencv-python-headless ultralytics==8.3.51` # specifying latest version I've been training with
* `python app.py`
