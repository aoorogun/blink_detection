# dynamic-biometrics

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31013/)

## Steps
1. Optional: Create a Virtual Environment and activate. [venv](https://docs.python.org/3/library/venv.html) is an option
2. Install dependencies by running
```bash 
pip install -r requirements.txt
```
and ensure all the packages install correctly

3. Run the python ```blink_detection.py``` script with the arguments
```python python blink_detection.py -lm serialized_models/shape_predictor_68_face_landmarks.dat -sc 0```
n

4. press q to exit the camera window

5. usage:
```bash
usage: blink_detection.py [-h] -lm LANDMARK_MODEL [-sc SHOW_CONTOUR]

options:
  -h, --help            show this help message and exit
  -lm LANDMARK_MODEL, --landmark-model LANDMARK_MODEL
                        path to facial landmark predictor
  -sc SHOW_CONTOUR, --show-contour SHOW_CONTOUR
                        visualize eye landmark contour. 0 to not view, 1 to view
```