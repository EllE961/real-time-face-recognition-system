# Face Recognizer

A simple PyQt5 application for managing face data and running real-time face recognition via webcam.

## Features
- **Face Manager Tab**  
  - Create/Delete person folders in `known_faces/`  
  - Add/Delete images (from disk)  
  - Capture new images from your webcam  
- **Recognition Tab**  
  - Reload face encodings  
  - Start/Stop live webcam recognition  
  - Adjustable skip frames, scale factor, and tolerance  
  - Smooth bounding boxes (no flicker on skipped frames)

## Demo
[![Face Recognizer Demo](https://img.youtube.com/vi/2KPsMDKF8c4/0.jpg)](https://youtube.com/shorts/2KPsMDKF8c4)
(Replace `VIDEO_ID` with your actual YouTube video ID.)

## Requirements
- **Python 3.7+**
- **Dependencies** (install via `pip install -r requirements.txt`):
  - PyQt5
  - face_recognition
  - opencv-python
  - numpy

## Setup
1. Clone or download this repository
2. Ensure `known_faces/` folder exists for storing images
3. `pip install -r requirements.txt`

## Usage
1. `python face_recognizer.py`
2. **Face Manager** tab:
   - Add a person (creates a subfolder)
   - Add images or capture via webcam
   - Delete person or images as needed
3. **Recognition** tab:
   - Click “Reload Encodings” if you changed faces
   - Start Recognition to open the webcam feed
   - Adjust skip frames, scale, and tolerance
   - Stop Recognition when finished

## Troubleshooting
- If the webcam fails to open, close other apps using the camera or check OS camera permissions
- If face recognition accuracy is low, add more images per person or adjust tolerance
- Ensure you have a C++ build environment if `face_recognition` or `dlib` fails to install

## Author
Yahya Al Salmi (2025)
