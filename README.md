# Head Detection

## Overview
This repository uses **three detection methods** in Python:
- **MediaPipe** (recommended): fast and easy to run
- **RetinaFace**: strong accuracy but slow
- **YOLOv8**: fast and normal accuracy

Supported inputs:
- Images
- Videos

> If you want to run quickly, start with **MediaPipe**.

---

## Structure
```
├── detect_mp_image.py # MediaPipe - image face detection
├── detect_mp_video.py # MediaPipe - video face detection
├── detect_retinaface_image.py # RetinaFace - image face detection (optional)
├── detect_retinaface_video.py # RetinaFace - video face detection (optional)
├── detect_yolov8_image.py # YOLOv8 - image face detection (optional)
├── detect_yolov8_video.py # YOLOv8 - video face detection (optional)
└── README.md
```

---

## Quick Start (using mediapipe)
```bash
# 1. Create and activate conda environment
conda create -n hd_mp python=3.9 -y
conda activate hd_mp

# 2. Install dependencies
python -m pip install mediapipe==0.10.21

# 3. Check installation (0.10.21, has solutions: True)
python -c "import mediapipe as mp; print(mp.__version__); print('has solutions:', hasattr(mp,'solutions'))"

# 4. Run
python detect_mp_image.py

python detect_mp_video.py
```
