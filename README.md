# Head Detection

## Overview

This repository contains several python codes using 3 different ways(mediapipe, retinaface, yolov8) to achieve face detection, it can detect images and videos.

- Recommend using mediapipe here

## mediapipe usage
```bash
# 1. Create and activate conda environment
conda create -n hd_mp python=3.9 -y
conda activate hd_mp
python -m pip install mediapipe==0.10.21

# 2. Check requirements (0.10.21, has solutions: True)
python -c "import mediapipe as mp; print(mp.__version__); print('has solutions:', hasattr(mp,'solutions'))"

# 3. Run
python detect_mp_image.py
python detect_mp_video.py
