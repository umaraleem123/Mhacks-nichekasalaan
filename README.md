# Logitech webcam finger detection

Real-time finger detection for a Logitech (or any UVC) webcam using OpenCV and MediaPipe Hand Landmarker.

## What it does

- Opens your webcam feed
- Detects one or two hands
- Marks raised fingertips
- Shows which fingers are up and a per-hand count

## Setup

```bash
# Linux: MediaPipe needs EGL
# sudo apt install libegl1 libgl1

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_model.py
```

Quick check without a webcam:

```bash
python -m pytest -q
python detect_fingers.py --image path/to/hand.jpg
```

## Run with the webcam

```bash
python detect_fingers.py
```

Useful options:

```bash
python detect_fingers.py --list          # find camera indexes
python detect_fingers.py --camera 1      # use another USB camera
python detect_fingers.py --width 640 --height 480
python detect_fingers.py --no-mirror     # disable selfie mirror
```

Press `q` to quit.

## Test on a still image

If you are on a machine without a camera (or want a quick smoke test):

```bash
python detect_fingers.py --image path/to/hand.jpg
```

This writes `hand_fingers.jpg` next to the input and prints the raised-finger summary.

## Project layout

- `detect_fingers.py` — CLI entrypoint
- `finger_detection/detector.py` — MediaPipe wrapper + finger counting
- `models/hand_landmarker.task` — MediaPipe model (downloaded by the script above)
- `scripts/download_model.py` — model downloader

## Notes for Logitech cameras on Linux

Most Logitech webcams appear as `/dev/video0` (or the next free index). If the window is black or empty, try `--list` and pick another index. The app prefers the V4L2 backend, which is the usual path for USB webcams on Linux.
