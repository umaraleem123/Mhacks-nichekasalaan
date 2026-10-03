# ASL Camera Recognition

Real-time **American Sign Language (ASL) alphabet** recognition from a webcam using OpenCV and MediaPipe Hand Landmarker.

## What it does

- Captures video from your webcam (Logitech / any UVC camera)
- Detects a hand and draws a landmark skeleton
- Classifies static ASL letters (A–Y; motion letters J/Z are not supported)
- Smooths predictions over a short window so the letter is stable
- Lets you spell words: **Space** appends the current letter, **Backspace** deletes, **c** clears

## Setup

```bash
# Linux: MediaPipe needs EGL / GL
# sudo apt install libegl1 libgl1

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_model.py
```

## Run with the webcam

```bash
python recognize_asl.py
```

Useful options:

```bash
python recognize_asl.py --list          # find camera indexes
python recognize_asl.py --camera 1      # use another USB camera
python recognize_asl.py --width 640 --height 480
python recognize_asl.py --no-mirror     # disable selfie mirror
```

Keys while the window is focused:

| Key        | Action                          |
|------------|---------------------------------|
| `q` / Esc  | Quit                            |
| `Space`    | Append the stable letter        |
| Backspace  | Delete last spelled character   |
| `c`        | Clear spelled text              |

## Still-image smoke test

No webcam? Pass a photo of a hand signing a letter:

```bash
python recognize_asl.py --image path/to/hand.jpg
```

Writes `hand_asl.jpg` next to the input and prints the top letter candidates.

## Tests

```bash
python -m pytest -q
```

Classifier unit tests use synthetic landmarks and do not need a camera.

## Project layout

- `recognize_asl.py` — CLI entrypoint (camera + still image)
- `asl_recognition/camera.py` — webcam open / list / read
- `asl_recognition/landmarks.py` — MediaPipe Hand Landmarker wrapper
- `asl_recognition/classifier.py` — rule-based ASL alphabet scores
- `asl_recognition/pipeline.py` — temporal smoothing + spelling buffer
- `asl_recognition/overlay.py` — skeleton + HUD drawing
- `models/hand_landmarker.task` — downloaded by `scripts/download_model.py`
- `tests/` — unit tests for letter heuristics

## Notes

- This is a **heuristic demo** for hackathon / learning use. Lighting, hand angle, and signer variation all affect accuracy.
- Fist-like letters (E/M/N/S/T) and oriented letters (G/H/P/Q) are the hardest for a landmark-only ruleset.
- Prefer a plain background and keep the hand roughly upright and fully in frame.
