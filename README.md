# Mhacks-nichekasalaan

Webcam hand and finger detection using **MediaPipe Hand Landmarker** and **OpenCV**.

Point any USB webcam (including Logitech) at your hands to see landmarks, fingertip labels, and a rough raised-finger count in real time.

## Requirements

- Python **3.9–3.12** (3.10 or 3.11 recommended)
- A webcam (built-in or USB)
- macOS, Windows, or Linux

## Quick start

```bash
# 1. Clone and enter the repo
git clone https://github.com/umaraleem123/Mhacks-nichekasalaan.git
cd Mhacks-nichekasalaan

# 2. Create a virtual environment
python3 -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run (uses camera index 0 by default)
python hand_detection/webcam_hands.py
```

On first run the script downloads the MediaPipe `hand_landmarker.task` model (~8 MB) into `hand_detection/models/`.

Press **q** or **Esc** to quit.

## Options

```bash
python hand_detection/webcam_hands.py --camera 0      # device index
python hand_detection/webcam_hands.py --max-hands 2   # up to 2 hands
python hand_detection/webcam_hands.py --width 1280 --height 720
```

If the wrong camera opens (common with multiple devices), try `--camera 1`.

## Platform notes

### macOS

- Grant **Camera** permission to Terminal, iTerm, VS Code/Cursor, or whichever app launches Python (System Settings → Privacy & Security → Camera).
- Apple Silicon: use a normal `venv` + `pip`; MediaPipe provides arm64 wheels for supported Python versions.

### Windows

- Allow camera access for the terminal/Python if Windows prompts you.
- Prefer PowerShell or Command Prompt from the project folder after activating `.venv`.

### Linux

- Your user may need access to `/dev/video0` (often via the `video` group): `sudo usermod -aG video $USER` then re-login.
- A desktop session is required for the OpenCV preview window (`cv2.imshow`).

## What you should see

- Skeleton overlays for each detected hand (21 landmarks)
- Tip labels: Thumb, Index, Middle, Ring, Pinky
- Per-hand raised-finger count and a total at the top of the window

## Project layout

```
hand_detection/
  webcam_hands.py     # main demo
  models/             # model downloaded on first run (gitignored)
requirements.txt
README.md
```

## Troubleshooting

| Issue | Fix |
| --- | --- |
| `Could not open camera` | Unplug/replug webcam; try `--camera 1`; check OS camera permissions |
| Black window / no preview | Close other apps using the camera (Zoom, browser, etc.) |
| Model download failed | Check internet access; retry — or download manually from the URL printed in the script and pass `--model /path/to/hand_landmarker.task` |
| `mediapipe` install errors | Use Python 3.9–3.12 in a fresh venv |

## License

See the repository for license details. MediaPipe models are subject to Google’s MediaPipe terms.
