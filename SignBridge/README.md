# SignBridge

> **Status: Milestone 1 — hand tracking.** Live webcam hand detection works.
> Sign recognition, speech, and the UI are not built yet. See
> [Development Setup](#development-setup) and
> [Milestone 1 — Hand Tracking](#milestone-1--hand-tracking).

SignBridge is a bidirectional communication system that bridges American Sign
Language and spoken English, built as a 24-hour hackathon MVP. It is a
cross-platform desktop application targeting **macOS and Windows**.

## Project goal

Two directions of translation, each a self-contained pipeline:

**1. ASL → Voice**

A webcam observes the user signing. The system recognizes a supported ASL sign,
converts it to text, and speaks that text aloud.

```
webcam → sign recognition → text → spoken audio
```

**2. Voice → ASL**

A microphone captures speech. The system transcribes it, matches it against the
supported phrase set, and displays a visual ASL representation.

```
microphone → speech-to-text → supported phrase → visual ASL representation
```

## Planned architecture

The design priority is simplicity and modularity: each package owns one stage of
a pipeline and talks to the others through plain text. That keeps the two
directions independent, so they can be built and demoed separately, and it lets
us swap any single stage (a different recognizer, a different voice) without
touching the rest.

```
SignBridge/
├── app.py            # entry point that wires the pipelines together
├── src/
│   ├── vision/       # camera capture and ASL sign recognition
│   ├── speech/       # speech-to-text and text-to-speech
│   ├── asl/          # supported vocabulary, text → ASL representation
│   ├── ui/           # user-facing interface
│   └── hardware/     # optional Raspberry Pi / peripheral integration
├── data/             # datasets and recorded samples (not committed)
├── models/           # trained model artifacts (not committed)
└── assets/           # ASL reference images and other media
```

Text is the interchange format between every stage. `vision` emits text,
`speech` consumes and produces text, and `asl` maps text to a visual
representation.

## Scope

A fixed, small set of supported signs and phrases rather than open-vocabulary
translation. The MVP succeeds if both directions work end to end on that set.

## Not yet decided

These are deliberately open; nothing in the repository assumes an answer.

- The sign recognition approach and any trained model (MediaPipe is committed to
  only for hand tracking, which is a separate step from classifying a sign)
- The speech-to-text and text-to-speech providers
- The form of the visual ASL output (static images, animation, or avatar)
- The interface (local desktop window versus a web frontend)
- Whether the Raspberry Pi hardware path is in scope at all

## Cross-platform requirements

SignBridge must run on both macOS and Windows from a single codebase. These
rules apply to all application code:

- **One codebase, both platforms.** The same source must run unmodified on macOS
  and Windows. Never branch on the operating system unless there is no
  alternative, and isolate it in one place if there is.
- **No hardcoded OS-specific paths.** Nothing like `C:\...`, `/Users/...`,
  `/System/...`, or `/Applications/...` belongs in the source.
- **Use `pathlib` for every filesystem path.** Build paths by joining with `/`
  on `Path` objects rather than concatenating strings with separators, so the
  correct separator is chosen per platform.
- **Prefer platform-independent APIs.** Use the standard library (`pathlib`,
  `tempfile`, `os.environ`, `shutil`) instead of reimplementing behavior that
  differs per platform.
- **Avoid OS-dependent shell commands from Python.** No shelling out to `ls`,
  `dir`, `open`, `start`, or similar. Do the work in Python.
- **Webcam access goes through OpenCV.** Use `cv2.VideoCapture` rather than
  AVFoundation, DirectShow, or any other OS-specific camera API.
- **All dependencies live in `requirements.txt`,** so every developer installs
  the same environment.

Locate project files relative to the source file rather than the current working
directory, which varies with how the app is launched:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"
```

## Development Setup

Current milestone: **webcam access + hand tracking.** The only dependencies are
OpenCV and MediaPipe. Nothing is implemented yet; these steps only prepare the
environment.

Target: **Python 3.11 or 3.12** on macOS and Windows. The pinned
`mediapipe==0.10.21` publishes wheels for Python 3.9–3.12 and has none for
3.13, so do not use a newer interpreter.

Run every command from the `SignBridge/` directory.

### Why MediaPipe is pinned to 0.10.21

SignBridge uses the legacy `mp.solutions.hands` API, and the pin is not
cosmetic — two separate failures forced it on Apple Silicon:

- **mediapipe 0.10.35 and 1.x removed `mp.solutions` entirely.** Importing it
  raises `AttributeError: module 'mediapipe' has no attribute 'solutions'`.
- **The replacement Tasks API crashes on Apple Silicon.** `HandLandmarker`
  aborts the whole process inside Metal with
  `Check failed: service_ Service is unavailable` / `DrishtiMetalHelper`,
  in both IMAGE and VIDEO modes and with the CPU delegate.

0.10.21 is the newest release that still ships `mp.solutions.hands`, and it
runs reliably. Two knock-on constraints come with it:

- It requires **numpy < 2**, so OpenCV is held at 4.x (opencv-python 5.x
  requires numpy ≥ 2).
- It depends on **opencv-contrib-python**, so `requirements.txt` pins that
  package rather than `opencv-python`. Installing both puts two competing
  copies of `cv2` in the same environment.

Treat the versions in `requirements.txt` as load-bearing. Upgrading MediaPipe
means porting the tracker to the Tasks API and re-testing on Apple Silicon.

### macOS

Check Python:

```bash
python3 --version
```

This must report 3.11.x or 3.12.x. macOS ships an older system Python (often
3.9) as `python3`; if that is what you see, install 3.11 with
`brew install python@3.11` and use `python3.11` in place of `python3` in the
next command.

Create environment:

```bash
python3 -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Verify:

```bash
python -c "import cv2, mediapipe as mp; print('OpenCV', cv2.__version__, '| MediaPipe', mp.__version__, '| solutions:', hasattr(mp, 'solutions'))"
```

Expect MediaPipe 0.10.21 and `solutions: True`. If it prints `False`, the wrong
MediaPipe version is installed — see "Reinstalling" below.

Deactivate:

```bash
deactivate
```

### Windows PowerShell

Check Python:

```powershell
py --version
```

If 3.11 is not installed, get it from
[python.org/downloads](https://www.python.org/downloads/) and tick "Add
python.exe to PATH". The `-3.11` flag below selects it regardless of which
version `py` reports by default.

Create environment:

```powershell
py -3.11 -m venv .venv
```

Activate:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, allow it for the current session
only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Verify:

```powershell
python -c "import cv2, mediapipe as mp; print('OpenCV', cv2.__version__, '| MediaPipe', mp.__version__, '| solutions:', hasattr(mp, 'solutions'))"
```

Expect MediaPipe 0.10.21 and `solutions: True`.

Deactivate:

```powershell
deactivate
```

### Reinstalling when MediaPipe is already installed

If the environment already has a different MediaPipe (or OpenCV 5.x, which
pulls in numpy 2), installing on top of it can leave a mismatched set of
packages. The pinned versions span a numpy major version, so the cleanest fix
is to delete `.venv` and rebuild it with the steps above.

macOS:

```bash
deactivate
rm -rf .venv
```

Windows PowerShell:

```powershell
deactivate
Remove-Item -Recurse -Force .venv
```

To repair in place instead, remove the conflicting packages first so pip
resolves the pins cleanly:

```bash
python -m pip uninstall -y mediapipe opencv-python opencv-contrib-python
python -m pip install -r requirements.txt
```

Then rerun the verify command and confirm it reports MediaPipe 0.10.21 with
`solutions: True`.

### Notes

Once the environment is active, your prompt is prefixed with `(.venv)` and
`python` refers to the environment's interpreter on both platforms — which is
why every command above uses `python -m pip` rather than a bare `pip`.

The verify step only imports the two libraries. It does not open the webcam.

## Milestone 1 — Hand Tracking

### What this does

Opens the default webcam, detects up to two hands per frame with MediaPipe,
and draws the 21 landmarks and their connections over the live video. It also
labels each hand Left or Right. This milestone visualizes hands only; it does
**not** recognize any sign.

Two files make it work:

- `src/vision/hand_tracker.py` — the reusable `HandTracker` class. Knows nothing
  about the camera or the window, so ASL recognition can reuse it later.
- `app.py` — opens the camera, runs the capture loop, and draws the overlay.

No model file to download: the pinned MediaPipe bundles the hand model inside
the package.

### Activate the virtual environment

macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Run the application

From the `SignBridge/` directory, with the environment active.

macOS:

```bash
python3 app.py
```

Windows:

```powershell
python app.py
```

### Expected behavior

A window titled **SignBridge** opens showing your webcam, mirrored like a
selfie camera. Hold a hand up and a skeleton of dots and lines tracks it. The
top-left corner shows `SignBridge`, the number of hands detected, a
`Left hand` / `Right hand` line per hand with its confidence, and a reminder
that `q` quits.

Press **q** with the video window focused to exit. The camera light turns off
and the window closes. The first launch takes a few seconds longer while
MediaPipe loads its model.

### Webcam troubleshooting

**"could not open the default webcam"**

Grant camera permission. On macOS the permission belongs to the *terminal app*
you launched from, not to Python: System Settings → Privacy & Security →
Camera, then enable Terminal, iTerm, or VS Code. You must fully quit and reopen
that app for the change to apply. On Windows: Settings → Privacy & security →
Camera, and make sure "Let desktop apps access your camera" is on.

If permission is already granted, another application is probably holding the
camera. Only one app can use it at a time, so quit video calls, browser tabs
with camera access, and recording tools, then try again.

**Window opens but the video is black.** Usually a permission prompt that was
dismissed. Check the permission settings above, then relaunch.

**"lost the connection to the webcam".** The camera stopped returning frames,
typically because it was unplugged or taken over by another app. Reconnect and
rerun.

**Laggy video.** Close other camera and CPU-heavy apps.

**`AttributeError: module 'mediapipe' has no attribute 'solutions'`.** The
installed MediaPipe is newer than 0.10.21. See "Reinstalling" above.

**`Check failed: service_ Service is unavailable` / `DrishtiMetalHelper` /
`zsh: abort`.** Same cause: a newer MediaPipe using the Tasks API, which
crashes in Metal on Apple Silicon. Reinstall the pinned version.
