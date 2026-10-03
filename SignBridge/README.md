# SignBridge

> **Status: Milestone 3 — Gemini temporal ASL recognition.** Hand tracking
> works, and a runnable demo records a short clip on a key press and has Gemini
> interpret it. The single-frame classifier from Milestone 2 is superseded,
> because a still image cannot show the movement that defines a sign. Speech
> and the UI are not built. See [Development Setup](#development-setup),
> [Milestone 1 — Hand Tracking](#milestone-1--hand-tracking),
> [Milestone 2 — ASL Recognition](#milestone-2--asl-recognition),
> [Gemini ASL Recognition](#gemini-asl-recognition), and
> [Milestone 3 — Gemini Temporal ASL Recognition](#milestone-3--gemini-temporal-asl-recognition).

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

Opens the webcam, detects up to two hands per frame with MediaPipe, and draws
the 21 landmarks and their connections over the live video. It also labels each
hand Left or Right. This milestone visualizes hands only; it does **not**
recognize any sign.

Camera selection is automatic: a Logitech Brio is used when one is connected,
otherwise the default camera. The chosen device is printed at startup and shown
in the window. On Windows capture goes through DirectShow, which avoids a long
stall before the Brio starts delivering frames. To pick a device yourself, pass
`--camera <index>` to the collector or the live recognizer.

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

**"could not open a webcam"**

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

## Milestone 2 — ASL Recognition

Recognizes a small set of static ASL signs from the webcam. The pipeline is:

```
webcam -> HandTracker -> 21 landmarks -> features -> classifier -> sign + confidence
```

Starting vocabulary: **hello**, **yes**, **no**.

The three steps are collect, train, run. Commands are identical on macOS and
Windows, and all of them run from the `SignBridge/` directory with the virtual
environment active.

### Feature representation

Each hand becomes 63 numbers: the 21 landmarks as (x, y, z), normalized so the
classifier does not care where the hand is, how big it is, or which hand it is.
Raw pixel coordinates are never used. `src/asl/features.py` documents the three
steps (mirror left hands, move the wrist to the origin, divide by the distance
to the furthest landmark). Hand rotation is intentionally preserved, because
orientation is part of what distinguishes one sign from another.

### 1. Collect data

```
python -m src.asl.data_collector
```

Keyboard controls, also shown on screen:

| Key | Action |
| --- | --- |
| `1` | select **hello** |
| `2` | select **yes** |
| `3` | select **no** |
| `R` | start / stop recording |
| `Q` | quit |

Number keys follow the order of `SIGNS` in `src/asl/__init__.py`, so a fourth
sign becomes `4` automatically.

Pick a sign, press `R`, hold the sign while moving your hand around the frame
and varying distance and angle slightly, then press `R` again. Samples are only
recorded while **exactly one** hand is visible, since two hands in frame have
no stable order and would mix into the same class. The window shows the current
sign, counts per sign, how many hands are visible, and whether recording is on.

Useful options:

```
python -m src.asl.data_collector --target 300      # per-sign goal (default 200)
python -m src.asl.data_collector --interval 0.05   # seconds between samples
```

Recording pauses automatically at `--target`. Running the collector again
appends to the existing data rather than replacing it.

**Aim for roughly 200 samples per sign** for the first test, collected in two
or three short bursts with your hand repositioned between them. Variety matters
much more than volume: 200 samples of a hand frozen in one spot will score
highly at training time and still fail live.

### 2. Train

```
python -m src.asl.train
```

Loads every sample, splits 80/20 with a fixed seed, trains a random forest, and
prints accuracy, a per-class report, and a confusion matrix. Requires at least
two signs with at least 20 samples each, and says exactly what is missing
otherwise.

Options: `--test-size`, `--trees`, `--seed`, `--data-dir`, `--model-out`.

Treat a reported accuracy near 100% with suspicion if all your samples came
from one continuous recording; consecutive frames are nearly identical, so some
of them land in both the training and test halves.

### 3. Run live recognition

```
python -m src.asl.live_recognition
```

Shows the camera with landmarks drawn, plus the predicted sign, confidence, and
FPS. Press `Q` to quit. Predictions below the confidence threshold display as
`SIGN: UNKNOWN` instead of guessing.

```
python -m src.asl.live_recognition --threshold 0.8   # stricter (default 0.6)
```

### Where things are stored

Training data is written to `data/asl/<sign>/samples.csv`, one row per sample:
63 feature values and a label. No images are saved. The trained model goes to
`models/asl_classifier.pkl`, which also stores the class list and feature
length so the recognizer can detect a stale model.

Both `data/` and `models/` are gitignored, so samples and models stay local and
each person collects their own.

### Adding more signs later

Append the name to `SIGNS` in `src/asl/__init__.py`:

```python
SIGNS: list[str] = ["hello", "yes", "no", "thanks"]
```

That is the only code change. The collector gives it the next number key, the
trainer picks up the new folder on its own, and the recognizer reads its
classes from the model. Then collect samples for the new sign and retrain:

```
python -m src.asl.data_collector
python -m src.asl.train
```

Retraining rebuilds the model from everything in `data/asl/`, so existing signs
are kept. Expect accuracy to dip as the vocabulary grows and signs start to
resemble each other; signs that differ only by motion will need a different
approach, since this classifier sees one frame at a time.

> **Superseded by Milestone 3.** That last limitation is exactly why this
> approach was set aside. The single-frame classifier code is still here and
> still runs, but the hello/yes/no samples and the trained model were deleted,
> so you would need to collect data again before using it. See
> [Gemini ASL Recognition](#gemini-asl-recognition).

## Gemini ASL Recognition

> **Status: prototype, work in progress.** The temporal capture and the Gemini
> client exist and are unit tested, but nothing is wired into a running
> application yet. This does **not** provide ASL translation, and it recognizes
> only three signs.

### Architecture

```
continuous webcam
  -> short temporal capture (a 2-second window of frames)
  -> Gemini 3.8 Flash, via the Interactions API
  -> structured ASL result (sign, confidence, description)
  -> application
```

| | |
| --- | --- |
| Model | `gemini-3.8-flash` |
| API | Interactions API (`client.interactions.create`) |
| SDK | `google-genai` >= 2.28 |
| Credential | `GEMINI_API_KEY` environment variable (required) |
| Vocabulary | hello, yes, no, plus `unknown` |

The recognizer analyzes a **temporal sequence of frames**, not individual
frames. Earlier Gemini models and the older `generateContent` endpoint are no
longer used: `gemini-2.5-flash` returns `404 NOT_FOUND` for new API keys, which
is what prompted the move to 3.8 Flash and the Interactions API.

### Why a single frame is not enough

The earlier classifier treated a sign as one hand pose, which cannot work,
because handshape is only one of the five parameters that distinguish ASL
signs. The others are palm orientation, location in signing space, non-manual
signals, and **movement** — and movement is invisible in a still image.

Concretely: "yes" is a fist bobbing at the wrist, and "no" is two fingers
snapping down onto the thumb. Freeze either one at the wrong instant and you
get a fist. The handshape is not what separates them; the motion is. Worse, two
unrelated signs can pass through *identical* poses at different moments, so a
confident read of one frame can be confidently wrong.

That is why the Gemini path is deliberately built as:

```
frames over time -> Gemini -> sign
```

and never as `frame -> Gemini -> sign`. The recognizer enforces this: handing
`recognize_sequence` a bare image raises an error, and a sequence of fewer than
two frames returns `unknown` without spending an API call.

### How the temporal capture works

`src/asl/sequence_capture.py` holds a rolling, rate-limited buffer and is
completely independent of Gemini — it only collects frames.

A sequence is an ordered list of frames, oldest first, each tagged with the
seconds elapsed since the capture began. Ordering carries meaning here, so
nothing ever reorders it.

Feed every camera frame to `add_frame()`; three settings in `CaptureConfig`
keep memory and payload bounded:

| Setting | Default | Purpose |
| --- | --- | --- |
| `duration_seconds` | 2.0 | length of the captured window |
| `sample_fps` | 6.0 | thins the ~30 fps camera stream |
| `max_frames` | 24 | hard cap, enforced by a fixed-length deque |
| `max_frame_width` | 640 | downscales wide frames before storing |

With the defaults, a 2-second capture stores 12 frames rather than the ~60 the
camera produced. Near-identical frames add payload without adding information
about the movement. Every value lives in `CaptureConfig`, so change it in one
place:

```python
from src.asl.sequence_capture import CaptureConfig, SequenceCapture

capture = SequenceCapture(CaptureConfig(duration_seconds=3.0, sample_fps=8.0))
capture.start()
while not capture.is_complete():
    capture.add_frame(frame)          # once per camera frame
sequence = capture.sequence()
```

### Supported vocabulary

**hello, yes, no** — and `unknown`.

That is the whole vocabulary, taken from `SIGNS` in `src/asl/__init__.py`. The
prompt tells Gemini it may answer only with those, and the response schema
constrains it with an enum. If a reply still names anything else, the
recognizer converts it to `unknown` rather than passing an unsupported label
on. Answers below the confidence threshold also become `unknown`: declining is
treated as a correct outcome, not a failure.

### Setup

The work lives on the `Umar` branch:

```
git switch Umar
```

Install dependencies (adds `google-genai` to the existing set):

```
python -m pip install -r requirements.txt
```

Set your API key. Create one at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey).

macOS:

```bash
export GEMINI_API_KEY='your-key-here'
```

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY = 'your-key-here'
```

Both set the key for the current shell session only. To keep it across
sessions, copy `.env.example` to `.env` and fill it in:

```bash
cp .env.example .env          # macOS
Copy-Item .env.example .env   # Windows PowerShell
```

Note that nothing loads `.env` automatically yet — the recognizer reads the
environment variable. Either export it as above or add a loader later.

### Security

The API key is read from the `GEMINI_API_KEY` environment variable and nothing
else. It is never hardcoded, never written to a committed file, and never
printed. When a Gemini request fails, the error reports only the exception
*type*, because provider messages sometimes echo request details back.

`.gitignore` excludes `.env` and any `.env.*`, with an explicit exception for
`.env.example`, which is a template containing no real key. If a key is ever
committed by accident, treat it as leaked and revoke it immediately — rewriting
history is not enough, since the value may already be cached elsewhere.

Be aware of what leaves the machine: this sends webcam frames to Google's API.
Nothing is uploaded until `recognize_sequence` is called, and no frames are
written to disk.

### Current limitations

- Three signs only, and no claim beyond them.
- Untested against the real API — every automated test uses a stub client.
- No retries and no batching.
- Single-sign clips only; no continuous signing or sentence structure.

## Milestone 3 — Gemini Temporal ASL Recognition

The first runnable Gemini integration. You press SPACE, it records a short clip
of one sign, sends the whole clip to Gemini in one request, and shows the
result.

> **Prototype with a three-sign vocabulary.** This recognizes **hello**,
> **yes**, and **no**, and answers `unknown` for anything else. It is not ASL
> translation and does not understand ASL generally.

Uses **Gemini 3.8 Flash** through the **Interactions API**, and requires
`GEMINI_API_KEY` to be set.

### Why ASL is treated as a temporal sequence

A sign is a movement, not a pose. Handshape is only one of the five parameters
that distinguish ASL signs; the others are palm orientation, location in
signing space, non-manual signals, and **movement**, which no still image can
show.

"yes" is a fist bobbing at the wrist. "no" is two fingers snapping down onto
the thumb. Freeze either at the wrong instant and you have a fist — the
handshape does not separate them, the motion does. Two unrelated signs can also
pass through identical poses at different moments, so a confident reading of one
frame can be confidently wrong.

So the pipeline is built as `frames over time -> Gemini -> sign`, never
`frame -> Gemini -> sign`:

```
webcam -> SequenceCapture -> ordered frame sequence -> Gemini
       -> structured result -> display
```

The ordered sequence goes to Gemini as **one** interaction with all frames
attached, each labeled with its position and timestamp. Frames are never sent
as separate requests, and the clip is never collapsed into a single image.

Concretely, the interaction `input` is one flat, ordered list of content items:
a preamble, then for each frame a `TextContent` label (`Frame 3 of 12,
t = 0.40s:`) followed by the frame itself as base64 JPEG `ImageContent`. The
answer comes back as JSON constrained by a response schema whose `sign` field
is an enum of the three signs plus `unknown`.

### Run it

Set your key first (see [Setup](#setup) above), then from `SignBridge/` with
the virtual environment active.

macOS:

```bash
python -m src.asl.gemini_demo
```

Windows PowerShell:

```powershell
python -m src.asl.gemini_demo
```

### Controls

| Key | Action |
| --- | --- |
| `SPACE` | record one clip and send one Gemini request |
| `Q` | quit |

`SPACE` is ignored while capturing or analyzing, so a key press cannot queue up
extra requests.

### What you will see

The window shows the mirrored camera with hand landmarks drawn, and a status
block that moves through four states:

```
READY       Press SPACE to capture a sign
CAPTURING   CAPTURING SIGN...  1.8s   (with a progress bar)
ANALYZING   ANALYZING WITH GEMINI...
RESULT      Detected sign: HELLO / Confidence: 94% / Description: ...
```

A failure shows `Gemini error` with `Press SPACE to try again.`, and the app
keeps running.

Begin signing right when you press SPACE: capture starts immediately, so a
late start wastes part of the window on your hand moving into position, which
is exactly the kind of clip Gemini should answer `unknown` for.

### Capture settings

Roughly a **2-second** clip sampled at **6 frames per second**, giving about 12
frames per request, capped at 24 and downscaled to 640px wide. All of it lives
in `CaptureConfig` in `src/asl/sequence_capture.py` — the demo reads its
defaults from there rather than repeating the numbers. Override per run:

```
python -m src.asl.gemini_demo --duration 3 --sample-fps 8
python -m src.asl.gemini_demo --model gemini-3.8-flash --threshold 0.75
```

### Cost and rate limiting

One key press is one request. Nothing is automatic: there is no continuous
recognition, no retry on failure, and no request triggered by hand detection.
A single-worker thread pool runs the call, which is what keeps the video live
while waiting and also guarantees one request at a time. Requests time out
after 30 seconds.

### Privacy

Captured frames exist in memory only for the duration of the request and are
never written to disk. The frames sent are the clean mirrored camera frames —
landmarks and status text are drawn only on the copy you see on screen, so
Gemini receives the hand rather than our overlay.

Webcam frames do leave the machine: they go to Google's API when you press
SPACE, and only then. The API key is read from `GEMINI_API_KEY` and is never
displayed, logged, or included in an error message.
