# SignBridge

> **Status: environment setup.** The repository contains the project skeleton
> and the dependencies for the first milestone. No application logic or models
> exist yet. See [Development Setup](#development-setup).

SignBridge is a bidirectional communication system that bridges American Sign
Language and spoken English, built as a 24-hour hackathon MVP.

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

## Development Setup

Current milestone: **webcam access + hand tracking.** The only dependencies are
OpenCV and MediaPipe. Nothing is implemented yet; these steps just prepare the
environment.

### 1. Python version

Use **Python 3.11**. MediaPipe ships prebuilt wheels for 3.9–3.12, and 3.11 is
the version with the most reliable OpenCV and MediaPipe wheel coverage.

### 2. Check your Python version

macOS/Linux:

```bash
python3.11 --version
```

Windows PowerShell:

```powershell
py -3.11 --version
```

Either should print `Python 3.11.x`. If the command is not found, install 3.11
first:

- **macOS (Homebrew):** `brew install python@3.11`
- **Windows:** install from [python.org/downloads](https://www.python.org/downloads/)
  and tick "Add python.exe to PATH"
- **Debian/Ubuntu:** `sudo apt install python3.11 python3.11-venv`

> Note: macOS ships a system Python (often 3.9) as plain `python3`. Do not build
> the virtual environment from it — call `python3.11` explicitly.

### 3. Create the virtual environment

Run from the `SignBridge/` directory.

macOS/Linux:

```bash
python3.11 -m venv .venv
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
```

### 4. Activate it

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, allow it for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Once active, your prompt is prefixed with `(.venv)`. Confirm with
`python --version`, which should now report 3.11.x.

### 5. Install dependencies

With the environment activated:

```bash
pip install -r requirements.txt
```

### 6. Verify the installation

```bash
python -c "import cv2, mediapipe; print('opencv', cv2.__version__); print('mediapipe', mediapipe.__version__)"
```

Two version lines and no traceback means the environment is ready. This only
imports the libraries — it does not open the webcam.

### Deactivating

```bash
deactivate
```
