# SignBridge

**SignBridge is a hackathon prototype demonstrating temporal ASL recognition
for a limited vocabulary. It is not a complete ASL translation system.**

A webcam watches a signer, MediaPipe tracks both hands over time, a small
local BiGRU classifies the **motion sequence**, and ElevenLabs speaks the
matching English phrase.

```
webcam → MediaPipe Hands → temporal landmark sequence
       → local BiGRU → recognized sign → sentence buffer
       → English text → ElevenLabs TTS
```

Live vocabulary:

`HELLO`, `YES`, `NO`, `THANK_YOU`, `PLEASE`, `HELP`, `SORRY`, `GOOD`, `BAD`,
`HOW_ARE_YOU`, `I_ME`, `YOU`, `WANT`, `NEED`, `UNDERSTAND`, `DONT_UNDERSTAND`,
`WHAT`, `WHERE`, `NAME`, `GOODBYE`

These are prototype labels, not a claim of full ASL translation.

## Setup

Target **Python 3.11 or 3.12** on macOS and Windows. MediaPipe 0.10.21 has no
Python 3.13 wheel.

MediaPipe is pinned to **0.10.21** because newer releases removed
`mp.solutions.hands`, and the Tasks API aborts on Apple Silicon. That pin
requires **numpy &lt; 2**, so OpenCV stays on 4.x (`opencv-contrib-python`).

Run every command from `SignBridge/` with the virtual environment active.

macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `ELEVENLABS_API_KEY` (and optionally
`ELEVENLABS_VOICE_ID`). Never commit `.env`.

## Example workflow

```bash
python -m src.asl.sequence_data_collector   # ~20–30 clips per sign
python -m src.asl.train_sequence_model
python -m src.asl.signbridge_demo
python -m unittest discover -s tests -v
```

Hand-tracking only (no classifier):

```bash
python app.py
```

## Data collection

```bash
python -m src.asl.sequence_data_collector
```

SPACE starts a clip, SPACE stops it and saves **one** `.npz` sequence under
`data/sequences/<sign>/`. N / P change the sign. Q quits.

Default max length is **2.5 seconds** (`SIGNBRIDGE_SEQUENCE_SECONDS`). You do
not have to start the motion on the exact frame recording begins. Aim for
about 20–30 complete motions per sign (200–300 total). Do not save individual
webcam frames as the training set.

## Collecting in parallel

The new signs are split so three people can record at the same time. Each
person uses their own branch and only saves clips for their signs. Those clips
live in different `data/sequences/<sign>/` folders, so merging the branches
back into `main` does not conflict.

| Branch | Command | Signs |
| --- | --- | --- |
| `collect/group-1` | `--group 1` | `i_me`, `you`, `name`, `goodbye` |
| `collect/group-2` | `--group 2` | `want`, `need`, `what`, `do`, `can`, `okay` |
| `collect/group-3` | `--group 3` | `understand`, `dont_understand`, `where` |

`i_me` is the single point-to-self sign (I / me). `understand` and
`dont_understand` stay with the same person because they are easy to confuse.

```bash
git checkout collect/group-1
python -m src.asl.sequence_data_collector --group 1
```

Commit only the new `.npz` files under your sign folders. Do not commit
`models/`. After all three branches are merged into `main`, train once:

```bash
python -m src.asl.train_sequence_model
```

## Training

```bash
python -m src.asl.train_sequence_model
```

Each `.npz` file is one sample. The split is **by sequence** (about 80/20),
and every class appears in both train and validation. Training applies
left/right hand **mirror copies** (default on,
`SIGNBRIDGE_MIRROR_AUGMENTATION=true`) plus light feature-level jitter.
Mirrored clips are generated in memory — they are not extra files on disk.
Validation is never mirrored or jittered.

The trainer prints the real train/validation accuracy each epoch and the best
validation accuracy at the end. That number is measured, not hardcoded.

Writes (gitignored):

- `models/asl_sequence_model.pt`
- `models/asl_sequence_labels.json`

Architecture: 252 features/frame → 2-layer BiGRU (hidden 128, bidirectional)
→ masked temporal attention → dropout → one output per sign that has clips.
Runs on laptop CPU.

## Live demo

```bash
python -m src.asl.signbridge_demo
```

The demo classifies a rolling 2–3 second window a few times per second. It
does **not** treat a single frame as a sign.

- Confidence below the threshold (default **0.80** in demo mode, **0.75**
  otherwise) shows `Unknown`.
- Still hands show `WAITING FOR SIGN` and are not classified.
- A label is accepted only after several matching inferences, then a cooldown
  blocks `HELLO HELLO HELLO` repeats.
- SPACE clears the sentence. T speaks it. Q quits. Signs are accepted from
  the camera automatically.

`SIGNBRIDGE_DEMO_MODE=true` (the default) is conservative: higher threshold,
stronger smoothing, longer cooldown. A demo that stays quiet is better than
one that confidently says the wrong word.

`SIGNBRIDGE_CONFIDENCE_THRESHOLD` overrides the numeric cutoff.

## ElevenLabs

```
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
```

If the key is missing or a request fails, the English translation still
appears. The app does not crash, and keys are never logged.

## Tests

```bash
python -m unittest discover -s tests -v
```

Unit tests do not open a webcam, call ElevenLabs, or download models.

## Unused / historical modules

These files are **not** on the live path. They are kept for reference:

- `src/asl/gemini_demo.py`, `src/asl/gemini_recognizer.py`
- `src/asl/pretrained_demo.py`, `src/asl/pretrained_recognizer.py`
- `src/asl/live_recognition.py` (old single-frame classifier)
- `src/asl/data_collector.py`, `src/asl/train.py`

The live commands are the sequence collector, `train_sequence_model`, and
`signbridge_demo` above.

## Cross-platform

One codebase for macOS and Windows. Use `pathlib`. No OS-specific paths or
shell commands in application code. Webcam access is OpenCV. Audio playback
is PortAudio via `sounddevice`.
