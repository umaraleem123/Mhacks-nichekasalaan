#!/usr/bin/env python3
"""Download the MediaPipe hand_landmarker.task model into models/."""

from __future__ import annotations

import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "models" / "hand_landmarker.task"


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.is_file() and DEST.stat().st_size > 0:
        print(f"Model already present: {DEST} ({DEST.stat().st_size} bytes)")
        return 0

    print(f"Downloading {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, DEST)
    print(f"Saved to {DEST} ({DEST.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
