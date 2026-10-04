"""Record complete sign motions as one `.npz` file each.

    python -m src.asl.sequence_data_collector

SPACE starts and stops a clip. N / P change the target sign. Q quits.

Each saved file is one training example covering the whole movement, not a
single frame. Target about 20–30 clips per sign.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from src import SEQUENCE_DATA_DIR
from src.asl import SIGNS, display_label
from src.asl.sequence_dataset import save_sequence
from src.asl.sequence_features import frame_features, positions_from_hands
from src.vision.camera import CAMERA_LOST_ERROR, open_camera
from src.vision.hand_tracker import HandTracker, HandTrackerError
from src.vision.overlay import AMBER, GREEN, RED, draw_text_lines

WINDOW_NAME = "SignBridge - Data Collection"
MAX_CONSECUTIVE_READ_FAILURES = 30
DEFAULT_MAX_SECONDS = 2.5
DEFAULT_TARGET = 25
MIN_FRAMES = 8
SEQUENCE_SECONDS_ENV = "SIGNBRIDGE_SEQUENCE_SECONDS"
KEY_SPACE = 32


def sequence_seconds(default: float = DEFAULT_MAX_SECONDS) -> float:
    raw = os.environ.get(SEQUENCE_SECONDS_ENV, "").strip()
    if not raw:
        return default
    try:
        return min(8.0, max(0.8, float(raw)))
    except ValueError:
        return default


def count_saved(data_dir: Path, sign: str) -> int:
    folder = Path(data_dir) / sign
    if not folder.is_dir():
        return 0
    return len(list(folder.glob("*.npz")))


def next_sequence_path(data_dir: Path, sign: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    return Path(data_dir) / sign / f"{stamp}.npz"


@dataclass
class SequenceRecorder:
    """Camera-free recorder: feed 126-d position frames, save one clip."""

    data_dir: Path
    max_seconds: float = DEFAULT_MAX_SECONDS
    min_frames: int = MIN_FRAMES
    clock: Callable[[], float] = time.perf_counter
    recording: bool = False
    sign: str = field(default_factory=lambda: SIGNS[0])
    frames: list[np.ndarray] = field(default_factory=list)
    last_message: str = ""
    last_saved: Path | None = None
    _previous: np.ndarray | None = field(default=None, repr=False)
    _started_at: float = 0.0

    @property
    def elapsed(self) -> float:
        if not self.recording:
            return 0.0
        return max(0.0, self.clock() - self._started_at)

    def start(self, sign: str) -> None:
        self.sign = sign
        self.frames = []
        self._previous = None
        self._started_at = self.clock()
        self.recording = True
        self.last_message = ""
        self.last_saved = None

    def add_positions(self, positions: np.ndarray) -> bool:
        """Append one frame. Returns True if the clip auto-stopped and saved."""
        if not self.recording:
            return False
        features = frame_features(positions, self._previous)
        self._previous = np.asarray(positions, dtype=np.float32).reshape(-1).copy()
        self.frames.append(features)
        if self.elapsed >= self.max_seconds:
            self.stop(save=True)
            return True
        return False

    def stop(self, save: bool = True) -> Path | None:
        self.recording = False
        if not save:
            self.frames = []
            return None
        if len(self.frames) < self.min_frames:
            self.last_message = (
                f"Clip too short ({len(self.frames)} frames). Record a longer motion."
            )
            self.frames = []
            return None
        path = next_sequence_path(self.data_dir, self.sign)
        stacked = np.stack(self.frames, axis=0)
        save_sequence(path, stacked, self.sign)
        self.last_saved = path
        self.last_message = f"Saved {path.name} ({stacked.shape[0]} frames)"
        self.frames = []
        return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect complete ASL sign sequences for local training.",
    )
    parser.add_argument("--data-dir", type=Path, default=SEQUENCE_DATA_DIR)
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--camera", type=int, default=None)
    return parser.parse_args(argv)


def overlay_lines(
    sign: str,
    recorder: SequenceRecorder,
    target: int,
    saved_count: int,
) -> list[str]:
    lines = [
        "SIGNBRIDGE DATA COLLECTION",
        "",
        f"Current sign: {display_label(sign)}",
        f"Saved: {saved_count} / ~{target}",
        "",
    ]
    if recorder.recording:
        lines += [
            "Press SPACE to STOP recording",
            f"Recording: {recorder.elapsed:.1f}s",
            f"Frames: {len(recorder.frames)}",
        ]
    else:
        lines += [
            "Press SPACE to START recording",
            "Press N for next sign",
            "Press P for previous sign",
            "Press Q to quit",
        ]
    if recorder.last_message:
        lines += ["", recorder.last_message]
    return lines


def run_loop(
    camera,
    tracker: HandTracker,
    recorder: SequenceRecorder,
    signs: list[str],
    target: int,
) -> int:
    index = 0
    read_failures = 0
    while True:
        captured, frame = camera.read()
        if not captured:
            read_failures += 1
            if read_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                print(CAMERA_LOST_ERROR, file=sys.stderr)
                return 1
            cv2.waitKey(10)
            continue
        read_failures = 0

        frame = cv2.flip(frame, 1)
        hands = tracker.process(frame)
        tracker.draw(frame, hands)
        if recorder.recording:
            recorder.add_positions(positions_from_hands(hands))

        sign = signs[index]
        saved = count_saved(recorder.data_dir, sign)
        color = AMBER if recorder.recording else GREEN
        if recorder.last_message.startswith("Clip too short"):
            color = RED
        draw_text_lines(frame, overlay_lines(sign, recorder, target, saved), color=color)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            if recorder.recording:
                recorder.stop(save=False)
            return 0
        if key == KEY_SPACE:
            if recorder.recording:
                recorder.stop(save=True)
            else:
                recorder.start(sign)
        elif key in (ord("n"), ord("N")) and not recorder.recording:
            index = (index + 1) % len(signs)
            recorder.last_message = ""
        elif key in (ord("p"), ord("P")) and not recorder.recording:
            index = (index - 1) % len(signs)
            recorder.last_message = ""


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    max_seconds = args.max_seconds if args.max_seconds is not None else sequence_seconds()
    data_dir = Path(args.data_dir)
    for sign in SIGNS:
        (data_dir / sign).mkdir(parents=True, exist_ok=True)

    opened = open_camera(args.camera)
    if opened is None:
        return 1
    camera, camera_name = opened
    print(f"Camera: {camera_name}")
    print(f"Saving sequences under {data_dir}")
    print(f"Max clip length: {max_seconds:.1f}s   target ~{args.target} per sign")

    recorder = SequenceRecorder(data_dir=data_dir, max_seconds=max_seconds)
    try:
        with HandTracker() as tracker:
            return run_loop(camera, tracker, recorder, SIGNS, args.target)
    except HandTrackerError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
