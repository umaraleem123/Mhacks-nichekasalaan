"""Record normalized hand-landmark samples for training the sign classifier.

Run it with:

    python -m src.asl.data_collector

Keyboard controls (also drawn on screen):

    1..9  select the sign to record, numbered as listed in `src.asl.SIGNS`
    R     start/stop recording
    Q     quit

Samples are appended to data/asl/<sign>/samples.csv, one row per sample: the
63 feature values followed by the label. No images are saved.

Samples are only recorded while exactly one hand is visible.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

from src import ASL_DATA_DIR
from src.asl import SIGNS
from src.asl.features import FeatureError, extract_features, feature_names
from src.vision.camera import CAMERA_LOST_ERROR, open_camera
from src.vision.hand_tracker import HandTracker, HandTrackerError
from src.vision.overlay import AMBER, GREEN, RED, draw_text_lines

WINDOW_NAME = "SignBridge - Data Collection"
MAX_CONSECUTIVE_READ_FAILURES = 30
LABEL_COLUMN = "label"


def sample_file(data_dir: Path, sign: str) -> Path:
    return data_dir / sign / "samples.csv"


def count_samples(path: Path) -> int:
    """Rows already stored for one sign, excluding the header."""
    if not path.is_file():
        return 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


class SampleWriter:
    """Appends feature rows to one CSV per sign, creating files on demand."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._columns = feature_names() + [LABEL_COLUMN]

    def append(self, sign: str, features) -> None:
        path = sample_file(self._data_dir, sign)
        path.parent.mkdir(parents=True, exist_ok=True)

        is_new = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if is_new:
                writer.writerow(self._columns)
            writer.writerow([f"{value:.6f}" for value in features] + [sign])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect hand-landmark samples for ASL signs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target", type=int, default=200,
        help="samples to aim for per sign; recording pauses on reaching it",
    )
    parser.add_argument(
        "--interval", type=float, default=0.1,
        help="seconds between recorded samples, to avoid near-duplicate frames",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=ASL_DATA_DIR,
        help="where sample CSVs are written",
    )
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    return parser.parse_args(argv)


def hand_status(hand_count: int) -> str:
    if hand_count == 1:
        return "HAND DETECTED"
    if hand_count == 0:
        return "NO HAND"
    return f"{hand_count} HANDS - SHOW ONLY ONE"


def build_status_lines(
    sign: str, counts: dict[str, int], target: int, hand_count: int, recording: bool
) -> list[str]:
    lines = [
        "SIGNBRIDGE - DATA COLLECTION",
        f"SIGN: {sign.upper()}  ({counts.get(sign, 0)}/{target})",
        "RECORDING" if recording else "PAUSED",
        hand_status(hand_count),
        "",
    ]
    lines += [
        f"  {index}: {name}  [{counts.get(name, 0)}]"
        for index, name in enumerate(SIGNS, start=1)
    ]
    lines.append("")
    lines.append("R = record on/off    Q = quit")
    return lines


def run_loop(camera, tracker: HandTracker, writer: SampleWriter, args) -> int:
    counts = {
        name: count_samples(sample_file(args.data_dir, name)) for name in SIGNS
    }
    sign = SIGNS[0]
    recording = False
    last_saved = 0.0
    read_failures = 0
    # Signs already auto-paused, so hitting the target stops recording once
    # instead of blocking you from deliberately collecting more.
    paused_at_target: set[str] = set()

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

        frame = cv2.flip(frame, 1)  # selfie view, and what handedness assumes
        hands = tracker.process(frame)
        tracker.draw(frame, hands)

        if recording and counts.get(sign, 0) >= args.target and sign not in paused_at_target:
            recording = False
            paused_at_target.add(sign)

        # Exactly one hand: with two in frame MediaPipe gives no stable order,
        # so recording either one would mix hands into the same class.
        single_hand = len(hands) == 1
        if recording and single_hand and time.perf_counter() - last_saved >= args.interval:
            try:
                writer.append(sign, extract_features(hands[0]))
            except FeatureError as exc:
                print(f"Skipped a sample: {exc}", file=sys.stderr)
            else:
                counts[sign] = counts.get(sign, 0) + 1
                last_saved = time.perf_counter()

        color = RED if recording else (GREEN if single_hand else AMBER)
        draw_text_lines(
            frame,
            build_status_lines(sign, counts, args.target, len(hands), recording),
            color=color,
        )
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return 0
        if key == ord("r"):
            recording = not recording
        elif ord("1") <= key <= ord("9"):
            index = key - ord("1")
            if index < len(SIGNS):
                sign, recording = SIGNS[index], False


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing samples to {args.data_dir}")
    print("Controls: " + "  ".join(
        f"{i}={name}" for i, name in enumerate(SIGNS, start=1)
    ) + "  R=record  Q=quit")

    camera = open_camera(args.camera)
    if camera is None:
        return 1

    try:
        with HandTracker() as tracker:
            return run_loop(camera, tracker, SampleWriter(args.data_dir), args)
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
