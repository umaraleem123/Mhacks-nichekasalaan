"""Live ASL sign recognition from the webcam.

Run it with:

    python -m src.asl.live_recognition

Press Q to quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2

from src.asl.features import FeatureError, extract_features
from src.asl.recognizer import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    UNKNOWN_SIGN,
    RecognizerError,
    SignRecognizer,
)
from src.vision.camera import CAMERA_LOST_ERROR, open_camera
from src.vision.hand_tracker import HandTracker, HandTrackerError
from src.vision.overlay import AMBER, GREEN, draw_text_lines

WINDOW_NAME = "SignBridge"
MAX_CONSECUTIVE_READ_FAILURES = 30
FPS_WINDOW = 30


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recognize ASL signs from the webcam.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="below this confidence the prediction is reported as unknown",
    )
    parser.add_argument("--model", type=Path, default=None, help="trained model path")
    parser.add_argument(
        "--camera", type=int, default=None,
        help="camera index; default prefers a Logitech Brio, then the default camera",
    )
    return parser.parse_args(argv)


def run_loop(camera, tracker: HandTracker, recognizer: SignRecognizer) -> int:
    frame_times: deque[float] = deque(maxlen=FPS_WINDOW)
    read_failures = 0
    last_tick = time.perf_counter()

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

        sign, confidence = UNKNOWN_SIGN, 0.0
        if hands:
            try:
                sign, confidence = recognizer.predict(extract_features(hands[0]))
            except FeatureError as exc:
                print(f"Skipped a frame: {exc}", file=sys.stderr)

        now = time.perf_counter()
        frame_times.append(now - last_tick)
        last_tick = now
        fps = len(frame_times) / sum(frame_times) if sum(frame_times) > 0 else 0.0

        lines = ["SIGNBRIDGE", f"SIGN: {sign.upper()}"]
        if hands:
            lines.append(f"CONFIDENCE: {confidence:.0%}")
        else:
            lines.append("NO HAND DETECTED")
        lines.append(f"FPS: {fps:.0f}")
        lines.append("Press 'q' to quit")

        recognized = bool(hands) and sign != UNKNOWN_SIGN
        draw_text_lines(frame, lines, color=GREEN if recognized else AMBER)
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        recognizer = SignRecognizer(args.model, confidence_threshold=args.threshold)
    except RecognizerError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Loaded model with signs: {', '.join(recognizer.classes)}")

    opened = open_camera(args.camera)
    if opened is None:
        return 1
    camera, _camera_name = opened

    try:
        with HandTracker() as tracker:
            return run_loop(camera, tracker, recognizer)
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
