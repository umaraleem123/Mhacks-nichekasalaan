"""SignBridge — Milestone 1: live webcam hand tracking.

Opens the default camera, overlays MediaPipe hand landmarks, and shows the
result in a window. No sign recognition here; that lives in
`src.asl.live_recognition`.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

from src.vision.camera import CAMERA_LOST_ERROR, open_camera
from src.vision.hand_tracker import DetectedHand, HandTracker, HandTrackerError
from src.vision.overlay import draw_text_lines

WINDOW_NAME = "SignBridge"
MAX_CONSECUTIVE_READ_FAILURES = 30


def draw_overlay(frame: np.ndarray, hands: list[DetectedHand]) -> None:
    """Draw the status text in the top-left corner."""
    lines = [WINDOW_NAME, f"Hands detected: {len(hands)}"]
    lines += [f"{hand.label} hand ({hand.confidence:.0%})" for hand in hands]
    lines.append("Press 'q' to quit")
    draw_text_lines(frame, lines)


def run_loop(camera: cv2.VideoCapture, tracker: HandTracker) -> int:
    """Capture, track, and display until the user quits. Returns an exit code."""
    read_failures = 0

    while True:
        captured, frame = camera.read()
        if not captured:
            read_failures += 1
            if read_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                print(CAMERA_LOST_ERROR, file=sys.stderr)
                return 1
            cv2.waitKey(10)  # brief pause in case the camera recovers
            continue
        read_failures = 0

        # Mirror first: selfie view for the user, and the orientation MediaPipe
        # expects when labeling hands Left or Right.
        frame = cv2.flip(frame, 1)

        hands = tracker.process(frame)
        tracker.draw(frame, hands)
        draw_overlay(frame, hands)
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return 0


def main() -> int:
    camera = open_camera()
    if camera is None:
        return 1

    try:
        with HandTracker() as tracker:
            return run_loop(camera, tracker)
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
