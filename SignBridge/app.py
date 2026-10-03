"""SignBridge — Milestone 1: live webcam hand tracking.

Opens the default camera, overlays MediaPipe hand landmarks, and shows the
result in a window. No sign recognition yet.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

from src.vision.hand_tracker import DetectedHand, HandTracker, HandTrackerError

WINDOW_NAME = "SignBridge"
REQUESTED_WIDTH = 1280
REQUESTED_HEIGHT = 720
MAX_CONSECUTIVE_READ_FAILURES = 30

CAMERA_OPEN_ERROR = """Error: could not open the default webcam.

Things to check:
  1. Camera permission. macOS: System Settings > Privacy & Security > Camera,
     and enable the terminal app you launched this from. Windows: Settings >
     Privacy & security > Camera, and allow desktop apps to access the camera.
  2. Another application may be holding the camera. Quit video calls, browser
     tabs, and recording tools, then try again.
  3. On a desktop machine, confirm an external webcam is plugged in."""

CAMERA_LOST_ERROR = """Error: lost the connection to the webcam.

The camera stopped returning frames. It may have been unplugged or claimed by
another application. Reconnect it and run the app again."""


def open_camera() -> cv2.VideoCapture | None:
    """Open the default camera, or return None with an explanation printed."""
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        print(CAMERA_OPEN_ERROR, file=sys.stderr)
        return None

    # Requests only; the driver picks the nearest supported mode.
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT)
    return camera


def draw_overlay(frame: np.ndarray, hands: list[DetectedHand]) -> None:
    """Draw the status text in the top-left corner."""
    lines = [WINDOW_NAME, f"Hands detected: {len(hands)}"]
    lines += [f"{hand.label} hand ({hand.confidence:.0%})" for hand in hands]
    lines.append("Press 'q' to quit")

    for index, line in enumerate(lines):
        position = (12, 32 + index * 30)
        # Dark pass underneath keeps the text readable on a bright background.
        cv2.putText(frame, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


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
