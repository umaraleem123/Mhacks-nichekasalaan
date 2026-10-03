"""Default-webcam access, shared by every SignBridge application.

Plain OpenCV `VideoCapture` so the same code runs on macOS and Windows.
"""

from __future__ import annotations

import sys

import cv2

REQUESTED_WIDTH = 1280
REQUESTED_HEIGHT = 720

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


def open_camera(index: int = 0) -> cv2.VideoCapture | None:
    """Open a camera, or return None with an explanation printed to stderr."""
    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        camera.release()
        print(CAMERA_OPEN_ERROR, file=sys.stderr)
        return None

    # Requests only; the driver picks the nearest supported mode.
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT)
    return camera
