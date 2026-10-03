"""Webcam capture helpers for ASL recognition."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2


def list_cameras(max_index: int = 8) -> List[int]:
    """Probe OpenCV camera indexes and return those that open successfully."""
    found: List[int] = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            found.append(index)
            cap.release()
    return found


def open_camera(
    index: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
) -> cv2.VideoCapture:
    """Open a webcam (prefers V4L2 on Linux for Logitech / UVC devices)."""
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        available = list_cameras()
        hint = (
            f"Available indexes: {available}" if available else "No cameras were detected."
        )
        raise RuntimeError(
            f"Could not open camera index {index}. "
            f"Plug in a webcam, then try --list or another --camera index. {hint}"
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def read_frame(
    cap: cv2.VideoCapture,
    *,
    mirror: bool = True,
) -> Tuple[bool, Optional[object]]:
    """Read one frame; optionally mirror for a selfie-style view."""
    ok, frame = cap.read()
    if not ok or frame is None:
        return False, None
    if mirror:
        frame = cv2.flip(frame, 1)
    return True, frame
