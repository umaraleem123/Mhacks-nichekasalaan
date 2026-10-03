#!/usr/bin/env python3
"""Live finger detection from a Logitech (or any) webcam.

Usage:
  python detect_fingers.py              # default camera index 0
  python detect_fingers.py --camera 1   # another USB camera
  python detect_fingers.py --list       # list available cameras
  python detect_fingers.py --image photo.jpg  # still-image smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from finger_detection.detector import FingerDetector, draw_finger_overlay


def list_cameras(max_index: int = 8) -> list[int]:
    found: list[int] = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            found.append(index)
            cap.release()
    return found


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    # CAP_V4L2 is the usual backend for Logitech webcams on Linux.
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {index}. "
            "Plug in the Logitech webcam, then try --list or another --camera index."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def run_webcam(camera: int, width: int, height: int, mirror: bool) -> int:
    try:
        cap = open_camera(camera, width, height)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        available = list_cameras()
        if available:
            print(f"Available camera indexes: {available}", file=sys.stderr)
        else:
            print("No cameras were detected on this machine.", file=sys.stderr)
        return 1

    print(f"Using camera {camera}. Press q to quit.")
    with FingerDetector() as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read a frame from the webcam.", file=sys.stderr)
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            results = detector.detect(frame)
            draw_finger_overlay(frame, results)
            cv2.imshow("Finger Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0


def run_image(image_path: Path) -> int:
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Could not read image: {image_path}", file=sys.stderr)
        return 1

    with FingerDetector(still_image=True) as detector:
        results = detector.detect(frame)
        draw_finger_overlay(frame, results)

    out = image_path.with_name(f"{image_path.stem}_fingers{image_path.suffix}")
    cv2.imwrite(str(out), frame)

    if results:
        for hand in results:
            names = ", ".join(hand.raised_names) or "none"
            print(f"{hand.handedness}: {hand.count} finger(s) up — {names}")
    else:
        print("No hands detected.")

    print(f"Wrote annotated image to {out}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect fingers with a Logitech webcam.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal mirror (mirror is on by default for a selfie-style view)",
    )
    parser.add_argument("--list", action="store_true", help="List available camera indexes and exit")
    parser.add_argument(
        "--image",
        type=Path,
        help="Run detection on a still image instead of the webcam (useful for testing)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list:
        cameras = list_cameras()
        if cameras:
            print("Available camera indexes:", ", ".join(map(str, cameras)))
        else:
            print("No cameras detected.")
        return 0

    if args.image is not None:
        return run_image(args.image)

    return run_webcam(
        camera=args.camera,
        width=args.width,
        height=args.height,
        mirror=not args.no_mirror,
    )


if __name__ == "__main__":
    raise SystemExit(main())
