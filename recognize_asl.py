#!/usr/bin/env python3
"""Camera capture and ASL sign recognition.

Usage:
  python recognize_asl.py                 # default camera index 0
  python recognize_asl.py --camera 1      # another USB camera
  python recognize_asl.py --list          # list available cameras
  python recognize_asl.py --image hand.jpg  # still-image smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from asl_recognition.camera import list_cameras, open_camera, read_frame
from asl_recognition.overlay import draw_asl_overlay
from asl_recognition.pipeline import ASLRecognizer
from asl_recognition.classifier import ASLPrediction


def run_webcam(camera: int, width: int, height: int, mirror: bool) -> int:
    try:
        cap = open_camera(camera, width=width, height=height)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Using camera {camera}.")
    print("Keys: q quit | space add letter | backspace delete | c clear text")

    with ASLRecognizer(still_image=False) as recognizer:
        while True:
            ok, frame = read_frame(cap, mirror=mirror)
            if not ok or frame is None:
                print("Failed to read a frame from the webcam.", file=sys.stderr)
                break

            stable = recognizer.process(frame)
            prediction = None
            if stable.raw is not None:
                # Show the stabilized letter when available, else the raw guess.
                if stable.letter:
                    prediction = ASLPrediction(
                        letter=stable.letter,
                        confidence=stable.confidence,
                        handedness=stable.raw.handedness,
                        candidates=stable.raw.candidates,
                    )
                else:
                    prediction = stable.raw

            draw_asl_overlay(
                frame,
                stable.hands,
                prediction,
                spelled=recognizer.spelled,
            )
            cv2.imshow("ASL Sign Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                added = recognizer.commit_letter(stable.letter)
                if added:
                    print(f"+ {added}  →  {recognizer.spelled}")
            elif key in (8, 127):  # backspace / delete
                recognizer.backspace()
            elif key == ord("c"):
                recognizer.clear_text()
                print("Cleared spelled text.")

    cap.release()
    cv2.destroyAllWindows()
    if recognizer.spelled:
        print(f"Final text: {recognizer.spelled}")
    return 0


def run_image(image_path: Path) -> int:
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Could not read image: {image_path}", file=sys.stderr)
        return 1

    with ASLRecognizer(still_image=True, history=1, min_votes=1) as recognizer:
        stable = recognizer.process(frame)
        prediction = stable.raw
        if stable.letter and prediction is not None:
            prediction = ASLPrediction(
                letter=stable.letter,
                confidence=stable.confidence,
                handedness=prediction.handedness,
                candidates=prediction.candidates,
            )
        draw_asl_overlay(frame, stable.hands, prediction, spelled="")

    out = image_path.with_name(f"{image_path.stem}_asl{image_path.suffix}")
    cv2.imwrite(str(out), frame)

    if prediction and prediction.letter:
        tops = ", ".join(f"{L}:{s:.0%}" for L, s in prediction.candidates[:5])
        print(f"Detected: {prediction.letter} ({prediction.confidence:.0%})")
        print(f"Top candidates: {tops}")
        print(f"Handedness: {prediction.handedness}")
    elif stable.hands:
        print("Hand detected, but no confident ASL letter.")
        if prediction:
            tops = ", ".join(f"{L}:{s:.0%}" for L, s in prediction.candidates[:5])
            print(f"Top candidates: {tops}")
    else:
        print("No hands detected.")

    print(f"Wrote annotated image to {out}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Camera capture and ASL sign recognition.",
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal mirror (mirror is on by default)",
    )
    parser.add_argument("--list", action="store_true", help="List available camera indexes and exit")
    parser.add_argument(
        "--image",
        type=Path,
        help="Run recognition on a still image instead of the webcam",
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
