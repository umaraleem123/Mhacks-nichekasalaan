#!/usr/bin/env python3
"""Webcam hand and finger detection using MediaPipe Hand Landmarker + OpenCV.

Opens the default camera, draws 21 hand landmarks and finger tips, shows an
approximate raised-finger count, and quits cleanly on 'q' or Esc.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"

# MediaPipe hand landmark indices for finger tips and pip joints.
FINGER_TIPS = {
    "Thumb": 4,
    "Index": 8,
    "Middle": 12,
    "Ring": 16,
    "Pinky": 20,
}
FINGER_PIPS = {
    "Thumb": 3,
    "Index": 6,
    "Middle": 10,
    "Ring": 14,
    "Pinky": 18,
}
# For non-thumb fingers, compare tip to this joint along the finger axis.
FINGER_MCP = {
    "Index": 5,
    "Middle": 9,
    "Ring": 13,
    "Pinky": 17,
}


def ensure_model(path: Path = MODEL_PATH) -> Path:
    """Download the Hand Landmarker model if it is not already present."""
    if path.is_file() and path.stat().st_size > 0:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Hand Landmarker model to {path} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
    except Exception as exc:  # noqa: BLE001 - surface download failures clearly
        if path.exists():
            path.unlink(missing_ok=True)
        raise SystemExit(
            f"Failed to download model from {MODEL_URL}\n{exc}\n"
            "Check your network, then retry."
        ) from exc
    print("Model ready.")
    return path


def count_raised_fingers(landmarks, handedness_label: str) -> int:
    """Return a rough count of raised fingers from normalized landmarks."""
    raised = 0

    # Thumb: use x relative to pip; mirror for left vs right hand.
    tip = landmarks[FINGER_TIPS["Thumb"]]
    pip = landmarks[FINGER_PIPS["Thumb"]]
    if handedness_label == "Right":
        if tip.x < pip.x:
            raised += 1
    else:
        if tip.x > pip.x:
            raised += 1

    for name in ("Index", "Middle", "Ring", "Pinky"):
        tip = landmarks[FINGER_TIPS[name]]
        mcp = landmarks[FINGER_MCP[name]]
        # Tip above MCP in image space (y grows downward).
        if tip.y < mcp.y:
            raised += 1

    return raised


def draw_fingertip_labels(frame, landmarks, width: int, height: int) -> None:
    for name, idx in FINGER_TIPS.items():
        lm = landmarks[idx]
        x, y = int(lm.x * width), int(lm.y * height)
        cv2.circle(frame, (x, y), 6, (0, 220, 255), -1)
        cv2.putText(
            frame,
            name,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect hands and fingers from a webcam using MediaPipe."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0, usually the built-in or first USB webcam).",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=2,
        help="Maximum number of hands to detect (default: 2).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Capture width request (default: 1280).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Capture height request (default: 720).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="Path to hand_landmarker.task (downloaded automatically if missing).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = ensure_model(args.model)

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=args.max_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(
            f"Could not open camera index {args.camera}.\n"
            "Tips:\n"
            "  • Plug in your Logitech (or any) webcam and try again\n"
            "  • On macOS, allow Camera access for Terminal/iTerm/your IDE\n"
            "  • Try --camera 1 if you have more than one device",
            file=sys.stderr,
        )
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("Webcam hand detection running. Press 'q' or Esc to quit.")

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        frame_timestamp_ms = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Failed to read frame from camera.", file=sys.stderr)
                    break

                # Mirror for a natural selfie view.
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                frame_timestamp_ms += 33  # ~30 FPS clock for VIDEO mode

                total_fingers = 0
                if result.hand_landmarks:
                    for i, hand_landmarks in enumerate(result.hand_landmarks):
                        drawing_utils.draw_landmarks(
                            frame,
                            hand_landmarks,
                            vision.HandLandmarksConnections.HAND_CONNECTIONS,
                            drawing_styles.get_default_hand_landmarks_style(),
                            drawing_styles.get_default_hand_connections_style(),
                        )
                        draw_fingertip_labels(frame, hand_landmarks, width, height)

                        label = "Unknown"
                        if result.handedness and i < len(result.handedness):
                            cats = result.handedness[i]
                            if cats:
                                label = cats[0].category_name

                        fingers = count_raised_fingers(hand_landmarks, label)
                        total_fingers += fingers
                        wrist = hand_landmarks[0]
                        wx, wy = int(wrist.x * width), int(wrist.y * height)
                        cv2.putText(
                            frame,
                            f"{label}: {fingers} fingers",
                            (wx - 40, max(24, wy - 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (80, 255, 120),
                            2,
                            cv2.LINE_AA,
                        )

                cv2.putText(
                    frame,
                    f"Hands: {len(result.hand_landmarks or [])}  "
                    f"Raised fingers: {total_fingers}",
                    (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    "Press q or Esc to quit",
                    (16, height - 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imshow("Hand / Finger Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or Esc
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    print("Stopped cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
