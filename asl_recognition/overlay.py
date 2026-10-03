"""Drawing helpers for the ASL recognition HUD."""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from .classifier import ASLPrediction
from .landmarks import FINGER_TIPS, HandLandmarks

# MediaPipe hand connections (subset for a clean skeleton)
_HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
)


def draw_hand_skeleton(frame: np.ndarray, hand: HandLandmarks) -> None:
    color_bone = (180, 180, 180)
    color_joint = (40, 200, 255)
    for a, b in _HAND_CONNECTIONS:
        cv2.line(frame, hand.pixel(a), hand.pixel(b), color_bone, 2, cv2.LINE_AA)
    for i in range(21):
        radius = 5 if i in FINGER_TIPS else 3
        cv2.circle(frame, hand.pixel(i), radius, color_joint, -1, cv2.LINE_AA)


def draw_asl_overlay(
    frame: np.ndarray,
    hands: Sequence[HandLandmarks],
    prediction: Optional[ASLPrediction],
    *,
    spelled: str = "",
) -> np.ndarray:
    """Draw landmarks + recognized letter HUD onto a BGR frame (in place)."""
    for hand in hands:
        draw_hand_skeleton(frame, hand)

    h, w = frame.shape[:2]
    panel_h = 110
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    if prediction and prediction.letter:
        letter = prediction.letter
        conf = prediction.confidence
        cv2.putText(
            frame,
            letter,
            (24, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.2,
            (80, 255, 160),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"{conf:.0%}  {prediction.handedness}",
            (110, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        tops = ", ".join(f"{L}:{s:.0%}" for L, s in prediction.candidates[:3])
        cv2.putText(
            frame,
            tops,
            (24, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
    elif hands:
        cv2.putText(
            frame,
            "Hold a clear ASL letter…",
            (24, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            "Show a hand to the camera",
            (24, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )

    if spelled:
        cv2.putText(
            frame,
            f"Text: {spelled}",
            (24, h - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 220, 120),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        "q quit | space add letter | backspace delete | c clear",
        (24, h - 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )
    return frame
