"""Finger detection helpers built on MediaPipe Hand Landmarker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# MediaPipe hand landmark indices:
# 0 wrist; thumb 1-4; index 5-8; middle 9-12; ring 13-16; pinky 17-20
FINGER_TIPS = (4, 8, 12, 16, 20)
FINGER_PIPS = (3, 6, 10, 14, 18)
FINGER_NAMES = ("Thumb", "Index", "Middle", "Ring", "Pinky")

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


@dataclass(frozen=True)
class FingerResult:
    """Detection result for one hand."""

    handedness: str
    raised: Tuple[bool, bool, bool, bool, bool]
    tip_pixels: Tuple[Tuple[int, int], ...]
    count: int

    @property
    def raised_names(self) -> List[str]:
        return [name for name, up in zip(FINGER_NAMES, self.raised) if up]


def _is_finger_up(
    landmarks: Sequence[Any],
    tip_idx: int,
    pip_idx: int,
    *,
    is_thumb: bool,
    handedness: str,
) -> bool:
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]

    if is_thumb:
        # Thumb opens sideways; use x relative to the PIP and handedness.
        if handedness.lower().startswith("right"):
            return tip.x < pip.x
        return tip.x > pip.x

    # Other fingers open when the tip is above the PIP (smaller y in image space).
    return tip.y < pip.y


class FingerDetector:
    """Detect raised fingers from BGR webcam frames or still images."""

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        *,
        max_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        still_image: bool = False,
    ) -> None:
        path = Path(model_path) if model_path else DEFAULT_MODEL
        if not path.is_file():
            raise FileNotFoundError(
                f"Hand landmarker model not found at {path}. "
                "Download it into models/hand_landmarker.task or pass model_path."
            )

        mode = mp_vision.RunningMode.IMAGE if still_image else mp_vision.RunningMode.VIDEO
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=mode,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._still_image = still_image
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FingerDetector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def detect(self, bgr_frame: np.ndarray) -> List[FingerResult]:
        """Return raised-finger results for every detected hand in a BGR frame."""
        if bgr_frame is None or bgr_frame.size == 0:
            return []

        rgb = bgr_frame[:, :, ::-1].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self._still_image:
            result = self._landmarker.detect(mp_image)
        else:
            self._timestamp_ms += 33  # ~30 FPS clock for VIDEO mode
            result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        return self._to_results(result, bgr_frame.shape[1], bgr_frame.shape[0])

    @staticmethod
    def _to_results(result: Any, width: int, height: int) -> List[FingerResult]:
        outputs: List[FingerResult] = []
        hand_landmarks = result.hand_landmarks or []
        handedness_list = result.handedness or []

        for idx, landmarks in enumerate(hand_landmarks):
            label = "Unknown"
            if idx < len(handedness_list) and handedness_list[idx]:
                label = handedness_list[idx][0].category_name

            raised_flags = [
                _is_finger_up(
                    landmarks,
                    tip,
                    pip,
                    is_thumb=(finger_i == 0),
                    handedness=label,
                )
                for finger_i, (tip, pip) in enumerate(zip(FINGER_TIPS, FINGER_PIPS))
            ]
            raised = (
                raised_flags[0],
                raised_flags[1],
                raised_flags[2],
                raised_flags[3],
                raised_flags[4],
            )
            tips = tuple(
                (int(landmarks[i].x * width), int(landmarks[i].y * height))
                for i in FINGER_TIPS
            )
            outputs.append(
                FingerResult(
                    handedness=label,
                    raised=raised,
                    tip_pixels=tips,
                    count=sum(raised),
                )
            )

        return outputs


def draw_finger_overlay(
    frame: np.ndarray,
    results: Iterable[FingerResult],
    *,
    tip_radius: int = 8,
) -> np.ndarray:
    """Draw fingertip markers and a count HUD onto a BGR frame (in place)."""
    import cv2

    y = 36
    for hand in results:
        color = (60, 200, 90) if hand.handedness.lower().startswith("right") else (80, 160, 255)
        for up, (x, tip_y) in zip(hand.raised, hand.tip_pixels):
            if up:
                cv2.circle(frame, (x, tip_y), tip_radius, color, -1, lineType=cv2.LINE_AA)
                cv2.circle(
                    frame,
                    (x, tip_y),
                    tip_radius + 3,
                    (255, 255, 255),
                    1,
                    lineType=cv2.LINE_AA,
                )

        label = f"{hand.handedness}: {hand.count}  ({', '.join(hand.raised_names) or 'none'})"
        cv2.putText(
            frame,
            label,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 32

    if not results:
        cv2.putText(
            frame,
            "Show a hand to the camera",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )

    return frame
