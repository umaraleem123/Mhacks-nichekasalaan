"""MediaPipe Hand Landmarker wrapper for ASL recognition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# MediaPipe hand landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_TIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
FINGER_PIPS = (THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP)
FINGER_MCPS = (THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
FINGER_NAMES = ("Thumb", "Index", "Middle", "Ring", "Pinky")

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


@dataclass(frozen=True)
class HandLandmarks:
    """Normalized hand landmarks plus image-space helpers."""

    handedness: str
    points: Tuple[Tuple[float, float, float], ...]  # (x, y, z) normalized
    width: int
    height: int

    def xyz(self, idx: int) -> Tuple[float, float, float]:
        return self.points[idx]

    def xy(self, idx: int) -> Tuple[float, float]:
        x, y, _ = self.points[idx]
        return x, y

    def pixel(self, idx: int) -> Tuple[int, int]:
        x, y = self.xy(idx)
        return int(x * self.width), int(y * self.height)

    def distance(self, a: int, b: int) -> float:
        ax, ay, az = self.points[a]
        bx, by, bz = self.points[b]
        return float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2))

    def distance2d(self, a: int, b: int) -> float:
        ax, ay = self.xy(a)
        bx, by = self.xy(b)
        return float(np.hypot(ax - bx, ay - by))


class HandTracker:
    """Detect hand landmarks from BGR webcam frames or still images."""

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        *,
        max_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        still_image: bool = False,
    ) -> None:
        path = Path(model_path) if model_path else DEFAULT_MODEL
        if not path.is_file():
            raise FileNotFoundError(
                f"Hand landmarker model not found at {path}. "
                "Run: python scripts/download_model.py"
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

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def detect(self, bgr_frame: np.ndarray) -> List[HandLandmarks]:
        if bgr_frame is None or bgr_frame.size == 0:
            return []

        rgb = bgr_frame[:, :, ::-1].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self._still_image:
            result = self._landmarker.detect(mp_image)
        else:
            self._timestamp_ms += 33
            result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        return self._to_hands(result, bgr_frame.shape[1], bgr_frame.shape[0])

    @staticmethod
    def _to_hands(result: Any, width: int, height: int) -> List[HandLandmarks]:
        outputs: List[HandLandmarks] = []
        hand_landmarks = result.hand_landmarks or []
        handedness_list = result.handedness or []

        for idx, landmarks in enumerate(hand_landmarks):
            label = "Unknown"
            if idx < len(handedness_list) and handedness_list[idx]:
                label = handedness_list[idx][0].category_name

            points = tuple((lm.x, lm.y, lm.z) for lm in landmarks)
            outputs.append(
                HandLandmarks(
                    handedness=label,
                    points=points,
                    width=width,
                    height=height,
                )
            )
        return outputs


def finger_extended(hand: HandLandmarks, finger_i: int) -> bool:
    """Return True if finger_i (0=thumb … 4=pinky) looks extended."""
    tip = FINGER_TIPS[finger_i]
    pip = FINGER_PIPS[finger_i]
    mcp = FINGER_MCPS[finger_i]
    wrist = WRIST

    if finger_i == 0:
        # Thumb: compare tip–wrist vs MCP–wrist, plus sideways open vs IP.
        tip_dist = hand.distance2d(tip, wrist)
        mcp_dist = hand.distance2d(mcp, wrist)
        tip_x, _ = hand.xy(tip)
        ip_x, _ = hand.xy(pip)
        sideways = tip_x < ip_x if hand.handedness.lower().startswith("right") else tip_x > ip_x
        return tip_dist > mcp_dist * 1.05 and sideways

    # Non-thumb: tip above PIP in image space AND tip farther from wrist than PIP.
    tip_y = hand.xy(tip)[1]
    pip_y = hand.xy(pip)[1]
    tip_dist = hand.distance2d(tip, wrist)
    pip_dist = hand.distance2d(pip, wrist)
    return tip_y < pip_y - 0.02 and tip_dist > pip_dist * 1.05


def finger_states(hand: HandLandmarks) -> Tuple[bool, bool, bool, bool, bool]:
    return tuple(finger_extended(hand, i) for i in range(5))  # type: ignore[return-value]


def landmark_sequence_from_points(
    points: Sequence[Tuple[float, float, float]],
    *,
    handedness: str = "Right",
    width: int = 640,
    height: int = 480,
) -> HandLandmarks:
    """Helper for unit tests: build HandLandmarks from raw point tuples."""
    if len(points) != 21:
        raise ValueError("Expected 21 landmarks")
    return HandLandmarks(
        handedness=handedness,
        points=tuple(points),
        width=width,
        height=height,
    )
