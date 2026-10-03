"""Hand detection and landmark tracking built on MediaPipe Hands.

Uses the legacy `mp.solutions.hands` API, which is why mediapipe is pinned to
0.10.21 in requirements.txt. Newer releases removed `mp.solutions`, and their
Tasks API crashes the process on Apple Silicon inside Metal. See the README.

Kept independent of the application loop so the same tracker can feed ASL
recognition in a later milestone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

_mp_hands = mp.solutions.hands
_mp_drawing = mp.solutions.drawing_utils
_mp_drawing_styles = mp.solutions.drawing_styles


class HandTrackerError(RuntimeError):
    """MediaPipe could not be initialized or used."""


@dataclass(frozen=True)
class DetectedHand:
    """A single hand found in one frame."""

    label: str  # "Left" or "Right"
    confidence: float
    landmarks: Any  # MediaPipe NormalizedLandmarkList of 21 points

    def points(self) -> list[tuple[float, float, float]]:
        """Landmarks as (x, y, z), with x and y normalized to [0, 1]."""
        return [(lm.x, lm.y, lm.z) for lm in self.landmarks.landmark]


class HandTracker:
    """Detects up to `max_hands` hands per frame and draws their landmarks.

    Usable as a context manager, which releases MediaPipe on exit:

        with HandTracker() as tracker:
            hands = tracker.process(frame)
    """

    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
    ) -> None:
        try:
            self._hands = _mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_hands,
                model_complexity=0,  # lightest model; enough for landmarks in real time
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
            )
        except Exception as exc:
            raise HandTrackerError(
                "Failed to initialize MediaPipe Hands. Check that mediapipe "
                "0.10.21 is installed in the active virtual environment "
                "(python -m pip install -r requirements.txt)."
            ) from exc
        self._closed = False

    def process(self, frame_bgr: np.ndarray) -> list[DetectedHand]:
        """Detect hands in one BGR frame from OpenCV.

        Pass the already-mirrored (selfie view) frame: MediaPipe assumes a
        mirrored image when labeling a hand Left or Right, so flipping first is
        what makes the labels match the user's actual hands.
        """
        if self._closed:
            raise HandTrackerError("HandTracker has already been closed.")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            results = self._hands.process(frame_rgb)
        except Exception as exc:
            raise HandTrackerError(f"MediaPipe failed to process a frame: {exc}") from exc

        if not results.multi_hand_landmarks:
            return []

        handedness = results.multi_handedness or []
        detected: list[DetectedHand] = []
        for index, landmarks in enumerate(results.multi_hand_landmarks):
            label, confidence = "Unknown", 0.0
            if index < len(handedness) and handedness[index].classification:
                classification = handedness[index].classification[0]
                label, confidence = classification.label, classification.score
            detected.append(DetectedHand(label, confidence, landmarks))
        return detected

    def draw(self, frame_bgr: np.ndarray, hands: list[DetectedHand]) -> np.ndarray:
        """Draw landmarks and connections onto the frame, in place."""
        for hand in hands:
            _mp_drawing.draw_landmarks(
                frame_bgr,
                hand.landmarks,
                _mp_hands.HAND_CONNECTIONS,
                _mp_drawing_styles.get_default_hand_landmarks_style(),
                _mp_drawing_styles.get_default_hand_connections_style(),
            )
        return frame_bgr

    def close(self) -> None:
        """Release MediaPipe resources. Safe to call more than once."""
        if not self._closed:
            self._hands.close()
            self._closed = True

    def __enter__(self) -> HandTracker:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
