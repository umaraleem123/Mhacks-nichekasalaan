"""Pose + both-hands tracking built on MediaPipe Holistic.

The pretrained ASL Citizen model needs 75 landmarks (33 pose, 21 left hand,
21 right hand). MediaPipe Hands only yields 21 points, so this tracker is a
separate module and does not replace `hand_tracker.py`.

Uses the legacy `mp.solutions.holistic` API on the pinned mediapipe 0.10.21
release. Face landmarks are computed by Holistic and ignored here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

_mp_holistic = mp.solutions.holistic
_mp_drawing = mp.solutions.drawing_utils
_mp_drawing_styles = mp.solutions.drawing_styles


class HolisticTrackerError(RuntimeError):
    """MediaPipe Holistic could not be initialized or used."""


@dataclass(frozen=True)
class HolisticLandmarks:
    """One frame of pose and hand landmarks, or None when that part is missing."""

    pose: Any | None
    left_hand: Any | None
    right_hand: Any | None
    raw: Any


class HolisticTracker:
    """Runs MediaPipe Holistic on successive webcam frames.

    Usable as a context manager, which releases MediaPipe on exit:

        with HolisticTracker() as tracker:
            landmarks = tracker.process(frame)
    """

    def __init__(
        self,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        model_complexity: int = 0,
    ) -> None:
        try:
            self._holistic = _mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=model_complexity,
                smooth_landmarks=True,
                enable_segmentation=False,
                refine_face_landmarks=False,
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
            )
        except Exception as exc:
            raise HolisticTrackerError(
                "Failed to initialize MediaPipe Holistic. Check that mediapipe "
                "0.10.21 is installed in the active virtual environment "
                "(python -m pip install -r requirements.txt)."
            ) from exc
        self._closed = False

    def process(self, frame_bgr: np.ndarray) -> HolisticLandmarks:
        """Detect pose and hands in one BGR frame from OpenCV.

        Pass the already-mirrored (selfie view) frame so Holistic handedness
        matches what the user sees.
        """
        if self._closed:
            raise HolisticTrackerError("HolisticTracker has already been closed.")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            results = self._holistic.process(frame_rgb)
        except Exception as exc:
            raise HolisticTrackerError(
                f"MediaPipe Holistic failed to process a frame: {exc}"
            ) from exc

        return HolisticLandmarks(
            pose=results.pose_landmarks,
            left_hand=results.left_hand_landmarks,
            right_hand=results.right_hand_landmarks,
            raw=results,
        )

    def draw(self, frame_bgr: np.ndarray, landmarks: HolisticLandmarks) -> np.ndarray:
        """Draw pose and hand skeletons onto the frame, in place."""
        if landmarks.pose is not None:
            _mp_drawing.draw_landmarks(
                frame_bgr,
                landmarks.pose,
                _mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=_mp_drawing_styles.get_default_pose_landmarks_style(),
            )
        if landmarks.left_hand is not None:
            _mp_drawing.draw_landmarks(
                frame_bgr,
                landmarks.left_hand,
                _mp_holistic.HAND_CONNECTIONS,
                _mp_drawing_styles.get_default_hand_landmarks_style(),
                _mp_drawing_styles.get_default_hand_connections_style(),
            )
        if landmarks.right_hand is not None:
            _mp_drawing.draw_landmarks(
                frame_bgr,
                landmarks.right_hand,
                _mp_holistic.HAND_CONNECTIONS,
                _mp_drawing_styles.get_default_hand_landmarks_style(),
                _mp_drawing_styles.get_default_hand_connections_style(),
            )
        return frame_bgr

    def close(self) -> None:
        """Release MediaPipe resources. Safe to call more than once."""
        if not self._closed:
            self._holistic.close()
            self._closed = True

    def __enter__(self) -> HolisticTracker:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
