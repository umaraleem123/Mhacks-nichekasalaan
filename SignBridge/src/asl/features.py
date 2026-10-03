"""Turn MediaPipe hand landmarks into a fixed-length feature vector.

Feature representation
----------------------
Every hand becomes exactly 63 floats: the 21 MediaPipe landmarks as
(x, y, z) triples in landmark order, after three normalization steps.

1. Mirror. Left hands are flipped across the x axis, so the same sign made
   with either hand lands in the same region of feature space. One training
   set covers both left- and right-handed signers.
2. Translate. The wrist (landmark 0) is moved to the origin. This removes
   where the hand sits in the frame, so a sign reads the same in any corner.
3. Scale. Every coordinate is divided by the distance from the wrist to the
   furthest landmark. This removes apparent hand size, so distance from the
   camera and the size of the signer's hand stop mattering.

Raw pixel coordinates are never used.

Rotation is deliberately left alone. Hand orientation is part of what
distinguishes one sign from another, so normalizing it away would make signs
that differ only by rotation impossible to tell apart.

This module has no MediaPipe or OpenCV import on purpose: the trainer loads
features from CSV and should not need the vision stack installed.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

LANDMARK_COUNT = 21
COORDS_PER_LANDMARK = 3
FEATURE_DIM = LANDMARK_COUNT * COORDS_PER_LANDMARK  # 63

# Guards against dividing by zero if every landmark collapses onto the wrist.
_MIN_SCALE = 1e-6


class FeatureError(ValueError):
    """Landmarks could not be converted into a feature vector."""


def feature_names() -> list[str]:
    """Column names matching the layout of `extract_features`."""
    return [
        f"lm{index:02d}_{axis}"
        for index in range(LANDMARK_COUNT)
        for axis in ("x", "y", "z")
    ]


def landmarks_to_points(landmarks: Any) -> np.ndarray:
    """Read a MediaPipe landmark list into an array of shape (21, 3).

    Accepts either a MediaPipe `NormalizedLandmarkList` or any sequence of
    objects exposing `.x`, `.y`, and `.z`, which keeps this module decoupled
    from the vision layer.
    """
    raw: Iterable[Any] = getattr(landmarks, "landmark", landmarks)
    points = np.array([(lm.x, lm.y, lm.z) for lm in raw], dtype=np.float32)

    if points.shape != (LANDMARK_COUNT, COORDS_PER_LANDMARK):
        raise FeatureError(
            f"Expected {LANDMARK_COUNT} landmarks of "
            f"{COORDS_PER_LANDMARK} coordinates, got shape {points.shape}."
        )
    return points


def normalize_points(points: np.ndarray, is_left_hand: bool = False) -> np.ndarray:
    """Apply the mirror, translate, and scale steps described above."""
    normalized = points.astype(np.float32, copy=True)

    if is_left_hand:
        normalized[:, 0] *= -1.0

    normalized -= normalized[0]  # wrist to origin

    scale = float(np.max(np.linalg.norm(normalized, axis=1)))
    return normalized / max(scale, _MIN_SCALE)


def extract_features(hand: Any) -> np.ndarray:
    """Build the 63-value feature vector for one detected hand.

    `hand` is a `DetectedHand` from `src.vision.hand_tracker`, or anything with
    `.landmarks` and a `.label` of "Left" or "Right".
    """
    points = landmarks_to_points(hand.landmarks)
    is_left = str(getattr(hand, "label", "")).strip().lower() == "left"
    return normalize_points(points, is_left).flatten()


def validate_feature_vector(features: Sequence[float]) -> np.ndarray:
    """Check a loaded or incoming vector has the expected length."""
    array = np.asarray(features, dtype=np.float32).reshape(-1)
    if array.size != FEATURE_DIM:
        raise FeatureError(
            f"Expected {FEATURE_DIM} features, got {array.size}. The data was "
            "probably produced by a different feature version; recollect it."
        )
    return array
