"""UNUSED / EXPERIMENTAL — not part of the live SignBridge demo.

Convert MediaPipe Holistic landmarks into the pretrained model's features.

The ASL Citizen BiGRU checkpoint expects one frame as 450 floats:

    75 landmarks × 3 coordinates = 225 position features
    plus 225 frame-to-frame velocity features

The 75 landmarks are 33 pose, 21 left hand, and 21 right hand, in that
order. Missing parts are zeros. This is not the 63-d Hands-only vector in
`features.py`, which the old local classifier still uses.

A clip of any length is interpolated to exactly 200 temporal positions
before velocity is computed, matching the dataset the model was trained on.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

POSE_LANDMARK_COUNT = 33
HAND_LANDMARK_COUNT = 21
LANDMARK_COUNT = POSE_LANDMARK_COUNT + HAND_LANDMARK_COUNT + HAND_LANDMARK_COUNT  # 75
COORDS_PER_LANDMARK = 3
POSITION_DIM = LANDMARK_COUNT * COORDS_PER_LANDMARK  # 225
VELOCITY_DIM = POSITION_DIM  # 225
FEATURE_DIM = POSITION_DIM + VELOCITY_DIM  # 450
SEQ_LEN = 200

# Pose landmark indices in MediaPipe Holistic / BlazePose.
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP = 23
_RIGHT_HIP = 24

# Guards against a collapsed skeleton (camera too close, or no torso).
_MIN_SCALE = 1e-3

LEFT_HAND_SLICE = slice(POSE_LANDMARK_COUNT, POSE_LANDMARK_COUNT + HAND_LANDMARK_COUNT)
RIGHT_HAND_SLICE = slice(
    POSE_LANDMARK_COUNT + HAND_LANDMARK_COUNT,
    LANDMARK_COUNT,
)


class PretrainedFeatureError(ValueError):
    """Landmarks could not be converted into the 450-d pretrained vector."""


def landmark_list_to_array(landmarks: Any | None, count: int) -> np.ndarray:
    """Read a MediaPipe landmark list into `(count, 3)`, or zeros if missing."""
    points = np.zeros((count, COORDS_PER_LANDMARK), dtype=np.float32)
    if landmarks is None:
        return points

    raw: Iterable[Any] = getattr(landmarks, "landmark", landmarks)
    for index, landmark in enumerate(raw):
        if index >= count:
            break
        points[index, 0] = float(landmark.x)
        points[index, 1] = float(landmark.y)
        points[index, 2] = float(landmark.z)
    return points


def extract_75_landmarks(
    pose: Any | None = None,
    left_hand: Any | None = None,
    right_hand: Any | None = None,
) -> np.ndarray:
    """Stack pose, left hand, and right hand into a `(75, 3)` array.

    Missing groups stay all zeros so the feature width never changes.
    """
    pose_points = landmark_list_to_array(pose, POSE_LANDMARK_COUNT)
    left_points = landmark_list_to_array(left_hand, HAND_LANDMARK_COUNT)
    right_points = landmark_list_to_array(right_hand, HAND_LANDMARK_COUNT)
    return np.concatenate([pose_points, left_points, right_points], axis=0)


def normalize_landmarks(
    points: np.ndarray,
    pose_present: bool = True,
    left_present: bool = True,
    right_present: bool = True,
) -> np.ndarray:
    """Translate to the hip midpoint and scale by shoulder width.

    Hands that were not detected stay zero; they are not shifted onto the
    origin, which would look like a hand sitting on the hip.
    """
    if points.shape != (LANDMARK_COUNT, COORDS_PER_LANDMARK):
        raise PretrainedFeatureError(
            f"Expected landmarks of shape ({LANDMARK_COUNT}, "
            f"{COORDS_PER_LANDMARK}), got {points.shape}."
        )

    normalized = np.asarray(points, dtype=np.float32).copy()
    if not pose_present:
        return normalized

    left_shoulder = normalized[_LEFT_SHOULDER]
    right_shoulder = normalized[_RIGHT_SHOULDER]
    origin = 0.5 * (normalized[_LEFT_HIP] + normalized[_RIGHT_HIP])
    if float(np.linalg.norm(origin)) < _MIN_SCALE:
        origin = 0.5 * (left_shoulder + right_shoulder)

    scale = float(np.linalg.norm(left_shoulder - right_shoulder))
    scale = max(scale, _MIN_SCALE)

    mask = np.zeros(LANDMARK_COUNT, dtype=bool)
    mask[:POSE_LANDMARK_COUNT] = True
    if left_present:
        mask[LEFT_HAND_SLICE] = True
    if right_present:
        mask[RIGHT_HAND_SLICE] = True

    normalized[mask] = (normalized[mask] - origin) / scale
    return normalized


def position_vector(
    pose: Any | None = None,
    left_hand: Any | None = None,
    right_hand: Any | None = None,
) -> np.ndarray:
    """One frame as 225 position features."""
    points = extract_75_landmarks(pose, left_hand, right_hand)
    normalized = normalize_landmarks(
        points,
        pose_present=pose is not None,
        left_present=left_hand is not None,
        right_present=right_hand is not None,
    )
    return normalized.reshape(POSITION_DIM)


def position_velocity(positions: np.ndarray) -> np.ndarray:
    """Frame-to-frame displacement. The first frame is zeros."""
    array = np.asarray(positions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != POSITION_DIM:
        raise PretrainedFeatureError(
            f"Expected positions of shape (T, {POSITION_DIM}), got {array.shape}."
        )
    velocity = np.zeros_like(array)
    if array.shape[0] > 1:
        velocity[1:] = array[1:] - array[:-1]
    return velocity


def sequence_features(positions: np.ndarray) -> np.ndarray:
    """Concatenate position and velocity to `(T, 450)`."""
    velocity = position_velocity(positions)
    return np.concatenate([positions.astype(np.float32), velocity], axis=-1)


def interpolate_positions(positions: np.ndarray, length: int = SEQ_LEN) -> np.ndarray:
    """Resample a variable-length clip onto `length` evenly spaced frames.

    Linear interpolation along time, independently per feature. A single
    frame is repeated. An empty clip becomes zeros. This is how a live
    buffer of any duration becomes the model's fixed 200-step input
    without asking the user to hold a sign for 200 camera frames.
    """
    if length <= 0:
        raise PretrainedFeatureError("Interpolation length must be positive.")

    array = np.asarray(positions, dtype=np.float32)
    if array.size == 0:
        return np.zeros((length, POSITION_DIM), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != POSITION_DIM:
        raise PretrainedFeatureError(
            f"Expected positions of shape (T, {POSITION_DIM}), got {array.shape}."
        )

    frame_count = array.shape[0]
    if frame_count == 1:
        return np.repeat(array, length, axis=0)
    if frame_count == length:
        return array.copy()

    source = np.linspace(0.0, 1.0, frame_count, dtype=np.float64)
    target = np.linspace(0.0, 1.0, length, dtype=np.float64)
    index = np.interp(target, source, np.arange(frame_count, dtype=np.float64))
    low = np.floor(index).astype(np.int64)
    high = np.minimum(low + 1, frame_count - 1)
    weight = (index - low).astype(np.float32)[:, None]
    return (array[low] * (1.0 - weight) + array[high] * weight).astype(np.float32)


def pad_or_interpolate(positions: np.ndarray, length: int = SEQ_LEN) -> np.ndarray:
    """Public name for the 200-frame temporal resize used by the recognizer."""
    return interpolate_positions(positions, length=length)


def model_input_from_positions(
    positions: Sequence[Sequence[float]] | np.ndarray,
    length: int = SEQ_LEN,
) -> np.ndarray:
    """Turn a rolling position buffer into one `(length, 450)` model clip."""
    array = np.asarray(positions, dtype=np.float32)
    resampled = interpolate_positions(array, length=length)
    features = sequence_features(resampled)
    if features.shape != (length, FEATURE_DIM):
        raise PretrainedFeatureError(
            f"Expected features of shape ({length}, {FEATURE_DIM}), "
            f"got {features.shape}."
        )
    return features
