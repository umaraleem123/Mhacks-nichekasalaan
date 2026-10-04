"""Two-hand temporal features for the local sequence classifier.

Each frame is 252 floats:

    left hand  21 × 3 = 63
    right hand 21 × 3 = 63
    -------------------------
    position              126
    velocity (Δ position) 126
    -------------------------
    per-frame vector      252

Missing hands are zeros. Frames are never classified on their own; a clip of
these vectors is one training example / one live inference window.

This module does not import MediaPipe or OpenCV. The 63-d single-hand vector
in `features.py` is unchanged and is not used by the live sequence path.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from src.asl.features import COORDS_PER_LANDMARK, LANDMARK_COUNT, landmarks_to_points

HAND_DIM = LANDMARK_COUNT * COORDS_PER_LANDMARK  # 63
POSITION_DIM = HAND_DIM * 2  # 126
VELOCITY_DIM = POSITION_DIM  # 126
FEATURE_DIM = POSITION_DIM + VELOCITY_DIM  # 252

WRIST_INDEX = 0
# Middle-finger MCP. More stable than "furthest landmark" when a finger flicks.
SCALE_LANDMARK_INDEX = 9
_MIN_SCALE = 1e-6


class SequenceFeatureError(ValueError):
    """Landmarks could not be converted into a sequence feature vector."""


def _as_points(landmarks: Any) -> np.ndarray:
    return np.asarray(landmarks_to_points(landmarks), dtype=np.float32)


def normalize_hand(points: np.ndarray) -> np.ndarray:
    """Wrist-center and scale one hand. Shape `(21, 3)` in, same out.

    Scale is the wrist-to-middle-MCP distance, falling back to the largest
    landmark radius if that bone is degenerate. Relative finger geometry is
    kept; absolute webcam position is not.
    """
    array = np.asarray(points, dtype=np.float32)
    if array.shape != (LANDMARK_COUNT, COORDS_PER_LANDMARK):
        raise SequenceFeatureError(
            f"Expected one hand of shape ({LANDMARK_COUNT}, "
            f"{COORDS_PER_LANDMARK}), got {array.shape}."
        )
    centered = array - array[WRIST_INDEX]
    scale = float(np.linalg.norm(centered[SCALE_LANDMARK_INDEX]))
    if scale < _MIN_SCALE:
        scale = float(np.max(np.linalg.norm(centered, axis=1)))
    return centered / max(scale, _MIN_SCALE)


def empty_hand() -> np.ndarray:
    return np.zeros((LANDMARK_COUNT, COORDS_PER_LANDMARK), dtype=np.float32)


def two_hand_positions(
    left: np.ndarray | None = None,
    right: np.ndarray | None = None,
) -> np.ndarray:
    """126-d position vector: normalized left, then normalized right."""
    left_vec = (
        empty_hand() if left is None else normalize_hand(left)
    ).reshape(HAND_DIM)
    right_vec = (
        empty_hand() if right is None else normalize_hand(right)
    ).reshape(HAND_DIM)
    return np.concatenate([left_vec, right_vec], axis=0).astype(np.float32)


def positions_from_hands(hands: Sequence[Any] | None) -> np.ndarray:
    """Build the 126-d vector from a list of `DetectedHand`-like objects."""
    left_points: np.ndarray | None = None
    right_points: np.ndarray | None = None
    extras: list[np.ndarray] = []

    for hand in hands or ():
        label = str(getattr(hand, "label", "")).strip().lower()
        landmarks = getattr(hand, "landmarks", hand)
        points = _as_points(landmarks)
        if label == "left" and left_points is None:
            left_points = points
        elif label == "right" and right_points is None:
            right_points = points
        else:
            extras.append(points)

    # A single unlabeled hand is treated as the right (dominant) slot.
    if left_points is None and right_points is None and extras:
        right_points = extras[0]
    elif right_points is None and extras:
        right_points = extras[0]
    elif left_points is None and extras:
        left_points = extras[0]

    return two_hand_positions(left_points, right_points)


def position_velocity(
    current: np.ndarray,
    previous: np.ndarray | None,
) -> np.ndarray:
    """126-d displacement. Zeros on the first frame of a clip."""
    now = np.asarray(current, dtype=np.float32).reshape(POSITION_DIM)
    if previous is None:
        return np.zeros(POSITION_DIM, dtype=np.float32)
    prior = np.asarray(previous, dtype=np.float32).reshape(POSITION_DIM)
    return now - prior


def frame_features(
    positions: np.ndarray,
    previous_positions: np.ndarray | None,
) -> np.ndarray:
    """Concatenate position and velocity to 252-d."""
    pos = np.asarray(positions, dtype=np.float32).reshape(POSITION_DIM)
    vel = position_velocity(pos, previous_positions)
    return np.concatenate([pos, vel], axis=0)


def features_from_position_sequence(positions: np.ndarray) -> np.ndarray:
    """`(T, 126)` positions → `(T, 252)` position+velocity."""
    array = np.asarray(positions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != POSITION_DIM:
        raise SequenceFeatureError(
            f"Expected positions of shape (T, {POSITION_DIM}), got {array.shape}."
        )
    if array.shape[0] == 0:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    velocity = np.zeros_like(array)
    if array.shape[0] > 1:
        velocity[1:] = array[1:] - array[:-1]
    return np.concatenate([array, velocity], axis=1)


def _as_position_clip(positions: np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(positions, dtype=np.float32)
    if array.ndim == 1:
        if array.size != POSITION_DIM:
            raise SequenceFeatureError(
                f"Expected a {POSITION_DIM}-d position vector, got {array.shape}."
            )
        return array.reshape(1, POSITION_DIM), True
    if array.ndim != 2 or array.shape[1] != POSITION_DIM:
        raise SequenceFeatureError(
            f"Expected positions of shape (T, {POSITION_DIM}), got {array.shape}."
        )
    return array, False


def _mirror_hand_x(hand: np.ndarray) -> np.ndarray:
    """Apply `mirrored_x = 1.0 - x`, then keep the wrist X where it was.

    Image-space left/right flip is `1 - x`. Our stored hands are already
    wrist-centered (wrist X is 0), so restoring the wrist after that flip
    is `x' = -x` and, importantly, an all-zero missing hand stays zero
    instead of turning into a phantom hand at x=1.
    """
    out = np.asarray(hand, dtype=np.float32).copy()
    wrist_x = out[..., WRIST_INDEX, 0].copy()
    out[..., 0] = 1.0 - out[..., 0]
    shift = out[..., WRIST_INDEX, 0] - wrist_x
    out[..., 0] = out[..., 0] - np.expand_dims(shift, axis=-1)
    return out


def mirror_positions(positions: np.ndarray) -> np.ndarray:
    """Lateral-flip a 126-d position clip and swap the left/right blocks.

    Order: mirror X on each hand, swap hands. Velocity is not computed here.
    """
    clip, squeeze = _as_position_clip(positions)
    frames = clip.shape[0]
    left = clip[:, :HAND_DIM].reshape(frames, LANDMARK_COUNT, COORDS_PER_LANDMARK)
    right = clip[:, HAND_DIM:].reshape(frames, LANDMARK_COUNT, COORDS_PER_LANDMARK)
    mirrored_left = _mirror_hand_x(left)
    mirrored_right = _mirror_hand_x(right)
    swapped = np.concatenate(
        [
            mirrored_right.reshape(frames, HAND_DIM),
            mirrored_left.reshape(frames, HAND_DIM),
        ],
        axis=1,
    )
    if squeeze:
        return swapped[0]
    return swapped


def mirror_sequence_features(features: np.ndarray) -> np.ndarray:
    """Mirror a `(T, 252)` clip: flip/swap positions, then recompute velocity.

    Labels are not involved. Length is unchanged.
    """
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != FEATURE_DIM:
        raise SequenceFeatureError(
            f"Expected features of shape (T, {FEATURE_DIM}), got {array.shape}."
        )
    if array.shape[0] == 0:
        return array.copy()
    mirrored_positions = mirror_positions(array[:, :POSITION_DIM])
    return features_from_position_sequence(mirrored_positions)


def motion_energy(features: np.ndarray) -> float:
    """Mean absolute velocity. Near zero when the hands are still."""
    array = np.asarray(features, dtype=np.float32)
    if array.size == 0:
        return 0.0
    if array.ndim == 1:
        velocity = array[POSITION_DIM:]
    else:
        velocity = array[:, POSITION_DIM:]
    return float(np.mean(np.abs(velocity)))


def landmark_list_to_points(landmarks: Any | None) -> np.ndarray | None:
    """Convenience for tests: MediaPipe-like lists or None."""
    if landmarks is None:
        return None
    raw: Iterable[Any] = getattr(landmarks, "landmark", landmarks)
    points = np.array([(lm.x, lm.y, lm.z) for lm in raw], dtype=np.float32)
    if points.shape != (LANDMARK_COUNT, COORDS_PER_LANDMARK):
        raise SequenceFeatureError(
            f"Expected {LANDMARK_COUNT} landmarks, got shape {points.shape}."
        )
    return points
