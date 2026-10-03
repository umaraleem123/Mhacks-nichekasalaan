"""Unit tests for raised-finger heuristics (no webcam required)."""

from __future__ import annotations

from types import SimpleNamespace

from finger_detection.detector import FINGER_NAMES, FingerResult, _is_finger_up


def _lm(x: float, y: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y)


def test_index_up_when_tip_above_pip() -> None:
    landmarks = [_lm(0, 0)] * 21
    landmarks[8] = _lm(0.5, 0.2)  # tip
    landmarks[6] = _lm(0.5, 0.4)  # pip
    assert _is_finger_up(landmarks, 8, 6, is_thumb=False, handedness="Right")


def test_index_down_when_tip_below_pip() -> None:
    landmarks = [_lm(0, 0)] * 21
    landmarks[8] = _lm(0.5, 0.5)
    landmarks[6] = _lm(0.5, 0.3)
    assert not _is_finger_up(landmarks, 8, 6, is_thumb=False, handedness="Right")


def test_right_thumb_up_opens_leftward() -> None:
    landmarks = [_lm(0, 0)] * 21
    landmarks[4] = _lm(0.2, 0.5)
    landmarks[3] = _lm(0.4, 0.5)
    assert _is_finger_up(landmarks, 4, 3, is_thumb=True, handedness="Right")


def test_left_thumb_up_opens_rightward() -> None:
    landmarks = [_lm(0, 0)] * 21
    landmarks[4] = _lm(0.6, 0.5)
    landmarks[3] = _lm(0.4, 0.5)
    assert _is_finger_up(landmarks, 4, 3, is_thumb=True, handedness="Left")


def test_finger_result_raised_names() -> None:
    result = FingerResult(
        handedness="Right",
        raised=(True, True, False, False, True),
        tip_pixels=((0, 0), (1, 1), (2, 2), (3, 3), (4, 4)),
        count=3,
    )
    assert result.raised_names == ["Thumb", "Index", "Pinky"]
    assert FINGER_NAMES[2] == "Middle"
