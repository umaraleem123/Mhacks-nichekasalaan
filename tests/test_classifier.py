"""Unit tests for ASL letter heuristics (no webcam required)."""

from __future__ import annotations

from asl_recognition.classifier import classify_asl
from asl_recognition.landmarks import landmark_sequence_from_points


def _base_hand() -> list[tuple[float, float, float]]:
    """Neutral roughly-open right hand template in normalized coords."""
    # Index order matches MediaPipe 0..20
    return [
        (0.50, 0.80, 0.0),  # 0 wrist
        (0.42, 0.72, 0.0),  # 1 thumb cmc
        (0.36, 0.64, 0.0),  # 2 thumb mcp
        (0.30, 0.56, 0.0),  # 3 thumb ip
        (0.24, 0.50, 0.0),  # 4 thumb tip
        (0.46, 0.60, 0.0),  # 5 index mcp
        (0.45, 0.48, 0.0),  # 6 index pip
        (0.44, 0.38, 0.0),  # 7 index dip
        (0.43, 0.28, 0.0),  # 8 index tip
        (0.52, 0.60, 0.0),  # 9 middle mcp
        (0.52, 0.46, 0.0),  # 10
        (0.52, 0.36, 0.0),  # 11
        (0.52, 0.26, 0.0),  # 12
        (0.58, 0.62, 0.0),  # 13 ring mcp
        (0.59, 0.50, 0.0),  # 14
        (0.60, 0.40, 0.0),  # 15
        (0.61, 0.30, 0.0),  # 16
        (0.64, 0.66, 0.0),  # 17 pinky mcp
        (0.66, 0.56, 0.0),  # 18
        (0.67, 0.48, 0.0),  # 19
        (0.68, 0.40, 0.0),  # 20
    ]


def _curl_finger(pts: list[tuple[float, float, float]], mcp: int, tip: int) -> None:
    """Pull tip down toward MCP to simulate a curled finger."""
    mx, my, mz = pts[mcp]
    for i in range(mcp + 1, tip + 1):
        t = (i - mcp) / (tip - mcp)
        pts[i] = (mx + 0.01 * t, my + 0.06 * t, mz)


def test_letter_l() -> None:
    pts = _base_hand()
    # Curl middle/ring/pinky; keep thumb + index extended.
    _curl_finger(pts, 9, 12)
    _curl_finger(pts, 13, 16)
    _curl_finger(pts, 17, 20)
    hand = landmark_sequence_from_points(pts, handedness="Right")
    pred = classify_asl(hand)
    assert pred.letter == "L"
    assert pred.confidence >= 0.55


def test_letter_v() -> None:
    pts = _base_hand()
    # Thumb tucked-ish, ring+pinky curled, index+middle up and apart.
    pts[4] = (0.40, 0.62, 0.0)  # thumb tip closer / less sideways
    pts[3] = (0.38, 0.60, 0.0)
    _curl_finger(pts, 13, 16)
    _curl_finger(pts, 17, 20)
    # Spread V
    pts[8] = (0.38, 0.22, 0.0)
    pts[12] = (0.58, 0.22, 0.0)
    hand = landmark_sequence_from_points(pts, handedness="Right")
    pred = classify_asl(hand)
    assert pred.letter == "V"
    assert pred.confidence >= 0.55


def test_letter_y() -> None:
    pts = _base_hand()
    _curl_finger(pts, 5, 8)
    _curl_finger(pts, 9, 12)
    _curl_finger(pts, 13, 16)
    # Pinky up, thumb out
    pts[20] = (0.72, 0.28, 0.0)
    pts[19] = (0.70, 0.36, 0.0)
    pts[18] = (0.68, 0.44, 0.0)
    pts[4] = (0.20, 0.48, 0.0)
    hand = landmark_sequence_from_points(pts, handedness="Right")
    pred = classify_asl(hand)
    assert pred.letter == "Y"


def test_letter_i() -> None:
    pts = _base_hand()
    _curl_finger(pts, 5, 8)
    _curl_finger(pts, 9, 12)
    _curl_finger(pts, 13, 16)
    pts[4] = (0.40, 0.62, 0.0)
    pts[3] = (0.38, 0.60, 0.0)
    pts[20] = (0.72, 0.28, 0.0)
    pts[19] = (0.70, 0.36, 0.0)
    pts[18] = (0.68, 0.44, 0.0)
    hand = landmark_sequence_from_points(pts, handedness="Right")
    pred = classify_asl(hand)
    assert pred.letter == "I"


def test_letter_w() -> None:
    pts = _base_hand()
    pts[4] = (0.40, 0.62, 0.0)
    pts[3] = (0.38, 0.60, 0.0)
    _curl_finger(pts, 17, 20)
    hand = landmark_sequence_from_points(pts, handedness="Right")
    pred = classify_asl(hand)
    assert pred.letter == "W"


def test_candidates_sorted() -> None:
    pts = _base_hand()
    _curl_finger(pts, 9, 12)
    _curl_finger(pts, 13, 16)
    _curl_finger(pts, 17, 20)
    hand = landmark_sequence_from_points(pts, handedness="Right")
    pred = classify_asl(hand)
    scores = [s for _, s in pred.candidates]
    assert scores == sorted(scores, reverse=True)
