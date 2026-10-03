"""Rule-based ASL alphabet classifier from hand landmarks.

Recognizes common static ASL letter shapes (A–Y, excluding motion letters
J and Z which need a trajectory). Scores are heuristic and intended for
interactive demos — not clinical/production accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .landmarks import (
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    PINKY_TIP,
    RING_MCP,
    RING_TIP,
    THUMB_TIP,
    WRIST,
    HandLandmarks,
    finger_states,
)


@dataclass(frozen=True)
class ASLPrediction:
    """Best-guess ASL letter for one hand pose."""

    letter: Optional[str]
    confidence: float
    handedness: str
    candidates: Tuple[Tuple[str, float], ...]

    @property
    def label(self) -> str:
        if self.letter is None:
            return "?"
        return self.letter


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _score_pattern(
    extended: Tuple[bool, bool, bool, bool, bool],
    want: Tuple[Optional[bool], Optional[bool], Optional[bool], Optional[bool], Optional[bool]],
) -> float:
    """Score how well finger extension matches a desired pattern (None = don't care)."""
    hits = 0
    total = 0
    for got, expected in zip(extended, want):
        if expected is None:
            continue
        total += 1
        if got == expected:
            hits += 1
    if total == 0:
        return 0.0
    return hits / total


def classify_asl(hand: HandLandmarks) -> ASLPrediction:
    """Classify a static ASL letter from a single HandLandmarks sample."""
    ext = finger_states(hand)
    thumb, index, middle, ring, pinky = ext
    n_up = sum(ext)

    tip_thumb_index = hand.distance2d(THUMB_TIP, INDEX_TIP)
    tip_thumb_middle = hand.distance2d(THUMB_TIP, MIDDLE_TIP)
    tip_index_middle = hand.distance2d(INDEX_TIP, MIDDLE_TIP)
    tip_middle_ring = hand.distance2d(MIDDLE_TIP, RING_TIP)
    tip_ring_pinky = hand.distance2d(RING_TIP, PINKY_TIP)

    # Palm scale: wrist → middle MCP
    scale = max(hand.distance2d(WRIST, MIDDLE_MCP), 1e-6)
    tip_thumb_index_n = tip_thumb_index / scale
    tip_thumb_middle_n = tip_thumb_middle / scale
    tip_index_middle_n = tip_index_middle / scale

    # Thumb relative height vs index MCP (useful for A/E/S)
    thumb_y = hand.xy(THUMB_TIP)[1]
    index_mcp_y = hand.xy(INDEX_MCP)[1]
    middle_mcp_y = hand.xy(MIDDLE_MCP)[1]

    scores: Dict[str, float] = {}

    # A: fist, thumb beside (not tucked). Fingers curled, thumb out.
    a = _score_pattern(ext, (True, False, False, False, False))
    if tip_thumb_index_n > 0.25:
        a += 0.15
    scores["A"] = _clamp01(a * 0.85)

    # B: four fingers up, thumb tucked across palm.
    b = _score_pattern(ext, (False, True, True, True, True))
    if tip_thumb_index_n < 0.45:
        b += 0.1
    scores["B"] = _clamp01(b * 0.9)

    # C: curved open hand — no fully extended fingers, tips form an arc, thumb apart.
    c_base = 0.0 if n_up >= 3 else 0.55
    # Tips should be somewhat spaced and thumb away from index tip.
    if 0.35 < tip_thumb_index_n < 1.2 and not index and not pinky:
        c_base += 0.25
    # Fingers not fully curled into fist (index tip not too close to MCP).
    if hand.distance2d(INDEX_TIP, INDEX_MCP) / scale > 0.35:
        c_base += 0.15
    scores["C"] = _clamp01(c_base)

    # D: index up, others down, thumb near middle fingertips.
    d = _score_pattern(ext, (None, True, False, False, False))
    if tip_thumb_middle_n < 0.45:
        d += 0.2
    if not middle and not ring and not pinky:
        d += 0.1
    scores["D"] = _clamp01(d * 0.85)

    # E: all fingers curled, thumb tucked under (thumb tip near index MCP).
    e = _score_pattern(ext, (False, False, False, False, False))
    if tip_thumb_index_n < 0.4 and thumb_y > index_mcp_y - 0.05:
        e += 0.25
    scores["E"] = _clamp01(e * 0.8)

    # F: OK circle (thumb+index tips touch), middle/ring/pinky up.
    f = _score_pattern(ext, (None, False, True, True, True))
    if tip_thumb_index_n < 0.28:
        f += 0.35
    scores["F"] = _clamp01(f * 0.85)

    # G: index pointing sideways, thumb parallel; others curled. Hard without orientation —
    # approximate as index+thumb extended, rest down, index roughly horizontal.
    g = _score_pattern(ext, (True, True, False, False, False))
    idx_dx = abs(hand.xy(INDEX_TIP)[0] - hand.xy(INDEX_MCP)[0])
    idx_dy = abs(hand.xy(INDEX_TIP)[1] - hand.xy(INDEX_MCP)[1])
    if idx_dx > idx_dy:
        g += 0.25
    scores["G"] = _clamp01(g * 0.75)

    # H: index+middle extended together (often sideways).
    h = _score_pattern(ext, (None, True, True, False, False))
    if tip_index_middle_n < 0.28:
        h += 0.2
    scores["H"] = _clamp01(h * 0.8)

    # I: pinky only.
    scores["I"] = _clamp01(_score_pattern(ext, (False, False, False, False, True)) * 0.95)

    # K: index+middle up, thumb touches middle MCP area.
    k = _score_pattern(ext, (True, True, True, False, False))
    if tip_thumb_middle_n < 0.45:
        k += 0.15
    scores["K"] = _clamp01(k * 0.8)

    # L: index up + thumb out (L shape), others down.
    l = _score_pattern(ext, (True, True, False, False, False))
    # Prefer more vertical index than G.
    if idx_dy >= idx_dx:
        l += 0.2
    if tip_thumb_index_n > 0.35:
        l += 0.1
    scores["L"] = _clamp01(l * 0.9)

    # M: thumb under three fingers (index/middle/ring over thumb) — all "down".
    m = _score_pattern(ext, (False, False, False, False, False))
    if hand.xy(THUMB_TIP)[0] < hand.xy(RING_MCP)[0] if hand.handedness.lower().startswith("right") else hand.xy(THUMB_TIP)[0] > hand.xy(RING_MCP)[0]:
        m += 0.1
    scores["M"] = _clamp01(m * 0.55)

    # N: similar to M with two fingers over thumb.
    scores["N"] = _clamp01(_score_pattern(ext, (False, False, False, False, False)) * 0.5)

    # O: fingertips gather toward thumb tip (circle).
    o = 0.0
    gather = (
        tip_thumb_index_n
        + tip_thumb_middle_n
        + hand.distance2d(THUMB_TIP, RING_TIP) / scale
        + hand.distance2d(THUMB_TIP, PINKY_TIP) / scale
    ) / 4.0
    if gather < 0.45 and n_up <= 1:
        o = 0.75 + (0.45 - gather)
    scores["O"] = _clamp01(o)

    # P: similar to K but pointing down — approximate with index+middle+thumb.
    scores["P"] = _clamp01(_score_pattern(ext, (True, True, True, False, False)) * 0.55)

    # Q: similar to G pointing down.
    scores["Q"] = _clamp01(_score_pattern(ext, (True, True, False, False, False)) * 0.5)

    # R: index+middle crossed. Approximate: both up and tips close / crossed x-order.
    r = _score_pattern(ext, (None, True, True, False, False))
    if tip_index_middle_n < 0.22:
        r += 0.15
    ix = hand.xy(INDEX_TIP)[0]
    mx = hand.xy(MIDDLE_TIP)[0]
    # Crossing relative to MCP order
    if (ix - mx) * (hand.xy(INDEX_MCP)[0] - hand.xy(MIDDLE_MCP)[0]) < 0:
        r += 0.2
    scores["R"] = _clamp01(r * 0.75)

    # S: fist with thumb in front of fingers.
    s = _score_pattern(ext, (False, False, False, False, False))
    if tip_thumb_index_n < 0.5 and thumb_y < middle_mcp_y:
        s += 0.25
    scores["S"] = _clamp01(s * 0.75)

    # T: fist with thumb between index and middle.
    t = _score_pattern(ext, (False, False, False, False, False))
    between = min(hand.xy(INDEX_MCP)[0], hand.xy(MIDDLE_MCP)[0]) < hand.xy(THUMB_TIP)[0] < max(
        hand.xy(INDEX_MCP)[0], hand.xy(MIDDLE_MCP)[0]
    )
    if between:
        t += 0.3
    scores["T"] = _clamp01(t * 0.7)

    # U: index+middle up together (touching), thumb tucked.
    u = _score_pattern(ext, (False, True, True, False, False))
    if tip_index_middle_n < 0.25:
        u += 0.25
    if ring:
        u *= 0.35
    scores["U"] = _clamp01(u * 0.9)

    # V: index+middle up apart (V shape).
    v = _score_pattern(ext, (False, True, True, False, False))
    if tip_index_middle_n > 0.28:
        v += 0.3
    if ring:
        v *= 0.35
    scores["V"] = _clamp01(v * 0.9)

    # W: index+middle+ring up.
    w = _score_pattern(ext, (False, True, True, True, False))
    if ring and index and middle and not pinky:
        w += 0.15
    scores["W"] = _clamp01(w * 0.95)

    # X: index hooked (bent), others down. Index not fully extended.
    x = _score_pattern(ext, (None, False, False, False, False))
    # Index tip below a fully-up position but away from MCP (hook).
    idx_ext_partial = hand.distance2d(INDEX_TIP, INDEX_MCP) / scale
    if 0.25 < idx_ext_partial < 0.55 and hand.xy(INDEX_TIP)[1] > hand.xy(INDEX_PIP)[1] - 0.02:
        x += 0.35
    scores["X"] = _clamp01(x * 0.7)

    # Y: thumb + pinky out (shaka).
    scores["Y"] = _clamp01(_score_pattern(ext, (True, False, False, False, True)) * 0.95)

    # Soften near-ties between fist-like letters when evidence is weak.
    if n_up == 0:
        for letter in ("M", "N", "Q", "P"):
            scores[letter] = min(scores.get(letter, 0.0), 0.45)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_letter, best_score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0

    # Require a minimum score and a margin over the runner-up.
    if best_score < 0.55 or (best_score - second) < 0.05:
        return ASLPrediction(
            letter=None,
            confidence=best_score,
            handedness=hand.handedness,
            candidates=tuple(ranked[:5]),
        )

    return ASLPrediction(
        letter=best_letter,
        confidence=_clamp01(best_score),
        handedness=hand.handedness,
        candidates=tuple(ranked[:5]),
    )


def top_candidates(prediction: ASLPrediction, n: int = 3) -> List[Tuple[str, float]]:
    return list(prediction.candidates[:n])
