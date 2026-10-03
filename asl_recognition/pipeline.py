"""End-to-end ASL recognition pipeline with temporal smoothing."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np

from .classifier import ASLPrediction, classify_asl
from .landmarks import HandLandmarks, HandTracker


@dataclass
class StablePrediction:
    """Smoothed letter after looking at a short history of raw predictions."""

    letter: Optional[str]
    confidence: float
    raw: Optional[ASLPrediction]
    hands: List[HandLandmarks]


class ASLRecognizer:
    """Camera-frame → landmarks → ASL letter, with majority-vote smoothing."""

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        *,
        still_image: bool = False,
        history: int = 8,
        min_votes: int = 4,
        min_confidence: float = 0.55,
    ) -> None:
        self._tracker = HandTracker(model_path=model_path, still_image=still_image)
        self._history: Deque[Tuple[Optional[str], float]] = deque(maxlen=history)
        self._min_votes = min_votes
        self._min_confidence = min_confidence
        self.spelled: str = ""

    def close(self) -> None:
        self._tracker.close()

    def __enter__(self) -> "ASLRecognizer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def process(self, bgr_frame: np.ndarray) -> StablePrediction:
        hands = self._tracker.detect(bgr_frame)
        if not hands:
            self._history.append((None, 0.0))
            return StablePrediction(letter=None, confidence=0.0, raw=None, hands=[])

        # Prefer the first / primary hand for lettering.
        raw = classify_asl(hands[0])
        self._history.append((raw.letter, raw.confidence))
        letter, conf = self._stabilize()
        return StablePrediction(letter=letter, confidence=conf, raw=raw, hands=hands)

    def _stabilize(self) -> Tuple[Optional[str], float]:
        votes = [letter for letter, conf in self._history if letter and conf >= self._min_confidence]
        if len(votes) < self._min_votes:
            return None, 0.0

        counts = Counter(votes)
        best, n = counts.most_common(1)[0]
        if n < self._min_votes:
            return None, 0.0

        confs = [c for letter, c in self._history if letter == best]
        avg_conf = sum(confs) / max(len(confs), 1)
        return best, avg_conf

    def commit_letter(self, letter: Optional[str] = None) -> Optional[str]:
        """Append the current stable letter (or an explicit letter) to spelled text."""
        if letter is None:
            letter, _ = self._stabilize()
        if not letter:
            return None
        self.spelled += letter
        return letter

    def backspace(self) -> None:
        self.spelled = self.spelled[:-1]

    def clear_text(self) -> None:
        self.spelled = ""
