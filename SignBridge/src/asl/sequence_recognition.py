"""Live temporal sign recognition — sequences, not single frames.

A rolling 5 second buffer of two-hand landmarks is classified a few times
per second. Predictions below the confidence threshold, or without enough
motion, become Unknown. Stable labels feed a sentence buffer and TTS.
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from src.asl import display_label
from src.asl.phrase_mapper import format_sentence, label_to_english
from src.asl.sequence_features import (
    FEATURE_DIM,
    POSITION_DIM,
    features_from_position_sequence,
    motion_energy,
)
from src.asl.sequence_model import SequenceBiGRU, softmax_probs

UNKNOWN = "unknown"
DEMO_MODE_ENV = "SIGNBRIDGE_DEMO_MODE"
CONFIDENCE_ENV = "SIGNBRIDGE_CONFIDENCE_THRESHOLD"
DEFAULT_BUFFER_SECONDS = 5.0
DEFAULT_INFERENCE_HZ = 4.0
DEFAULT_MIN_FRAMES = 12


def demo_mode_enabled() -> bool:
    raw = os.environ.get(DEMO_MODE_ENV, "true").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def confidence_threshold_from_env() -> float:
    raw = os.environ.get(CONFIDENCE_ENV, "").strip()
    if raw:
        try:
            return min(0.99, max(0.0, float(raw)))
        except ValueError:
            pass
    return 0.80 if demo_mode_enabled() else 0.75


def demo_settings() -> dict:
    demo = demo_mode_enabled()
    return {
        "demo_mode": demo,
        "threshold": confidence_threshold_from_env(),
        "window": 5,
        "min_agree": 4 if demo else 3,
        "cooldown": 1.6 if demo else 0.8,
        "motion": 0.018 if demo else 0.012,
        "inference_hz": DEFAULT_INFERENCE_HZ,
        "buffer_seconds": DEFAULT_BUFFER_SECONDS,
    }


@dataclass
class SequencePrediction:
    label: str | None
    display_label: str
    confidence: float
    class_id: int
    probabilities: np.ndarray


class MotionGate:
    """Ignore still hands so idle webcam time is not classified."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def is_signing(self, features: np.ndarray) -> bool:
        return motion_energy(features) >= self.threshold


class PredictionSmoother:
    """Accept a label only when it repeats consistently in the recent window."""

    def __init__(self, window: int = 5, min_agree: int = 3, threshold: float = 0.75) -> None:
        self.window = window
        self.min_agree = min_agree
        self.threshold = threshold
        self.recent: deque[str | None] = deque(maxlen=window)

    def reset(self) -> None:
        self.recent.clear()

    def update(self, label: str | None, confidence: float) -> str | None:
        if label is None or confidence < self.threshold:
            self.recent.append(None)
            return None
        self.recent.append(label)
        if len(self.recent) < self.min_agree:
            return None
        tail = list(self.recent)[-self.min_agree :]
        if all(item == label for item in tail):
            return label
        return None


class Cooldown:
    def __init__(self, seconds: float, clock: Callable[[], float] | None = None) -> None:
        self.seconds = seconds
        self._clock = clock or time.perf_counter
        self._until = 0.0

    def active(self) -> bool:
        return self._clock() < self._until

    def trigger(self) -> None:
        self._until = self._clock() + self.seconds


@dataclass
class SentenceBuffer:
    labels: list[str] = field(default_factory=list)
    max_signs: int = 8

    def add(self, label: str) -> bool:
        if self.labels and self.labels[-1] == label:
            return False
        self.labels.append(label)
        if len(self.labels) > self.max_signs:
            self.labels = self.labels[-self.max_signs :]
        return True

    def clear(self) -> None:
        self.labels.clear()

    def english(self) -> str:
        return format_sentence(self.labels)


class LiveSequenceRecognizer:
    """Run the BiGRU on a padded clip and apply the confidence threshold."""

    def __init__(
        self,
        model: SequenceBiGRU,
        classes: list[str],
        threshold: float,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.classes = classes
        self.threshold = threshold
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, features: np.ndarray) -> SequencePrediction:
        array = np.asarray(features, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"Expected (T, {FEATURE_DIM}) features, got {array.shape}."
            )
        tensor = torch.from_numpy(array[None, ...]).to(self.device)
        lengths = torch.tensor([array.shape[0]], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(tensor, lengths)
        probs = softmax_probs(logits.detach().cpu().numpy()[0])
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        mapped = self.classes[class_id] if 0 <= class_id < len(self.classes) else None
        if mapped is None or confidence < self.threshold:
            return SequencePrediction(
                label=None,
                display_label="Unknown",
                confidence=confidence,
                class_id=class_id,
                probabilities=probs,
            )
        return SequencePrediction(
            label=mapped,
            display_label=display_label(mapped),
            confidence=confidence,
            class_id=class_id,
            probabilities=probs,
        )


class RecognitionSession:
    """Rolling buffer + motion gate + smoother + sentence buffer + TTS hooks."""

    def __init__(
        self,
        recognizer: LiveSequenceRecognizer,
        speaker=None,
        clock: Callable[[], float] | None = None,
        settings: dict | None = None,
        executor=None,
    ) -> None:
        self.recognizer = recognizer
        self.speaker = speaker
        self._clock = clock or time.perf_counter
        self.settings = settings or demo_settings()
        self.gate = MotionGate(self.settings["motion"])
        self.smoother = PredictionSmoother(
            window=int(self.settings["window"]),
            min_agree=int(self.settings["min_agree"]),
            threshold=float(self.settings["threshold"]),
        )
        self.cooldown = Cooldown(float(self.settings["cooldown"]), clock=self._clock)
        self.sentence = SentenceBuffer()
        self.buffer: deque[np.ndarray] = deque()
        self.buffer_seconds = float(self.settings["buffer_seconds"])
        self.inference_interval = 1.0 / float(self.settings["inference_hz"])
        self.min_frames = int(self.settings.get("min_frames", DEFAULT_MIN_FRAMES))
        self.status = "WAITING FOR SIGN"
        self.detected = "Unknown"
        self.confidence = 0.0
        self.translation = ""
        self.error: str | None = None
        self._last_infer = 0.0
        self._rested = True
        self._last_spoken = ""
        self._pending = None
        self._executor = executor

    def push_positions(self, positions: np.ndarray, timestamp: float | None = None) -> None:
        now = self._clock() if timestamp is None else timestamp
        vector = np.asarray(positions, dtype=np.float32).reshape(POSITION_DIM)
        self.buffer.append(np.concatenate([vector, np.array([now], dtype=np.float32)]))
        cutoff = now - self.buffer_seconds
        while self.buffer and float(self.buffer[0][-1]) < cutoff:
            self.buffer.popleft()

    def _position_clip(self) -> np.ndarray:
        if not self.buffer:
            return np.zeros((0, POSITION_DIM), dtype=np.float32)
        stacked = np.stack([frame[:-1] for frame in self.buffer], axis=0)
        return stacked.astype(np.float32)

    def maybe_infer(self) -> SequencePrediction | None:
        if self.status == "SPEAKING...":
            return None
        if self._pending is not None and not getattr(self._pending, "done", lambda: True)():
            return None
        now = self._clock()
        if now - self._last_infer < self.inference_interval:
            return None
        if len(self.buffer) < self.min_frames:
            self.status = "WAITING FOR SIGN"
            return None
        self._last_infer = now
        features = features_from_position_sequence(self._position_clip())
        moving = self.gate.is_signing(features)
        if not moving:
            self._rested = True
            self.smoother.reset()
            self.status = "WAITING FOR SIGN"
            self.detected = "Unknown"
            self.confidence = 0.0
            return None

        prediction = self.recognizer.predict(features)
        self.confidence = prediction.confidence
        self.detected = prediction.display_label
        accepted = self.smoother.update(prediction.label, prediction.confidence)
        if accepted is None:
            if self.status != "SPEAKING...":
                self.status = "READY" if prediction.label else "READY"
            return prediction

        if self.cooldown.active() or not self._rested:
            return prediction

        added = self.sentence.add(accepted)
        self._rested = False
        self.cooldown.trigger()
        english = label_to_english(accepted)
        self.translation = english
        if added:
            self._speak(english)
        return prediction

    def _speak(self, text: str) -> None:
        if not text or text == self._last_spoken:
            return
        if self.speaker is None:
            self._last_spoken = text
            return
        self.status = "SPEAKING..."
        self._last_spoken = text

        def _run() -> bool:
            try:
                return bool(self.speaker.speak(text))
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                return False

        if self._executor is not None:
            self._pending = self._executor.submit(_run)
        else:
            _run()
            self.status = "READY"

    def poll(self) -> None:
        pending = self._pending
        if pending is None:
            return
        done = pending.done() if hasattr(pending, "done") else True
        if not done:
            return
        try:
            pending.result()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        self._pending = None
        self.status = "READY"

    def speak_sentence(self) -> None:
        text = self.sentence.english()
        if not text:
            return
        self._last_spoken = ""
        self._speak(text)

    def clear_sentence(self) -> None:
        self.sentence.clear()
        self.translation = ""
        self._last_spoken = ""
        self.status = "READY"

    def overlay_lines(self) -> list[str]:
        percent = f"{self.confidence * 100:.0f}%"
        return [
            "SIGNBRIDGE",
            "ASL → English → Voice",
            "",
            "Detected:",
            self.detected,
            "",
            "Confidence:",
            percent,
            "",
            "Translation:",
            self.translation or "—",
            "",
            "Status:",
            self.status,
            "",
            "Sentence:",
            self.sentence.english() or "—",
            "",
            "SPACE = clear sentence",
            "T = speak current sentence",
            "Q = quit",
        ]
