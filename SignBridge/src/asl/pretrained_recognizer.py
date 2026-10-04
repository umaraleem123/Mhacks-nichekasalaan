"""UNUSED / EXPERIMENTAL — not part of the live SignBridge demo.

Kept for reference. The live path uses `src.asl.sequence_recognition` and a
locally trained BiGRU, not this Hugging Face ASL Citizen checkpoint.

Pretrained isolated-sign recognition (ASL Citizen BiGRU + attention).

Downloads the Hugging Face checkpoint on first use, keeps a rolling
landmark buffer, and classifies the *movement* of a sign — never a single
frame. This is isolated recognition of 200 ASL signs, not continuous
sentence translation.

Checkpoint
----------
SharoonArshad/asl-citizen-bigru-attention-encoder-200
    checkpoints/best_bigru_attention_model.pt
    metadata/id_to_label.json

Input shape: (B, 200, 450)
"""

from __future__ import annotations

import json
import os
import re
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from src.asl.pretrained_features import (
    FEATURE_DIM,
    POSITION_DIM,
    SEQ_LEN,
    model_input_from_positions,
)

HF_REPO_ID = "SharoonArshad/asl-citizen-bigru-attention-encoder-200"
CHECKPOINT_FILENAME = "checkpoints/best_bigru_attention_model.pt"
LABELS_FILENAME = "metadata/id_to_label.json"

DEFAULT_CONFIDENCE_THRESHOLD = 0.70
DEFAULT_STABLE_COUNT = 3
DEFAULT_INFERENCE_HZ = 4.0
DEFAULT_MIN_FRAMES = 12
DEFAULT_MAX_BUFFER_FRAMES = 90
CONFIDENCE_ENV = "SIGNBRIDGE_ASL_CONFIDENCE"
STABLE_ENV = "SIGNBRIDGE_ASL_STABLE_COUNT"
INFERENCE_HZ_ENV = "SIGNBRIDGE_ASL_INFERENCE_HZ"


def project_root() -> Path:
    """SignBridge/ directory, independent of the process working directory."""
    return Path(__file__).resolve().parents[2]


def default_pretrained_dir() -> Path:
    return project_root() / "models" / "pretrained"


def confidence_threshold_from_env(default: float = DEFAULT_CONFIDENCE_THRESHOLD) -> float:
    raw = os.environ.get(CONFIDENCE_ENV, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return min(1.0, max(0.0, value))


def stable_count_from_env(default: int = DEFAULT_STABLE_COUNT) -> int:
    raw = os.environ.get(STABLE_ENV, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def inference_hz_from_env(default: float = DEFAULT_INFERENCE_HZ) -> float:
    raw = os.environ.get(INFERENCE_HZ_ENV, "").strip()
    if not raw:
        return default
    try:
        return min(8.0, max(1.0, float(raw)))
    except ValueError:
        return default


@dataclass(frozen=True)
class PretrainedPaths:
    checkpoint: Path
    labels: Path


def resolve_pretrained_paths(
    cache_dir: Path | str | None = None,
    download: bool = True,
    hub_download: Callable[..., str] | None = None,
) -> PretrainedPaths:
    """Return local checkpoint and label-map paths, downloading if needed.

    `hub_download` is injected in tests so the real Hugging Face Hub is never
    contacted from the unit suite.
    """
    root = Path(cache_dir) if cache_dir is not None else default_pretrained_dir()
    checkpoint = root / CHECKPOINT_FILENAME
    labels = root / LABELS_FILENAME
    if checkpoint.is_file() and labels.is_file():
        return PretrainedPaths(checkpoint=checkpoint, labels=labels)
    if not download:
        raise FileNotFoundError(
            f"Pretrained files were not found under {root}. "
            "Run the demo once with a network connection to download them."
        )
    return download_pretrained_files(cache_dir=root, hub_download=hub_download)


def download_pretrained_files(
    cache_dir: Path | str | None = None,
    hub_download: Callable[..., str] | None = None,
) -> PretrainedPaths:
    """Fetch the checkpoint and id_to_label.json into models/pretrained/."""
    root = Path(cache_dir) if cache_dir is not None else default_pretrained_dir()
    root.mkdir(parents=True, exist_ok=True)

    downloader = hub_download
    if downloader is None:
        from huggingface_hub import hf_hub_download

        downloader = hf_hub_download

    checkpoint = Path(
        downloader(
            repo_id=HF_REPO_ID,
            filename=CHECKPOINT_FILENAME,
            local_dir=str(root),
        )
    )
    labels = Path(
        downloader(
            repo_id=HF_REPO_ID,
            filename=LABELS_FILENAME,
            local_dir=str(root),
        )
    )
    return PretrainedPaths(checkpoint=checkpoint, labels=labels)


def load_id_to_label(path: Path | str) -> dict[int, str]:
    """Load `{class_id: "HELLO", ...}` from the Hugging Face metadata file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("id_to_label.json must be a JSON object.")
    mapping: dict[int, str] = {}
    for key, value in payload.items():
        mapping[int(key)] = str(value)
    if not mapping:
        raise ValueError("id_to_label.json is empty.")
    return mapping


def softmax_probs(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    array = np.asarray(logits, dtype=np.float64)
    shifted = array - np.max(array, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=-1, keepdims=True)).astype(np.float32)


def label_to_spoken_english(label: str) -> str:
    """Turn a dataset gloss such as HELLO or WHAT1 into speakable English."""
    stripped = re.sub(r"\d+$", "", label.strip())
    spaced = re.sub(r"[\s_\-]+", " ", stripped).strip()
    return spaced.title() if spaced else label


class AttentionPool(nn.Module):
    """Softmax attention over the time axis of a BiGRU encoding."""

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attention(hidden), dim=1)
        return torch.sum(weights * hidden, dim=1)


class IsolatedSignBiGRU(nn.Module):
    """Architecture matching checkpoints/best_bigru_attention_model.pt."""

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        projection_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 200,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = SEQ_LEN
        self.num_classes = num_classes
        self.input_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        self.gru = nn.GRU(
            input_size=projection_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        encoded_dim = hidden_dim * 2
        self.attention_pool = AttentionPool(encoded_dim, hidden_dim)
        self.embedding = nn.Sequential(
            nn.Linear(encoded_dim, encoded_dim),
            nn.LayerNorm(encoded_dim),
        )
        self.classifier = nn.Linear(encoded_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input_norm(inputs)
        hidden = self.projection(hidden)
        hidden = self.dropout(hidden)
        hidden, _ = self.gru(hidden)
        pooled = self.attention_pool(hidden)
        embedded = self.embedding(pooled)
        embedded = self.dropout(embedded)
        return self.classifier(embedded)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_isolated_model(
    checkpoint_path: Path | str,
    device: torch.device | str | None = None,
) -> tuple[IsolatedSignBiGRU, dict[str, Any]]:
    """Load the full classifier from the Hugging Face checkpoint dict."""
    resolved_device = torch.device(device) if device is not None else pick_device()
    payload = torch.load(
        checkpoint_path, map_location=resolved_device, weights_only=False
    )
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(
            "Checkpoint is not the expected training dict "
            "(missing model_state_dict)."
        )
    config = dict(payload.get("config") or {})
    model = IsolatedSignBiGRU(
        input_dim=int(config.get("input_dim", FEATURE_DIM)),
        projection_dim=int(config.get("projection_dim", 256)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        num_layers=int(config.get("num_gru_layers", 2)),
        num_classes=int(config.get("num_classes", 200)),
        dropout=float(config.get("dropout", 0.35)),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(resolved_device)
    model.eval()
    return model, config


class LandmarkBuffer:
    """Rolling window of 225-d position vectors, later resampled to 200 frames."""

    def __init__(self, max_frames: int = DEFAULT_MAX_BUFFER_FRAMES) -> None:
        if max_frames < 2:
            raise ValueError("LandmarkBuffer needs at least two frames.")
        self.max_frames = max_frames
        self._frames: deque[np.ndarray] = deque(maxlen=max_frames)

    def append(self, position: np.ndarray | Sequence[float]) -> None:
        vector = np.asarray(position, dtype=np.float32).reshape(-1)
        if vector.size != POSITION_DIM:
            raise ValueError(
                f"Expected {POSITION_DIM} position features, got {vector.size}."
            )
        self._frames.append(vector.copy())

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)

    def as_array(self) -> np.ndarray:
        if not self._frames:
            return np.zeros((0, POSITION_DIM), dtype=np.float32)
        return np.stack(self._frames, axis=0)

    def to_model_input(self, length: int = SEQ_LEN) -> np.ndarray:
        return model_input_from_positions(self.as_array(), length=length)


@dataclass(frozen=True)
class IsolatedPrediction:
    """One inference result. `label` is None when confidence is too low."""

    class_id: int
    confidence: float
    label: str | None
    display_label: str
    logits: np.ndarray
    probabilities: np.ndarray


class PretrainedRecognizer:
    """Run the BiGRU on a 200-frame clip and map the class id to a gloss."""

    def __init__(
        self,
        model: nn.Module,
        id_to_label: dict[int, str],
        device: torch.device | str | None = None,
        threshold: float | None = None,
    ) -> None:
        self.model = model
        self.id_to_label = id_to_label
        self.device = torch.device(device) if device is not None else pick_device()
        self.threshold = (
            DEFAULT_CONFIDENCE_THRESHOLD if threshold is None else float(threshold)
        )
        self.model.to(self.device)
        self.model.eval()

    def predict_tensor(self, features: np.ndarray) -> IsolatedPrediction:
        """`features` is `(200, 450)` or `(B, 200, 450)`."""
        array = np.asarray(features, dtype=np.float32)
        if array.ndim == 2:
            array = array[None, ...]
        if array.ndim != 3 or array.shape[1] != SEQ_LEN or array.shape[2] != FEATURE_DIM:
            raise ValueError(
                f"Expected model input of shape (B, {SEQ_LEN}, {FEATURE_DIM}), "
                f"got {array.shape}."
            )
        tensor = torch.from_numpy(array).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        logits_np = logits.detach().cpu().numpy()[0]
        probabilities = softmax_probs(logits_np)
        class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[class_id])
        mapped = self.id_to_label.get(class_id)
        if mapped is None:
            label = None
            display = "Unknown"
        elif confidence < self.threshold:
            label = None
            display = "Unknown"
        else:
            label = mapped
            display = mapped
        return IsolatedPrediction(
            class_id=class_id,
            confidence=confidence,
            label=label,
            display_label=display,
            logits=logits_np,
            probabilities=probabilities,
        )

    def predict_positions(self, positions: np.ndarray) -> IsolatedPrediction:
        """Classify a variable-length position buffer."""
        return self.predict_tensor(model_input_from_positions(positions))


@dataclass
class StabilizerResult:
    display_label: str
    confidence: float
    accepted_label: str | None
    should_speak: bool
    stable: bool


class PredictionStabilizer:
    """Require the same sign across several inferences before accepting it.

    After a sign is accepted, it is not spoken again until the prediction
    changes or confidence falls below the threshold.
    """

    def __init__(
        self,
        consecutive: int = DEFAULT_STABLE_COUNT,
        threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.consecutive = max(1, int(consecutive))
        self.threshold = float(threshold)
        self._recent: list[str] = []
        self._locked_label: str | None = None

    def reset(self) -> None:
        self._recent.clear()
        self._locked_label = None

    def update(
        self, label: str | None, confidence: float, display_label: str
    ) -> StabilizerResult:
        if label is None or confidence < self.threshold:
            self._recent.clear()
            self._locked_label = None
            return StabilizerResult(
                display_label="Unknown",
                confidence=confidence,
                accepted_label=None,
                should_speak=False,
                stable=False,
            )

        self._recent.append(label)
        if len(self._recent) > self.consecutive:
            self._recent = self._recent[-self.consecutive :]

        stable = len(self._recent) == self.consecutive and all(
            item == label for item in self._recent
        )
        should_speak = False
        accepted: str | None = None
        if stable:
            accepted = label
            if self._locked_label != label:
                should_speak = True
                self._locked_label = label
        elif self._locked_label is not None and label != self._locked_label:
            self._locked_label = None

        shown = display_label if display_label else (label or "Unknown")
        return StabilizerResult(
            display_label=shown,
            confidence=confidence,
            accepted_label=accepted,
            should_speak=should_speak,
            stable=stable,
        )


@dataclass
class SentenceBuffer:
    """Remember recently accepted isolated signs. No NLP — just a list."""

    signs: list[str] = field(default_factory=list)
    max_signs: int = 8

    def add(self, label: str) -> None:
        if self.signs and self.signs[-1] == label:
            return
        self.signs.append(label)
        if len(self.signs) > self.max_signs:
            self.signs = self.signs[-self.max_signs :]

    def clear(self) -> None:
        self.signs.clear()

    def display(self) -> str:
        return " + ".join(self.signs) if self.signs else ""
