"""Lightweight bidirectional GRU for isolated sign sequences.

Input:  (B, T, 252) plus a length vector
Output: (B, num_classes) logits

Packed sequences keep the model honest about variable clip lengths. Attention
pools the valid timesteps; padding is masked out.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.asl.sequence_features import FEATURE_DIM

DEFAULT_HIDDEN_DIM = 128
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.35


class SequenceBiGRU(nn.Module):
    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_layers: int = DEFAULT_NUM_LAYERS,
        num_classes: int = 10,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        encoded_dim = hidden_dim * 2
        self.attention = nn.Linear(encoded_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoded_dim, num_classes)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """`inputs` is `(B, T, 252)`; `lengths` is `(B,)` on CPU or same device."""
        lengths_cpu = lengths.detach().to("cpu").long().clamp(min=1)
        packed = pack_padded_sequence(
            inputs, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        encoded, _ = pad_packed_sequence(packed_out, batch_first=True)
        scores = self.attention(encoded).squeeze(-1)
        time_index = torch.arange(encoded.size(1), device=encoded.device)
        mask = time_index.unsqueeze(0) < lengths.to(encoded.device).unsqueeze(1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        return self.classifier(self.dropout(pooled))


def softmax_probs(logits: np.ndarray) -> np.ndarray:
    array = np.asarray(logits, dtype=np.float64)
    shifted = array - np.max(array, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=-1, keepdims=True)).astype(np.float32)


def load_sequence_model(
    checkpoint_path: Path,
    labels_path: Path,
    device: torch.device | str = "cpu",
) -> tuple[SequenceBiGRU, list[str], dict]:
    import json

    meta = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    classes = [str(item) for item in meta["classes"]]
    payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state = payload["model_state_dict"]
        hidden = int(payload.get("hidden_dim", meta.get("hidden_dim", DEFAULT_HIDDEN_DIM)))
        layers = int(payload.get("num_layers", meta.get("num_layers", DEFAULT_NUM_LAYERS)))
        input_dim = int(payload.get("input_dim", meta.get("feature_dim", FEATURE_DIM)))
    else:
        state = payload
        hidden = int(meta.get("hidden_dim", DEFAULT_HIDDEN_DIM))
        layers = int(meta.get("num_layers", DEFAULT_NUM_LAYERS))
        input_dim = int(meta.get("feature_dim", FEATURE_DIM))

    model = SequenceBiGRU(
        input_dim=input_dim,
        hidden_dim=hidden,
        num_layers=layers,
        num_classes=len(classes),
    )
    model.load_state_dict(state, strict=True)
    model.to(torch.device(device))
    model.eval()
    return model, classes, meta
