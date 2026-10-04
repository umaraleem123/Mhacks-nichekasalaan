"""On-disk temporal sequences and train/validation splitting.

Each `.npz` file is one complete signing motion:

    features         (T, 252) float32
    label            string
    sequence_length  int

Sequences are never split into frames for the train/val cut.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.asl import SIGNS
from src.asl.sequence_features import FEATURE_DIM, mirror_sequence_features

MIRROR_AUG_ENV = "SIGNBRIDGE_MIRROR_AUGMENTATION"

DEFAULT_VAL_RATIO = 0.20
RANDOM_SEED = 42


class SequenceDataError(RuntimeError):
    """Sequence files are missing or unusable."""


@dataclass(frozen=True)
class SequenceExample:
    path: Path
    features: np.ndarray
    label: str
    sequence_length: int


def save_sequence(
    path: Path,
    features: np.ndarray,
    label: str,
) -> Path:
    """Write one complete sign clip. `features` is `(T, 252)`."""
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != FEATURE_DIM:
        raise SequenceDataError(
            f"Expected features of shape (T, {FEATURE_DIM}), got {array.shape}."
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=array,
        label=np.asarray(label),
        sequence_length=np.int32(array.shape[0]),
    )
    return path


def load_sequence(path: Path) -> SequenceExample:
    payload = np.load(Path(path), allow_pickle=True)
    try:
        features = np.asarray(payload["features"], dtype=np.float32)
        raw_label = payload["label"]
        length = int(np.asarray(payload["sequence_length"]))
    finally:
        payload.close()

    if raw_label.shape == ():
        label = str(raw_label.item())
    else:
        label = str(raw_label)

    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise SequenceDataError(
            f"{path}: expected (T, {FEATURE_DIM}) features, got {features.shape}."
        )
    if length != features.shape[0]:
        length = int(features.shape[0])
    return SequenceExample(
        path=Path(path),
        features=features,
        label=label,
        sequence_length=length,
    )


def list_sequence_paths(data_dir: Path) -> list[Path]:
    root = Path(data_dir)
    if not root.is_dir():
        return []
    return sorted(root.glob("*/*.npz"))


def load_all_sequences(data_dir: Path) -> list[SequenceExample]:
    paths = list_sequence_paths(data_dir)
    if not paths:
        raise SequenceDataError(
            f"No sequence files under {data_dir}.\n"
            "Record some with:  python -m src.asl.sequence_data_collector"
        )
    return [load_sequence(path) for path in paths]


def split_by_sequence(
    examples: list[SequenceExample],
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[list[SequenceExample], list[SequenceExample]]:
    """80/20 split by whole recordings. Every class appears in both sets."""
    if val_ratio <= 0.0 or val_ratio >= 1.0:
        raise SequenceDataError("val_ratio must be between 0 and 1.")

    grouped: dict[str, list[SequenceExample]] = {}
    for example in examples:
        grouped.setdefault(example.label, []).append(example)

    rng = np.random.default_rng(seed)
    train: list[SequenceExample] = []
    val: list[SequenceExample] = []

    for label, items in sorted(grouped.items()):
        if len(items) < 2:
            raise SequenceDataError(
                f"Class '{label}' has {len(items)} sequence(s). Need at least 2 "
                "so the class can appear in both training and validation."
            )
        order = np.arange(len(items))
        rng.shuffle(order)
        n_val = max(1, int(round(len(items) * val_ratio)))
        n_val = min(n_val, len(items) - 1)
        chosen = [items[index] for index in order]
        val.extend(chosen[:n_val])
        train.extend(chosen[n_val:])

    return train, val


def mirror_augmentation_enabled(default: bool = True) -> bool:
    """SIGNBRIDGE_MIRROR_AUGMENTATION, default on."""
    raw = os.environ.get(MIRROR_AUG_ENV, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def mirror_example(example: SequenceExample) -> SequenceExample:
    """Same label and length; geometry is left/right flipped. No file write."""
    mirrored = mirror_sequence_features(example.features)
    return SequenceExample(
        path=example.path,
        features=mirrored,
        label=example.label,
        sequence_length=int(mirrored.shape[0]),
    )


def with_mirrored_copies(
    examples: list[SequenceExample],
) -> tuple[list[SequenceExample], int, int]:
    """Append an in-memory mirrored copy of each example. Disk is unchanged."""
    originals = list(examples)
    mirrored = [mirror_example(example) for example in originals]
    return originals + mirrored, len(originals), len(mirrored)


def pad_batch(
    sequences: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Pad variable-length clips to `(B, T_max, 252)` plus a length vector."""
    if not sequences:
        return (
            np.zeros((0, 1, FEATURE_DIM), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    lengths = np.array([int(seq.shape[0]) for seq in sequences], dtype=np.int64)
    max_len = int(np.max(lengths))
    batch = np.zeros((len(sequences), max_len, FEATURE_DIM), dtype=np.float32)
    for index, seq in enumerate(sequences):
        array = np.asarray(seq, dtype=np.float32)
        batch[index, : array.shape[0]] = array
    return batch, lengths


def augment_sequence(
    features: np.ndarray,
    rng: np.random.Generator | None = None,
    noise_std: float = 0.015,
    scale_range: tuple[float, float] = (0.92, 1.08),
    stretch_range: tuple[float, float] = (0.88, 1.12),
    max_shift: int = 3,
) -> np.ndarray:
    """Light feature-level jitter. Not for validation clips."""
    sequence = np.asarray(features, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[1] != FEATURE_DIM:
        raise SequenceDataError(
            f"Expected (T, {FEATURE_DIM}) to augment, got {sequence.shape}."
        )
    engine = rng or np.random.default_rng()
    positions = sequence[:, : FEATURE_DIM // 2].copy()

    scale = float(engine.uniform(*scale_range))
    positions *= scale
    positions += engine.normal(0.0, noise_std, size=positions.shape).astype(np.float32)

    stretch = float(engine.uniform(*stretch_range))
    new_len = max(4, int(round(positions.shape[0] * stretch)))
    if new_len != positions.shape[0]:
        src = np.linspace(0.0, 1.0, positions.shape[0])
        dst = np.linspace(0.0, 1.0, new_len)
        stretched = np.zeros((new_len, positions.shape[1]), dtype=np.float32)
        for dim in range(positions.shape[1]):
            stretched[:, dim] = np.interp(dst, src, positions[:, dim])
        positions = stretched

    shift = int(engine.integers(-max_shift, max_shift + 1))
    if shift > 0:
        pad = np.repeat(positions[:1], shift, axis=0)
        positions = np.concatenate([pad, positions], axis=0)
    elif shift < 0:
        drop = min(-shift, max(0, positions.shape[0] - 4))
        if drop:
            positions = positions[drop:]

    velocity = np.zeros_like(positions)
    if positions.shape[0] > 1:
        velocity[1:] = positions[1:] - positions[:-1]
    return np.concatenate([positions, velocity], axis=1).astype(np.float32)


def write_label_map(path: Path, classes: list[str], extra: dict | None = None) -> None:
    payload = {"classes": list(classes)}
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_label_map(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def class_index_map(classes: list[str] | None = None) -> dict[str, int]:
    labels = list(classes or SIGNS)
    return {label: index for index, label in enumerate(labels)}
