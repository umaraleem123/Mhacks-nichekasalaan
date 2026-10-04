"""Train the local temporal ASL classifier.

    python -m src.asl.train_sequence_model

Reads complete sign clips from data/sequences/<label>/*.npz, splits by
sequence (not by frame), trains a small BiGRU on CPU, and writes:

    models/asl_sequence_model.pt
    models/asl_sequence_labels.json
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src import SEQUENCE_DATA_DIR, SEQUENCE_LABELS_PATH, SEQUENCE_MODEL_PATH
from src.asl.sequence_dataset import (
    SequenceDataError,
    SequenceExample,
    augment_sequence,
    load_all_sequences,
    mirror_augmentation_enabled,
    pad_batch,
    split_by_sequence,
    with_mirrored_copies,
    write_label_map,
)
from src.asl.sequence_features import FEATURE_DIM
from src.asl.sequence_model import SequenceBiGRU

DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 1e-3
MIN_SEQUENCES_PER_CLASS = 2
RANDOM_SEED = 42


class SequenceClipDataset(Dataset):
    def __init__(
        self,
        examples: list[SequenceExample],
        label_to_id: dict[str, int],
        augment: bool = False,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.examples = examples
        self.label_to_id = label_to_id
        self.augment = augment
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        example = self.examples[index]
        features = example.features
        if self.augment:
            features = augment_sequence(features, rng=self._rng)
        return features, self.label_to_id[example.label]


def collate_clips(batch: list[tuple[np.ndarray, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    padded, lengths = pad_batch(sequences)
    return (
        torch.from_numpy(padded),
        torch.from_numpy(lengths),
        labels,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the SignBridge temporal ASL classifier.",
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args(argv)


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == labels).float().mean().item())


def run_epoch(
    model: SequenceBiGRU,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0.0
    total = 0
    for inputs, lengths, labels in loader:
        if training:
            optimizer.zero_grad()
        logits = model(inputs, lengths)
        loss = criterion(logits, labels)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        total_correct += float((logits.argmax(1) == labels).sum().item())
        total += int(labels.size(0))
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, total_correct / total


def train(
    data_dir=None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    seed: int = RANDOM_SEED,
    model_path=None,
    labels_path=None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    examples = load_all_sequences(data_dir or SEQUENCE_DATA_DIR)
    counts = Counter(example.label for example in examples)
    for label, count in sorted(counts.items()):
        if count < MIN_SEQUENCES_PER_CLASS:
            raise SequenceDataError(
                f"Class '{label}' has {count} sequence(s); need at least "
                f"{MIN_SEQUENCES_PER_CLASS}."
            )

    train_examples, val_examples = split_by_sequence(examples, seed=seed)
    original_train = len(train_examples)
    mirrored_train = 0
    if mirror_augmentation_enabled():
        train_examples, original_train, mirrored_train = with_mirrored_copies(
            train_examples
        )
    classes = sorted({example.label for example in examples})
    label_to_id = {label: index for index, label in enumerate(classes)}

    print(f"Classes: {len(classes)}")
    print(f"Original training sequences: {original_train}")
    print(f"Mirrored training sequences: {mirrored_train}")
    print(f"Effective training sequences: {len(train_examples)}")
    print(f"Validation sequences: {len(val_examples)}")
    print("Per-class counts:")
    for label in classes:
        print(f"  {label}: {counts[label]}")

    train_loader = DataLoader(
        SequenceClipDataset(train_examples, label_to_id, augment=True, seed=seed),
        batch_size=min(batch_size, len(train_examples)),
        shuffle=True,
        collate_fn=collate_clips,
    )
    val_loader = DataLoader(
        SequenceClipDataset(val_examples, label_to_id, augment=False, seed=seed),
        batch_size=min(batch_size, len(val_examples)),
        shuffle=False,
        collate_fn=collate_clips,
    )

    model = SequenceBiGRU(input_dim=FEATURE_DIM, num_classes=len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_state = None
    best_val = -1.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None)
        print(f"Epoch {epoch}/{epochs}")
        print(f"Train accuracy: {train_acc:.3f}   loss: {train_loss:.4f}")
        print(f"Validation accuracy: {val_acc:.3f}   loss: {val_loss:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        best_val = 0.0

    print(f"Best validation accuracy: {best_val:.3f}")

    out_model = Path(model_path or SEQUENCE_MODEL_PATH)
    out_labels = Path(labels_path or SEQUENCE_LABELS_PATH)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "classes": classes,
            "input_dim": FEATURE_DIM,
            "hidden_dim": model.hidden_dim,
            "num_layers": 2,
            "best_val_accuracy": best_val,
        },
        out_model,
    )
    write_label_map(
        Path(out_labels),
        classes,
        extra={
            "feature_dim": FEATURE_DIM,
            "hidden_dim": model.hidden_dim,
            "num_layers": 2,
            "best_val_accuracy": best_val,
            "train_sequences": len(train_examples),
            "original_train_sequences": original_train,
            "mirrored_train_sequences": mirrored_train,
            "val_sequences": len(val_examples),
        },
    )
    print(f"Wrote {out_model}")
    print(f"Wrote {out_labels}")
    return {
        "classes": classes,
        "train_sequences": len(train_examples),
        "original_train_sequences": original_train,
        "mirrored_train_sequences": mirrored_train,
        "val_sequences": len(val_examples),
        "best_val_accuracy": best_val,
        "model_path": str(out_model),
        "labels_path": str(out_labels),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        train(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )
    except SequenceDataError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
