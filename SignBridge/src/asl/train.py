"""Train the ASL sign classifier from collected landmark features.

Run it with:

    python -m src.asl.train

Reads every CSV under data/asl/<sign>/, trains a random forest, prints an
evaluation, and writes models/asl_classifier.pkl. Classes come from the data
directory, so a newly collected sign is picked up without editing this file.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src import ASL_DATA_DIR, ASL_MODEL_PATH
from src.asl.features import FEATURE_DIM, FeatureError, validate_feature_vector

MIN_SAMPLES_PER_CLASS = 20
MIN_CLASSES = 2
RANDOM_SEED = 42


class TrainingDataError(RuntimeError):
    """The collected data is missing or not usable for training."""


def load_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load every sample CSV into a feature matrix and a label vector."""
    if not data_dir.is_dir():
        raise TrainingDataError(
            f"No training data directory at {data_dir}.\n"
            "Collect samples first:  python -m src.asl.data_collector"
        )

    features: list[np.ndarray] = []
    labels: list[str] = []

    for csv_path in sorted(data_dir.glob("*/*.csv")):
        label = csv_path.parent.name
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                raw = [row[key] for key in row if key != "label"]
                try:
                    features.append(validate_feature_vector([float(v) for v in raw]))
                except (FeatureError, TypeError, ValueError) as exc:
                    raise TrainingDataError(
                        f"{csv_path} line {row_number}: {exc}"
                    ) from exc
                labels.append(row.get("label") or label)

    if not features:
        raise TrainingDataError(
            f"No samples found under {data_dir}.\n"
            "Collect samples first:  python -m src.asl.data_collector"
        )
    return np.vstack(features), np.array(labels)


def check_dataset(labels: np.ndarray) -> Counter:
    """Fail with an actionable message if the data cannot train a model."""
    counts = Counter(labels.tolist())

    if len(counts) < MIN_CLASSES:
        raise TrainingDataError(
            f"Need at least {MIN_CLASSES} different signs to train, found "
            f"{len(counts)}: {sorted(counts)}.\n"
            "Collect samples for another sign with the data collector."
        )

    too_few = {name: n for name, n in counts.items() if n < MIN_SAMPLES_PER_CLASS}
    if too_few:
        detail = ", ".join(f"{name} has {n}" for name, n in sorted(too_few.items()))
        raise TrainingDataError(
            f"Every sign needs at least {MIN_SAMPLES_PER_CLASS} samples, but "
            f"{detail}.\nCollect more with:  python -m src.asl.data_collector"
        )
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the SignBridge ASL classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=ASL_DATA_DIR)
    parser.add_argument("--model-out", type=Path, default=ASL_MODEL_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        features, labels = load_dataset(args.data_dir)
        counts = check_dataset(labels)
    except TrainingDataError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(features)} samples of {FEATURE_DIM} features from {args.data_dir}")
    for name, count in sorted(counts.items()):
        print(f"  {name:<12} {count}")

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    model = RandomForestClassifier(n_estimators=args.trees, random_state=args.seed)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print(f"\nTrained on {len(x_train)} samples, tested on {len(x_test)}.")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.1%}\n")
    print(classification_report(y_test, predictions, zero_division=0))

    print("Confusion matrix (rows = actual, columns = predicted)")
    print("            " + "  ".join(f"{name:>8}" for name in model.classes_))
    matrix = confusion_matrix(y_test, predictions, labels=model.classes_)
    for name, row in zip(model.classes_, matrix):
        print(f"{name:<12}" + "  ".join(f"{value:>8}" for value in row))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "classes": list(model.classes_),
            "feature_dim": FEATURE_DIM,
        },
        args.model_out,
    )
    print(f"\nSaved model to {args.model_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
