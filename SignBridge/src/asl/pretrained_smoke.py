"""UNUSED / EXPERIMENTAL — not part of the live SignBridge demo.

Download-and-load smoke test for the pretrained ASL Citizen model.

    python -m src.asl.pretrained_smoke

Downloads the Hugging Face checkpoint if it is not already in
models/pretrained/, loads the BiGRU, and runs one dummy inference.
No webcam is required. This is not part of the mocked unit suite.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from src.asl.pretrained_features import FEATURE_DIM, SEQ_LEN
from src.asl.pretrained_recognizer import (
    IsolatedSignBiGRU,
    download_pretrained_files,
    load_id_to_label,
    load_isolated_model,
    pick_device,
    resolve_pretrained_paths,
    softmax_probs,
)


def main() -> int:
    print("Resolving pretrained checkpoint (download on first run)...")
    try:
        paths = resolve_pretrained_paths(download=True)
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    if not paths.checkpoint.is_file() or not paths.labels.is_file():
        paths = download_pretrained_files()

    print(f"Checkpoint: {paths.checkpoint}")
    print(f"Labels:     {paths.labels}")

    id_to_label = load_id_to_label(paths.labels)
    device = pick_device()
    model, config = load_isolated_model(paths.checkpoint, device=device)

    input_dim = int(config.get("input_dim", FEATURE_DIM))
    seq_len = int(config.get("seq_len", SEQ_LEN))
    num_classes = int(config.get("num_classes", len(id_to_label)))

    print(f"Device: {device}")
    print(f"input_dim: {input_dim}")
    print(f"seq_len: {seq_len}")
    print(f"num_classes: {num_classes}")
    print(f"label map size: {len(id_to_label)}")

    if input_dim != FEATURE_DIM:
        print(f"Error: expected input_dim {FEATURE_DIM}, got {input_dim}", file=sys.stderr)
        return 1
    if seq_len != SEQ_LEN:
        print(f"Error: expected seq_len {SEQ_LEN}, got {seq_len}", file=sys.stderr)
        return 1
    if not isinstance(model, IsolatedSignBiGRU):
        print("Error: loaded object is not IsolatedSignBiGRU", file=sys.stderr)
        return 1
    if len(id_to_label) != num_classes:
        print(
            f"Error: label map has {len(id_to_label)} entries, config has {num_classes}",
            file=sys.stderr,
        )
        return 1

    dummy = torch.zeros(1, seq_len, input_dim, device=device)
    with torch.no_grad():
        logits = model(dummy)
    if tuple(logits.shape) != (1, num_classes):
        print(f"Error: logits shape {tuple(logits.shape)}", file=sys.stderr)
        return 1

    logits_np = logits.detach().cpu().numpy()[0]
    probabilities = softmax_probs(logits_np)
    class_id = int(np.argmax(probabilities))
    if class_id not in id_to_label:
        print(f"Error: class id {class_id} missing from label map", file=sys.stderr)
        return 1

    print(
        f"Dummy inference: class_id={class_id} "
        f"label={id_to_label[class_id]} "
        f"confidence={float(probabilities[class_id]):.4f}"
    )
    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
