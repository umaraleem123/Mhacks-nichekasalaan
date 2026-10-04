"""Supported sign vocabulary and ASL recognition.

`SIGNS` is the single source of truth for the live temporal classifier.
Folder names under `data/sequences/` match these labels. This is a limited
hackathon vocabulary, not complete ASL.
"""

from __future__ import annotations

SIGNS: list[str] = [
    "hello",
    "yes",
    "no",
    "thank_you",
    "please",
    "help",
    "sorry",
    "good",
    "bad",
    "how_are_you",
    "i_me",
    "you",
    "want",
    "need",
    "understand",
    "dont_understand",
    "what",
    "where",
    "name",
    "goodbye",
    "nice_to_meet_you",
]

# The first ten signs. collect/original records and trains only these.
ORIGINAL_SIGNS: list[str] = [
    "bad",
    "good",
    "hello",
    "help",
    "how_are_you",
    "no",
    "please",
    "sorry",
    "thank_you",
    "yes",
]

# Each numbered group records a disjoint set of the newer signs. Clips land in
# data/sequences/<sign>/, so those branches merge by adding different folders.
COLLECTION_GROUPS: dict[str, list[str]] = {
    "1": ["i_me", "you", "name", "goodbye"],
    "2": ["want", "need", "what"],
    "3": ["understand", "dont_understand", "where", "nice_to_meet_you"],
    "original": list(ORIGINAL_SIGNS),
}


def normalize_sign(label: str) -> str:
    """Folder-safe label: \"Don't Understand\" → dont_understand."""
    return (
        str(label)
        .strip()
        .lower()
        .replace("'", "")
        .replace("\u2019", "")
        .replace(" ", "_")
        .replace("-", "_")
    )


def collection_signs(group: str | None = None, signs: str | None = None) -> list[str]:
    """Signs one person should record.

    `--group 1` uses COLLECTION_GROUPS. `--signs a,b` picks labels explicitly.
    With neither, the full vocabulary is returned.
    """
    if group and signs:
        raise ValueError("Pass either a group or a sign list, not both.")
    if group is not None:
        key = str(group).strip()
        if key not in COLLECTION_GROUPS:
            known = ", ".join(sorted(COLLECTION_GROUPS))
            raise ValueError(f"Unknown collection group '{key}'. Choose {known}.")
        return list(COLLECTION_GROUPS[key])
    if signs is None or not str(signs).strip():
        return list(SIGNS)
    chosen: list[str] = []
    for part in str(signs).split(","):
        if not part.strip():
            continue
        label = normalize_sign(part)
        if label not in SIGNS:
            raise ValueError(f"Unknown sign '{part.strip()}'.")
        if label not in chosen:
            chosen.append(label)
    if not chosen:
        raise ValueError("No signs selected.")
    return chosen


def display_label(sign: str) -> str:
    """Uppercase gloss for on-screen display: hello → HELLO."""
    return str(sign).strip().upper()
