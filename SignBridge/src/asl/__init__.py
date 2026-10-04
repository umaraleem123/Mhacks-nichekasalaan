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
]


def display_label(sign: str) -> str:
    """Uppercase gloss for on-screen display: hello → HELLO."""
    return str(sign).strip().upper()
