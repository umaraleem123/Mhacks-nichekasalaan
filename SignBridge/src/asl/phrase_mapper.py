"""Map limited-vocabulary sign labels to spoken English phrases.

This is a lookup table, not a translator. Unknown labels stay empty rather
than being guessed.
"""

from __future__ import annotations

from src.asl import SIGNS

PHRASE_BY_LABEL: dict[str, str] = {
    "hello": "Hello",
    "yes": "Yes",
    "no": "No",
    "thank_you": "Thank you",
    "please": "Please",
    "help": "Help",
    "sorry": "Sorry",
    "good": "Good",
    "bad": "Bad",
    "how_are_you": "How are you?",
    "i_me": "I",
    "you": "You",
    "want": "Want",
    "need": "Need",
    "understand": "Understand",
    "dont_understand": "Don't understand",
    "what": "What",
    "where": "Where",
    "name": "Name",
    "goodbye": "Goodbye",
}


def normalize_label(label: str) -> str:
    return str(label).strip().lower().replace(" ", "_").replace("-", "_")


def label_to_english(label: str) -> str:
    """HELLO / hello / thank_you → the matching English phrase."""
    key = normalize_label(label)
    if key in PHRASE_BY_LABEL:
        return PHRASE_BY_LABEL[key]
    return PHRASE_BY_LABEL.get(key.replace("__", "_"), "")


def format_sentence(labels: list[str]) -> str:
    """Join accepted signs: ['hello', 'how_are_you'] → 'Hello, how are you?'."""
    phrases = [label_to_english(label) for label in labels]
    phrases = [part for part in phrases if part]
    if not phrases:
        return ""
    rest = [part[0].lower() + part[1:] if part else part for part in phrases[1:]]
    return ", ".join([phrases[0], *rest])


def known_labels() -> list[str]:
    return list(SIGNS)
