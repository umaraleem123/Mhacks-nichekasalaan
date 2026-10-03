"""Supported sign vocabulary and ASL recognition.

`SIGNS` is the single source of truth for which signs SignBridge knows about.
To add one, append it here and collect data for it; the collector keys, the
trainer, and the recognizer all derive their classes from this list or from the
data directory, so no other file needs to change.
"""

SIGNS: list[str] = ["hello", "yes", "no"]
