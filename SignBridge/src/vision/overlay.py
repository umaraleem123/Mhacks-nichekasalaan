"""Shared on-frame text drawing for the SignBridge windows."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

GREEN = (0, 255, 0)
AMBER = (0, 200, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_text_lines(
    frame: np.ndarray,
    lines: Sequence[str],
    origin: tuple[int, int] = (12, 32),
    color: tuple[int, int, int] = GREEN,
    scale: float = 0.7,
    line_height: int = 30,
) -> None:
    """Draw stacked lines of text, in place."""
    x, y = origin
    for index, line in enumerate(lines):
        position = (x, y + index * line_height)
        # Dark pass underneath keeps the text readable on a bright background.
        cv2.putText(frame, line, position, _FONT, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, position, _FONT, scale, color, 2, cv2.LINE_AA)
