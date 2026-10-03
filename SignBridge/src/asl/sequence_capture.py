"""Capture a short, time-ordered sequence of webcam frames.

An ASL sign is a movement, not a pose: the same handshape means different
things depending on where it starts, where it travels, and how it rotates. A
single frame throws all of that away, so recognition needs a short window of
frames instead.

This module only collects and thins frames. It knows nothing about Gemini or
about any recognizer, so it can feed whatever comes next.

How a sequence is represented
-----------------------------
A `FrameSequence` is an ordered list of `CapturedFrame` objects, oldest first,
each holding a BGR image and the seconds elapsed since the capture started.
Ordering is the meaning here, so nothing reorders the list.

Three things bound memory, all configurable through `CaptureConfig`:

- `sample_fps` thins the incoming camera stream. A webcam delivers ~30 fps, far
  more than is needed to see a movement, and near-identical frames add payload
  without adding information.
- `max_frames` caps the buffer with a fixed-length deque, so a capture left
  running cannot grow without bound.
- `max_frame_width` downscales wide frames before they are stored.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterator

import cv2
import numpy as np

DEFAULT_DURATION_SECONDS = 2.0
DEFAULT_SAMPLE_FPS = 6.0
DEFAULT_MAX_FRAMES = 24
DEFAULT_MAX_FRAME_WIDTH = 640


class SequenceCaptureError(ValueError):
    """The capture configuration or a frame was not usable."""


@dataclass(frozen=True)
class CaptureConfig:
    """Tunables for one capture. Change these rather than editing callers."""

    duration_seconds: float = DEFAULT_DURATION_SECONDS
    sample_fps: float = DEFAULT_SAMPLE_FPS
    max_frames: int = DEFAULT_MAX_FRAMES
    max_frame_width: int = DEFAULT_MAX_FRAME_WIDTH

    def __post_init__(self) -> None:
        if self.duration_seconds <= 0:
            raise SequenceCaptureError("duration_seconds must be positive.")
        if self.sample_fps <= 0:
            raise SequenceCaptureError("sample_fps must be positive.")
        if self.max_frames < 2:
            raise SequenceCaptureError(
                "max_frames must be at least 2; one frame cannot show movement."
            )
        if self.max_frame_width <= 0:
            raise SequenceCaptureError("max_frame_width must be positive.")

    @property
    def sample_interval(self) -> float:
        """Seconds between stored frames."""
        return 1.0 / self.sample_fps

    @property
    def target_frame_count(self) -> int:
        """Frames a full-duration capture stores, after the max_frames cap."""
        wanted = int(round(self.duration_seconds * self.sample_fps))
        return max(2, min(wanted, self.max_frames))


@dataclass(frozen=True)
class CapturedFrame:
    """One stored frame and when it arrived, relative to the capture start."""

    image: np.ndarray
    timestamp: float


@dataclass(frozen=True)
class FrameSequence:
    """An ordered window of frames, oldest first."""

    frames: list[CapturedFrame] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[CapturedFrame]:
        return iter(self.frames)

    @property
    def images(self) -> list[np.ndarray]:
        return [frame.image for frame in self.frames]

    @property
    def timestamps(self) -> list[float]:
        return [frame.timestamp for frame in self.frames]

    @property
    def duration(self) -> float:
        """Seconds between the first and last stored frame."""
        if len(self.frames) < 2:
            return 0.0
        return self.frames[-1].timestamp - self.frames[0].timestamp

    @property
    def shows_movement(self) -> bool:
        """Whether there are enough frames for movement to be visible."""
        return len(self.frames) >= 2


class SequenceCapture:
    """A rolling, rate-limited buffer of recent frames.

    Feed it every frame from the capture loop; it keeps a thinned, bounded
    window of the most recent ones.

        capture = SequenceCapture()
        capture.start()
        while not capture.is_complete():
            capture.add_frame(frame)      # called once per camera frame
        sequence = capture.sequence()
    """

    def __init__(
        self,
        config: CaptureConfig | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config or CaptureConfig()
        self._clock = clock
        self._frames: deque[CapturedFrame] = deque(maxlen=self.config.max_frames)
        self._started_at: float | None = None
        self._last_stored_at: float | None = None

    @property
    def is_running(self) -> bool:
        return self._started_at is not None

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def elapsed(self) -> float:
        if self._started_at is None:
            return 0.0
        return self._clock() - self._started_at

    @property
    def progress(self) -> float:
        """How far along the capture is, from 0.0 to 1.0, for a progress bar."""
        if self._started_at is None:
            return 0.0
        by_time = self.elapsed / self.config.duration_seconds
        by_frames = self.frame_count / self.config.target_frame_count
        return float(min(1.0, max(by_time, by_frames)))

    def start(self) -> None:
        """Begin a capture, discarding anything buffered from a previous one."""
        self._frames.clear()
        self._started_at = self._clock()
        self._last_stored_at = None

    def reset(self) -> None:
        """Stop and clear the buffer."""
        self._frames.clear()
        self._started_at = None
        self._last_stored_at = None

    def add_frame(self, frame: np.ndarray) -> bool:
        """Offer a frame. Returns True if it was stored.

        Frames arriving faster than `sample_fps` are dropped, which is how the
        ~30 fps camera stream is thinned down.
        """
        if self._started_at is None:
            return False
        if frame is None or getattr(frame, "size", 0) == 0:
            raise SequenceCaptureError("Cannot store an empty frame.")

        now = self._clock()
        if (
            self._last_stored_at is not None
            and now - self._last_stored_at < self.config.sample_interval
        ):
            return False

        self._frames.append(
            CapturedFrame(self._downscale(frame), now - self._started_at)
        )
        self._last_stored_at = now
        return True

    def is_complete(self) -> bool:
        """True once the window covers the configured duration or fills up."""
        if self._started_at is None:
            return False
        if self.frame_count >= self.config.target_frame_count:
            return True
        return self.elapsed >= self.config.duration_seconds

    def sequence(self) -> FrameSequence:
        """Snapshot the buffer, oldest frame first."""
        return FrameSequence(list(self._frames))

    def _downscale(self, frame: np.ndarray) -> np.ndarray:
        """Shrink wide frames and copy, so later edits cannot alter the buffer."""
        width = frame.shape[1]
        if width <= self.config.max_frame_width:
            return frame.copy()

        scale = self.config.max_frame_width / width
        height = max(1, int(round(frame.shape[0] * scale)))
        return cv2.resize(
            frame, (self.config.max_frame_width, height), interpolation=cv2.INTER_AREA
        )
