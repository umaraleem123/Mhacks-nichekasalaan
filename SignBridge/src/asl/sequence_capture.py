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

`CaptureConfig()` holds one sign; `CaptureConfig.for_sentence()` holds a whole
signed sentence, ended by the user or by a duration ceiling.

Three things bound memory, all configurable through `CaptureConfig`:

- `sample_fps` thins the incoming camera stream. A webcam delivers ~30 fps, far
  more than is needed to see a movement, and near-identical frames add payload
  without adding information.
- `max_frames` caps the buffer with a fixed-length deque, so a capture left
  running cannot grow without bound.
- `max_frame_width` downscales wide frames before they are stored.
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterator

import cv2
import numpy as np

# Single-sign defaults: long enough to contain the movement of one sign, short
# enough to stay interactive. Nine frames shows a trajectory; one would not.
DEFAULT_DURATION_SECONDS = 1.5
DEFAULT_SAMPLE_FPS = 6.0
DEFAULT_MAX_FRAMES = 12
DEFAULT_MAX_FRAME_WIDTH = 640

# Sentence defaults: a short signed sentence, ended by the user or by the cap.
# 3 fps over 6 seconds is ~18 frames. A 30 fps camera stream would be ~180.
MAX_RECORDING_ENV = "SIGNBRIDGE_MAX_RECORDING_SECONDS"
SENTENCE_MAX_SECONDS = 6.0
SENTENCE_HARD_MAX_SECONDS = 15.0
SENTENCE_SAMPLE_FPS = 3.0
SENTENCE_MAX_FRAME_WIDTH = 512

# Consecutive sampled frames whose 48x48 grayscale mean-abs difference is at
# or below this (on a 0–255 scale) are treated as the same pose.
DUPLICATE_MEAN_ABS = 4.0
THUMB_SIZE = 48


def env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def sentence_max_seconds() -> float:
    """Recording ceiling, from SIGNBRIDGE_MAX_RECORDING_SECONDS or 6s."""
    value = env_float(MAX_RECORDING_ENV, SENTENCE_MAX_SECONDS)
    return min(SENTENCE_HARD_MAX_SECONDS, max(1.0, value))


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

    @classmethod
    def for_sentence(
        cls,
        max_seconds: float | None = None,
        sample_fps: float = SENTENCE_SAMPLE_FPS,
        max_frame_width: int = SENTENCE_MAX_FRAME_WIDTH,
    ) -> CaptureConfig:
        """Config for recording a whole signed sentence.

        `max_seconds` is a ceiling, not a target: the user normally ends the
        sentence early by pressing SPACE again. When omitted, the value comes
        from SIGNBRIDGE_MAX_RECORDING_SECONDS (default 6).
        """
        if max_seconds is None:
            max_seconds = sentence_max_seconds()
        if max_seconds > SENTENCE_HARD_MAX_SECONDS:
            raise SequenceCaptureError(
                f"max_seconds cannot exceed {SENTENCE_HARD_MAX_SECONDS:g}."
            )
        frames = int(round(max_seconds * sample_fps))
        return cls(
            duration_seconds=max_seconds,
            sample_fps=sample_fps,
            max_frames=max(4, frames + max(2, frames // 5)),
            max_frame_width=max_frame_width,
        )

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


@dataclass(frozen=True)
class CaptureStats:
    """Counts printed after SPACE stops a sentence."""

    captured_seconds: float = 0.0
    frames_before_optimization: int = 0
    frames_after_sampling: int = 0
    frames_after_deduplication: int = 0


def _gray_thumb(image: np.ndarray, size: int = THUMB_SIZE) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


def frames_are_near_duplicates(
    first: np.ndarray,
    second: np.ndarray,
    max_mean_abs: float = DUPLICATE_MEAN_ABS,
) -> bool:
    """True when two frames are visually almost the same pose."""
    thumb_a = _gray_thumb(first).astype(np.float32)
    thumb_b = _gray_thumb(second).astype(np.float32)
    return float(np.mean(np.abs(thumb_a - thumb_b))) <= max_mean_abs


def deduplicate_sequence(
    sequence: FrameSequence, max_mean_abs: float = DUPLICATE_MEAN_ABS
) -> FrameSequence:
    """Drop consecutive near-duplicates, keeping chronological order.

    The first frame is always kept. Later frames are kept only when they
    differ from the last kept frame, so idle holds do not pad the request.
    """
    frames = list(sequence.frames)
    if len(frames) <= 2:
        return sequence

    kept = [frames[0]]
    for frame in frames[1:]:
        if not frames_are_near_duplicates(kept[-1].image, frame.image, max_mean_abs):
            kept.append(frame)
    if len(kept) < 2:
        kept.append(frames[-1])
    return FrameSequence(kept)


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
        self._offered: int = 0
        self.last_stats = CaptureStats()

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
        self._offered = 0

    def reset(self) -> None:
        """Stop and clear the buffer."""
        self._frames.clear()
        self._started_at = None
        self._last_stored_at = None
        self._offered = 0

    def stop(self) -> FrameSequence:
        """End the capture, drop near-duplicates, and return the sequence."""
        sampled = self.sequence()
        elapsed = self.elapsed
        offered = self._offered
        optimized = deduplicate_sequence(sampled)
        self.last_stats = CaptureStats(
            captured_seconds=elapsed,
            frames_before_optimization=offered,
            frames_after_sampling=len(sampled),
            frames_after_deduplication=len(optimized),
        )
        self.reset()
        return optimized

    def add_frame(self, frame: np.ndarray) -> bool:
        """Offer a frame. Returns True if it was stored.

        Frames arriving faster than `sample_fps` are dropped, which is how the
        ~30 fps camera stream is thinned down.
        """
        if self._started_at is None:
            return False
        if frame is None or getattr(frame, "size", 0) == 0:
            raise SequenceCaptureError("Cannot store an empty frame.")

        self._offered += 1
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
