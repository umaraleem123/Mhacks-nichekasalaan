"""UNUSED / EXPERIMENTAL — not part of the live SignBridge demo.

The live application is:

    python -m src.asl.signbridge_demo

This Gemini sentence demo is kept for reference only.

Manual-trigger Gemini ASL sentence demo.

    python -m src.asl.gemini_demo

Press SPACE to start recording a signed sentence, press SPACE again to stop
and send the complete clip to Gemini. The English translation is spoken
aloud. Press Q to quit.

    webcam -> temporal ASL sentence -> Gemini 3.7 Flash
           -> English sentence -> text-to-speech

One completed sentence produces exactly one Gemini request. Nothing is sent
while the user is still signing. SPACE is ignored while a request or spoken
utterance is in flight.

Captured frames live in memory only. If they are encoded to a short video,
that file is written to a temporary directory and deleted immediately.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from typing import Any, Callable

import cv2
import numpy as np

from src.asl.gemini_recognizer import (
    DEFAULT_MODEL,
    GeminiConfig,
    GeminiRecognizerError,
    GeminiRequestError,
    GeminiSignRecognizer,
    SentenceTranslation,
    api_key_from_env,
    gemini_timeout_seconds,
)
from src.asl.sequence_capture import (
    SENTENCE_HARD_MAX_SECONDS,
    SENTENCE_SAMPLE_FPS,
    CaptureConfig,
    SequenceCapture,
    sentence_max_seconds,
)
from src.speech.text_to_speech import TextToSpeech
from src.vision.camera import CAMERA_LOST_ERROR, open_camera
from src.vision.hand_tracker import HandTracker, HandTrackerError
from src.vision.overlay import AMBER, GREEN, RED, WHITE, draw_text_lines

WINDOW_NAME = "SignBridge - ASL to Voice"
MAX_CONSECUTIVE_READ_FAILURES = 30
KEY_SPACE = 32
DESCRIPTION_WRAP_WIDTH = 52
PROGRESS_BAR_SIZE = (320, 18)


class Status(Enum):
    """What the app is doing, shown on screen."""

    READY = "READY"
    CAPTURING = "CAPTURING"
    ANALYZING = "ANALYZING"
    RESULT = "RESULT"
    SPEAKING = "SPEAKING"
    ERROR = "ERROR"


class DemoSession:
    """Owns the state machine for one signed sentence.

    Kept free of OpenCV so the flow can be tested without a camera or window.
    Gemini and TTS share a single worker thread, which keeps the video live
    and guarantees one request and one utterance at a time.
    """

    def __init__(
        self,
        recognizer: GeminiSignRecognizer,
        capture_config: CaptureConfig | None = None,
        speaker: TextToSpeech | None = None,
        executor: ThreadPoolExecutor | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._recognizer = recognizer
        self._capture = SequenceCapture(
            capture_config or CaptureConfig.for_sentence(),
            clock=clock or time.perf_counter,
        )
        self._speaker = speaker or TextToSpeech.from_env()
        self._executor = executor or ThreadPoolExecutor(max_workers=1)
        self._owns_executor = executor is None
        self._pending: Future[Any] | None = None
        self._pending_kind: str | None = None
        self._analyzing_since: float | None = None

        self.status = Status.READY
        self.result: SentenceTranslation | None = None
        self.error: str | None = None

    @property
    def capture_config(self) -> CaptureConfig:
        return self._capture.config

    @property
    def is_busy(self) -> bool:
        """True while Gemini or TTS is running, when SPACE is ignored."""
        return self.status in (Status.ANALYZING, Status.SPEAKING)

    @property
    def capture_progress(self) -> float:
        return self._capture.progress

    @property
    def capture_elapsed(self) -> float:
        return self._capture.elapsed

    @property
    def frames_captured(self) -> int:
        return self._capture.frame_count

    @property
    def analyzing_elapsed(self) -> float:
        """Seconds spent waiting on Gemini, for the on-screen timer."""
        if self._analyzing_since is None:
            return 0.0
        return time.perf_counter() - self._analyzing_since

    def handle_space(self) -> bool:
        """Toggle sentence recording. Returns False when ignored."""
        if self.is_busy:
            return False

        if self.status is Status.CAPTURING:
            self._submit()
            return True

        self.result = None
        self.error = None
        self._capture.start()
        self.status = Status.CAPTURING
        return True

    def request_capture(self) -> bool:
        """Backward-compatible name for tests that still call start."""
        return self.handle_space()

    def offer_frame(self, frame: np.ndarray) -> None:
        """Feed one camera frame. Only stored while capturing."""
        if self.status is not Status.CAPTURING:
            return

        self._capture.add_frame(frame)
        if self._capture.is_complete():
            self._submit()

    def _submit(self) -> None:
        """Send the finished sentence as a single Gemini request."""
        sequence = self._capture.stop()
        stats = self._capture.last_stats
        print(f"Captured: {stats.captured_seconds:.1f} sec")
        print(f"Frames before optimization: {stats.frames_before_optimization}")
        print(f"Frames after sampling: {stats.frames_after_sampling}")
        print(f"Frames after deduplication: {stats.frames_after_deduplication}")

        if not sequence.shows_movement:
            self.status = Status.ERROR
            self.error = (
                f"Captured only {len(sequence)} frame(s), too few to show "
                "movement. Check that the camera is delivering frames."
            )
            return

        self.status = Status.ANALYZING
        self._analyzing_since = time.perf_counter()
        self._pending_kind = "gemini"
        self._pending = self._executor.submit(
            self._recognizer.translate_sequence, sequence
        )

    def _speak_safely(self, text: str) -> bool:
        """Speak one sentence. Failures stay here so the UI cannot crash."""
        try:
            return bool(self._speaker.speak(text))
        except Exception as exc:
            print(f"Speech failed ({type(exc).__name__})", file=sys.stderr)
            return False

    def _start_speech(self, text: str) -> None:
        self.status = Status.SPEAKING
        self._pending_kind = "speech"
        self._pending = self._executor.submit(self._speak_safely, text)

    def poll(self) -> None:
        """Collect Gemini or TTS once it is ready. Call once per frame."""
        while self._pending is not None and self._pending.done():
            pending, self._pending = self._pending, None
            kind, self._pending_kind = self._pending_kind, None

            if kind == "speech":
                spoken = False
                try:
                    spoken = bool(pending.result())
                except Exception as exc:
                    print(f"Speech failed ({type(exc).__name__})", file=sys.stderr)
                self.status = Status.READY if spoken else Status.RESULT
                if spoken:
                    self.result = None
                continue

            elapsed = self.analyzing_elapsed
            self._analyzing_since = None
            try:
                self.result = pending.result()
                self.status = Status.RESULT
                print(f"Gemini answered in {elapsed:.1f}s")
                spoken_text = self.result.speakable_text
                if spoken_text:
                    self._start_speech(spoken_text)
            except GeminiRequestError as exc:
                print(exc.diagnostics(), file=sys.stderr)
                if exc.is_timeout:
                    self.status = Status.READY
                    self.error = GeminiRequestError.TIMEOUT_MESSAGE
                    self.result = None
                else:
                    self.status = Status.ERROR
                    self.error = exc.user_message
            except GeminiRecognizerError as exc:
                self.status = Status.ERROR
                self.error = str(exc)
                print(f"Gemini request failed\nError: {exc}", file=sys.stderr)
            except Exception as exc:
                self.status = Status.ERROR
                self.error = f"Unexpected failure ({type(exc).__name__})."
                print(
                    f"Unexpected failure\nError type: {type(exc).__name__}",
                    file=sys.stderr,
                )

    def close(self) -> None:
        if self._owns_executor:
            self._executor.shutdown(wait=False)


def status_lines(session: DemoSession) -> list[str]:
    """The text block drawn in the corner, derived from session state."""
    lines = [
        "SIGNBRIDGE",
        "ASL → VOICE",
        "",
        f"Status: {session.status.value}",
        "",
    ]

    if session.status is Status.CAPTURING:
        lines.append("RECORDING ASL SENTENCE")
        lines.append(f"Time: {session.capture_elapsed:.1f}s")
        lines.append(f"frames: {session.frames_captured}")
    elif session.status is Status.ANALYZING:
        lines.append("ANALYZING...")
        lines.append(f"{session.analyzing_elapsed:.1f}s")
    elif session.status is Status.SPEAKING and session.result is not None:
        result = session.result
        lines.append("TRANSLATION:")
        lines += textwrap.wrap(
            f'"{result.english_translation}"', DESCRIPTION_WRAP_WIDTH
        )[:4]
        lines.append("")
        lines.append("CONFIDENCE:")
        lines.append(f"{result.confidence:.0%}")
        lines.append("")
        lines.append("SPEAKING...")
    elif session.status is Status.RESULT and session.result is not None:
        result = session.result
        if result.is_unknown:
            lines.append("TRANSLATION:")
            lines.append("(none)")
            if result.notes:
                lines.append("")
                lines += textwrap.wrap(result.notes, DESCRIPTION_WRAP_WIDTH)[:3]
        else:
            lines.append("TRANSLATION:")
            lines += textwrap.wrap(
                result.english_translation, DESCRIPTION_WRAP_WIDTH
            )[:4]
            lines.append("")
            lines.append("CONFIDENCE:")
            lines.append(f"{result.confidence:.0%}")
        lines.append("")
        lines.append("Press SPACE for another sentence")
    elif session.status is Status.ERROR:
        lines.append("Gemini error")
        lines += textwrap.wrap(session.error or "", DESCRIPTION_WRAP_WIDTH)[:4]
        lines.append("")
        lines.append("Press SPACE to try again.")
    else:
        lines.append("READY")
        if session.error:
            lines += textwrap.wrap(session.error, DESCRIPTION_WRAP_WIDTH)[:2]
        else:
            lines.append("Press SPACE to start signing")
        duration = session.capture_config.duration_seconds
        lines.append(f"(max {duration:.0f}s, then one Gemini request)")

    lines.append("")
    lines.append("SPACE = Start/Stop sentence")
    lines.append("Q = Quit")
    return lines


def status_color(session: DemoSession) -> tuple[int, int, int]:
    if session.status is Status.CAPTURING:
        return RED
    if session.status in (Status.ANALYZING, Status.SPEAKING):
        return AMBER
    if session.status is Status.ERROR:
        return AMBER
    if session.status is Status.RESULT and session.result is not None:
        return AMBER if session.result.is_unknown else GREEN
    return WHITE


def draw_progress_bar(frame: np.ndarray, progress: float) -> None:
    """A capture progress bar along the bottom of the frame."""
    width, height = PROGRESS_BAR_SIZE
    x = max(12, (frame.shape[1] - width) // 2)
    y = frame.shape[0] - height - 24
    filled = int(width * min(1.0, max(0.0, progress)))

    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 0), -1)
    if filled:
        cv2.rectangle(frame, (x, y), (x + filled, y + height), RED, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), WHITE, 2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual-trigger Gemini ASL sentence demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name")
    parser.add_argument(
        "--max-duration",
        type=float,
        default=sentence_max_seconds(),
        help=f"maximum seconds of signing (ceiling, up to {SENTENCE_HARD_MAX_SECONDS:g})",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=SENTENCE_SAMPLE_FPS,
        help="frames sampled per second from the camera stream",
    )
    gemini_defaults = GeminiConfig()
    parser.add_argument(
        "--threshold",
        type=float,
        default=gemini_defaults.confidence_threshold,
        help="below this confidence the result is treated as unrecognized",
    )
    parser.add_argument(
        "--thinking",
        default=gemini_defaults.thinking_level,
        choices=("minimal", "low", "medium", "high"),
        help="Gemini reasoning effort; low keeps the demo responsive",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=gemini_timeout_seconds(),
        help="seconds before the request is abandoned",
    )
    parser.add_argument(
        "--frames-only",
        action="store_true",
        help="unused; the demo always sends a sampled JPEG sequence, not a video",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="camera index; default prefers a Logitech Brio, then the default camera",
    )
    return parser.parse_args(argv)


def run_loop(camera, tracker: HandTracker, session: DemoSession) -> int:
    read_failures = 0

    while True:
        captured, frame = camera.read()
        if not captured:
            read_failures += 1
            if read_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                print(CAMERA_LOST_ERROR, file=sys.stderr)
                return 1
            cv2.waitKey(10)
            continue
        read_failures = 0

        # Mirror first: selfie view, and the orientation MediaPipe assumes when
        # labeling hands Left or Right.
        frame = cv2.flip(frame, 1)

        # Hand the clean frame to the capture before any landmarks or text are
        # drawn, so Gemini sees the hand rather than our overlay.
        session.offer_frame(frame)
        session.poll()

        hands = tracker.process(frame)
        tracker.draw(frame, hands)
        draw_text_lines(frame, status_lines(session), color=status_color(session))
        if session.status is Status.CAPTURING:
            draw_progress_bar(frame, session.capture_progress)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return 0
        if key == KEY_SPACE:
            session.handle_space()


def main(argv: list[str] | None = None) -> int:
    print(
        "This Gemini demo is unused. The live app is:\n"
        "  python -m src.asl.signbridge_demo",
        file=sys.stderr,
    )
    args = parse_args(argv)

    # Fail before opening the camera, so a missing key is obvious immediately
    # rather than after recording a sentence.
    try:
        api_key_from_env()
    except GeminiRecognizerError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        capture_config = CaptureConfig.for_sentence(
            max_seconds=args.max_duration,
            sample_fps=args.sample_fps,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    recognizer = GeminiSignRecognizer(
        config=GeminiConfig(
            model=args.model,
            confidence_threshold=args.threshold,
            thinking_level=args.thinking,
            request_timeout_seconds=args.timeout,
            send_as_video=False,
        )
    )
    print(f"Model: {args.model}  (thinking: {args.thinking})")
    print(
        f"Sentence capture: up to {capture_config.duration_seconds:.1f}s at "
        f"{capture_config.sample_fps:g} fps "
        f"(~{int(round(capture_config.duration_seconds * capture_config.sample_fps))} frames)"
    )
    print(
        f"Sending as: sampled JPEG sequence"
        f"    timeout: {args.timeout:g}s    retries: none"
        f"    API: models.generate_content"
    )
    print("SPACE = start / stop sentence    Q = quit")

    opened = open_camera(args.camera)
    if opened is None:
        return 1
    camera, _camera_name = opened

    session = DemoSession(recognizer, capture_config)
    try:
        with HandTracker() as tracker:
            return run_loop(camera, tracker, session)
    except HandTrackerError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        session.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
