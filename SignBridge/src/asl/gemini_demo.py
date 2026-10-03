"""Manual-trigger Gemini ASL recognition demo.

    python -m src.asl.gemini_demo

Press SPACE to record a short clip of one sign and send it to Gemini; press Q
to quit. One key press produces exactly one request: nothing is captured or
sent automatically, and SPACE is ignored while a request is in flight.

    webcam -> SequenceCapture -> ordered frame sequence -> Gemini
           -> structured result -> display

The whole ordered sequence goes to Gemini in a single request. Frames are never
sent individually, because an ASL sign is defined partly by movement and one
frame cannot show it.

Captured frames live in memory only, for the duration of the request, and are
never written to disk.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum

import cv2
import numpy as np

from src.asl.gemini_recognizer import (
    DEFAULT_MODEL,
    GeminiConfig,
    GeminiRecognizerError,
    GeminiRequestError,
    GeminiSignRecognizer,
    SignInterpretation,
    api_key_from_env,
)
from src.asl.sequence_capture import CaptureConfig, SequenceCapture
from src.vision.camera import CAMERA_LOST_ERROR, open_camera
from src.vision.hand_tracker import HandTracker, HandTrackerError
from src.vision.overlay import AMBER, GREEN, RED, WHITE, draw_text_lines

WINDOW_NAME = "SignBridge - Gemini ASL"
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
    ERROR = "ERROR"


class DemoSession:
    """Owns the state machine: idle, capturing, awaiting Gemini, showing result.

    Kept free of OpenCV so the flow can be tested without a camera or window.
    Gemini runs on a single worker thread, which keeps the video live while a
    request is in flight; the single worker is also what guarantees one request
    at a time.
    """

    def __init__(
        self,
        recognizer: GeminiSignRecognizer,
        capture_config: CaptureConfig | None = None,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self._recognizer = recognizer
        self._capture = SequenceCapture(capture_config or CaptureConfig())
        self._executor = executor or ThreadPoolExecutor(max_workers=1)
        self._owns_executor = executor is None
        self._pending: Future[SignInterpretation] | None = None
        self._analyzing_since: float | None = None

        self.status = Status.READY
        self.result: SignInterpretation | None = None
        self.error: str | None = None

    @property
    def capture_config(self) -> CaptureConfig:
        return self._capture.config

    @property
    def is_busy(self) -> bool:
        """True while capturing or waiting on Gemini, when SPACE is ignored."""
        return self.status in (Status.CAPTURING, Status.ANALYZING)

    @property
    def capture_progress(self) -> float:
        return self._capture.progress

    @property
    def capture_remaining(self) -> float:
        return max(0.0, self.capture_config.duration_seconds - self._capture.elapsed)

    @property
    def frames_captured(self) -> int:
        return self._capture.frame_count

    @property
    def analyzing_elapsed(self) -> float:
        """Seconds spent waiting on Gemini, for the on-screen timer."""
        if self._analyzing_since is None:
            return 0.0
        return time.perf_counter() - self._analyzing_since

    def request_capture(self) -> bool:
        """Handle SPACE. Returns False when ignored because work is in flight."""
        if self.is_busy:
            return False

        self.result = None
        self.error = None
        self._capture.start()
        self.status = Status.CAPTURING
        return True

    def offer_frame(self, frame: np.ndarray) -> None:
        """Feed one camera frame. Only stored while capturing."""
        if self.status is not Status.CAPTURING:
            return

        self._capture.add_frame(frame)
        if self._capture.is_complete():
            self._submit()

    def _submit(self) -> None:
        """Send the finished sequence as a single Gemini request."""
        sequence = self._capture.sequence()
        self._capture.reset()

        if not sequence.shows_movement:
            self.status = Status.ERROR
            self.error = (
                f"Captured only {len(sequence)} frame(s), too few to show "
                "movement. Check that the camera is delivering frames."
            )
            return

        self.status = Status.ANALYZING
        self._analyzing_since = time.perf_counter()
        self._pending = self._executor.submit(
            self._recognizer.recognize_sequence, sequence
        )

    def poll(self) -> None:
        """Collect the Gemini result once it is ready. Call once per frame."""
        if self._pending is None or not self._pending.done():
            return

        pending, self._pending = self._pending, None
        elapsed = self.analyzing_elapsed
        self._analyzing_since = None
        try:
            self.result = pending.result()
            self.status = Status.RESULT
            print(f"Gemini answered in {elapsed:.1f}s")
        except GeminiRequestError as exc:
            # Full detail to the terminal for debugging, one line on screen.
            # Both have already been through redact_secrets.
            self.status = Status.ERROR
            self.error = exc.user_message
            print(exc.diagnostics(), file=sys.stderr)
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
    lines = ["SIGNBRIDGE", "Mode: Gemini ASL", "", f"Status: {session.status.value}", ""]

    if session.status is Status.CAPTURING:
        lines.append("CAPTURING SIGN...")
        lines.append(f"{session.capture_remaining:.1f}s")
        lines.append(f"frames: {session.frames_captured}")
    elif session.status is Status.ANALYZING:
        lines.append("ANALYZING...")
        lines.append(f"{session.analyzing_elapsed:.1f}s")
    elif session.status is Status.RESULT and session.result is not None:
        result = session.result
        lines.append("Detected sign:")
        lines.append(result.sign.upper())
        lines.append("")
        lines.append("Confidence:")
        lines.append(f"{result.confidence:.0%}")
        if result.description:
            lines.append("")
            lines.append("Description:")
            lines += textwrap.wrap(result.description, DESCRIPTION_WRAP_WIDTH)[:4]
    elif session.status is Status.ERROR:
        lines.append("Gemini error")
        lines += textwrap.wrap(session.error or "", DESCRIPTION_WRAP_WIDTH)[:4]
        lines.append("")
        lines.append("Press SPACE to try again.")
    else:
        lines.append("Press SPACE to capture a sign")
        duration = session.capture_config.duration_seconds
        lines.append(f"({duration:.0f}s clip, then one Gemini request)")

    lines.append("")
    lines.append("SPACE = capture    Q = quit")
    return lines


def status_color(session: DemoSession) -> tuple[int, int, int]:
    if session.status is Status.CAPTURING:
        return RED
    if session.status is Status.ANALYZING:
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
        description="Manual-trigger Gemini ASL recognition demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    defaults = CaptureConfig()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name")
    parser.add_argument(
        "--duration", type=float, default=defaults.duration_seconds,
        help="seconds of video per capture",
    )
    parser.add_argument(
        "--sample-fps", type=float, default=defaults.sample_fps,
        help="frames sampled per second from the camera stream",
    )
    parser.add_argument(
        "--max-frames", type=int, default=defaults.max_frames,
        help="hard cap on frames per sequence",
    )
    gemini_defaults = GeminiConfig()
    parser.add_argument(
        "--threshold", type=float, default=gemini_defaults.confidence_threshold,
        help="below this confidence the result is reported as unknown",
    )
    parser.add_argument(
        "--thinking", default=gemini_defaults.thinking_level,
        choices=("minimal", "low", "medium", "high"),
        help="Gemini reasoning effort; low keeps the demo responsive",
    )
    parser.add_argument(
        "--timeout", type=float, default=gemini_defaults.request_timeout_seconds,
        help="seconds before the request is abandoned",
    )
    parser.add_argument(
        "--frames-only", action="store_true",
        help="send labeled JPEG frames instead of a short video",
    )
    parser.add_argument(
        "--camera", type=int, default=None,
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
            session.request_capture()  # ignored while busy


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Fail before opening the camera, so a missing key is obvious immediately
    # rather than after recording a clip.
    try:
        api_key_from_env()
    except GeminiRecognizerError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        capture_config = CaptureConfig(
            duration_seconds=args.duration,
            sample_fps=args.sample_fps,
            max_frames=args.max_frames,
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
            send_as_video=not args.frames_only,
        )
    )
    print(f"Model: {args.model}  (thinking: {args.thinking})")
    print(
        f"Capture: {capture_config.duration_seconds:.1f}s at "
        f"{capture_config.sample_fps:g} fps "
        f"(~{capture_config.target_frame_count} frames per request)"
    )
    print(
        f"Sending as: {'short video' if not args.frames_only else 'labeled frames'}"
        f"    timeout: {args.timeout:g}s    retries: none"
    )
    print(f"Signs: {', '.join(recognizer.supported_signs)}, or unknown")
    print("SPACE = capture and send one request    Q = quit")

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
