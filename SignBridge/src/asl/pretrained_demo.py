"""UNUSED / EXPERIMENTAL — not part of the live SignBridge demo.

The live application is:

    python -m src.asl.signbridge_demo

This module remains for reference only. It must not be wired into app.py
or the sequence classifier path.

Live demo: pretrained isolated ASL recognition → English label → TTS.

    python -m src.asl.pretrained_demo

This is isolated sign recognition over 200 ASL Citizen classes. It is not
full continuous ASL sentence translation. A sign has to be stable across
several inferences before ElevenLabs speaks it, and the same sign is not
spoken twice in a row.

    webcam → MediaPipe Holistic → temporal landmarks
           → pretrained BiGRU → English label → ElevenLabs TTS
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from typing import Any, Callable

import cv2
import numpy as np

from src.asl.pretrained_features import position_vector
from src.asl.pretrained_recognizer import (
    DEFAULT_MIN_FRAMES,
    IsolatedPrediction,
    LandmarkBuffer,
    PretrainedRecognizer,
    PredictionStabilizer,
    SentenceBuffer,
    confidence_threshold_from_env,
    download_pretrained_files,
    inference_hz_from_env,
    label_to_spoken_english,
    load_id_to_label,
    load_isolated_model,
    pick_device,
    resolve_pretrained_paths,
    stable_count_from_env,
)
from src.speech.text_to_speech import TextToSpeech, load_local_env
from src.vision.camera import CAMERA_LOST_ERROR, open_camera
from src.vision.holistic_tracker import HolisticTracker, HolisticTrackerError
from src.vision.overlay import AMBER, GREEN, RED, WHITE, draw_text_lines

WINDOW_NAME = "SignBridge - Pretrained ASL Recognition"
MAX_CONSECUTIVE_READ_FAILURES = 30


class Status(Enum):
    READY = "READY"
    SPEAKING = "SPEAKING..."
    ERROR = "ERROR"


class IsolatedDemoSession:
    """Camera-free state machine for buffer → infer → debounce → speak."""

    def __init__(
        self,
        recognizer: PretrainedRecognizer,
        speaker: TextToSpeech | None = None,
        executor: ThreadPoolExecutor | None = None,
        clock: Callable[[], float] | None = None,
        inference_hz: float | None = None,
        min_frames: int = DEFAULT_MIN_FRAMES,
        stable_count: int | None = None,
    ) -> None:
        self._recognizer = recognizer
        self._speaker = speaker or TextToSpeech.from_env()
        self._executor = executor or ThreadPoolExecutor(max_workers=1)
        self._owns_executor = executor is None
        self._clock = clock or time.perf_counter
        self.inference_interval = 1.0 / (inference_hz or inference_hz_from_env())
        self.min_frames = min_frames

        self.buffer = LandmarkBuffer()
        self.stabilizer = PredictionStabilizer(
            consecutive=stable_count if stable_count is not None else stable_count_from_env(),
            threshold=recognizer.threshold,
        )
        self.sentence = SentenceBuffer()

        self.status = Status.READY
        self.prediction: IsolatedPrediction | None = None
        self.detected_sign = "Unknown"
        self.confidence = 0.0
        self.error: str | None = None
        self._pending: Future[Any] | None = None
        self._pending_kind: str | None = None
        self._last_inference_at = 0.0

    @property
    def is_busy(self) -> bool:
        return self._pending is not None and not self._pending.done()

    def push_positions(self, position: np.ndarray) -> None:
        self.buffer.append(position)

    def maybe_infer(self) -> None:
        if self.is_busy or self.status == Status.SPEAKING:
            return
        if len(self.buffer) < self.min_frames:
            return
        now = self._clock()
        if now - self._last_inference_at < self.inference_interval:
            return
        self._last_inference_at = now
        clip = self.buffer.to_model_input()
        self._pending_kind = "infer"
        self._pending = self._executor.submit(self._recognizer.predict_tensor, clip)

    def poll(self) -> None:
        if self._pending is None or not self._pending.done():
            return
        future = self._pending
        kind = self._pending_kind
        self._pending = None
        self._pending_kind = None
        try:
            result = future.result()
        except Exception as exc:
            self.status = Status.ERROR
            self.error = f"{type(exc).__name__}: {exc}"
            return

        if kind == "infer":
            self._on_prediction(result)
        elif kind == "speak":
            self.status = Status.READY

    def _on_prediction(self, prediction: IsolatedPrediction) -> None:
        self.prediction = prediction
        self.confidence = prediction.confidence
        self.error = None
        stable = self.stabilizer.update(
            prediction.label, prediction.confidence, prediction.display_label
        )
        self.detected_sign = stable.display_label
        if stable.should_speak and stable.accepted_label:
            self.sentence.add(stable.accepted_label)
            spoken = label_to_spoken_english(stable.accepted_label)
            self.status = Status.SPEAKING
            self._pending_kind = "speak"
            self._pending = self._executor.submit(self._speak, spoken)
        elif self.status != Status.SPEAKING:
            self.status = Status.READY

    def _speak(self, text: str) -> bool:
        try:
            return bool(self._speaker.speak(text))
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return False

    def overlay_lines(self) -> list[str]:
        percent = f"{self.confidence * 100:.0f}%"
        lines = [
            "SIGNBRIDGE",
            "Pretrained ASL Recognition",
            "",
            "Detected sign:",
            self.detected_sign,
            "",
            "Confidence:",
            percent,
            "",
            "Status:",
            self.status.value,
        ]
        if self.sentence.display():
            lines.extend(["", "Signs:", self.sentence.display()])
        if self.error:
            lines.extend(["", self.error])
        lines.extend(["", "Press Q to quit."])
        return lines

    def close(self) -> None:
        if self._owns_executor:
            self._executor.shutdown(wait=False)


def overlay_color(status: Status) -> tuple[int, int, int]:
    if status == Status.ERROR:
        return RED
    if status == Status.SPEAKING:
        return AMBER
    return GREEN


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Isolated ASL recognition from a webcam using a pretrained "
            "BiGRU model. Not full sentence translation."
        )
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Minimum softmax confidence (default: 0.70 or SIGNBRIDGE_ASL_CONFIDENCE).",
    )
    parser.add_argument(
        "--inference-hz",
        type=float,
        default=None,
        help="Model inferences per second (default: 4).",
    )
    return parser.parse_args(argv)


def load_recognizer(threshold: float | None) -> PretrainedRecognizer:
    load_local_env()
    print("Loading pretrained ASL Citizen BiGRU model...")
    print("Isolated sign recognition — not full ASL sentence translation.")
    paths = resolve_pretrained_paths(download=True)
    if not paths.checkpoint.is_file() or not paths.labels.is_file():
        paths = download_pretrained_files()
    id_to_label = load_id_to_label(paths.labels)
    device = pick_device()
    model, config = load_isolated_model(paths.checkpoint, device=device)
    resolved_threshold = (
        float(threshold) if threshold is not None else confidence_threshold_from_env()
    )
    print(
        f"Loaded {config.get('num_classes', len(id_to_label))} classes, "
        f"input {(config.get('seq_len', 200), config.get('input_dim', 450))}, "
        f"device {device}, threshold {resolved_threshold:.2f}."
    )
    return PretrainedRecognizer(
        model=model,
        id_to_label=id_to_label,
        device=device,
        threshold=resolved_threshold,
    )


def run_loop(
    camera: cv2.VideoCapture,
    tracker: HolisticTracker,
    session: IsolatedDemoSession,
) -> int:
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

        frame = cv2.flip(frame, 1)
        landmarks = tracker.process(frame)
        tracker.draw(frame, landmarks)
        session.push_positions(
            position_vector(
                pose=landmarks.pose,
                left_hand=landmarks.left_hand,
                right_hand=landmarks.right_hand,
            )
        )
        session.maybe_infer()
        session.poll()

        color = overlay_color(session.status)
        draw_text_lines(frame, session.overlay_lines(), color=color)
        draw_text_lines(
            frame,
            ["Isolated signs only — not sentence translation."],
            origin=(12, frame.shape[0] - 24),
            color=WHITE,
            scale=0.5,
            line_height=20,
        )
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return 0


def main(argv: list[str] | None = None) -> int:
    print(
        "This pretrained demo is unused. The live app is:\n"
        "  python -m src.asl.signbridge_demo",
        file=sys.stderr,
    )
    args = parse_args(argv)

    try:
        recognizer = load_recognizer(args.threshold)
    except Exception as exc:
        print(f"Error: could not load the pretrained model: {exc}", file=sys.stderr)
        return 1

    opened = open_camera()
    if opened is None:
        return 1
    camera, camera_name = opened
    print(f"Camera: {camera_name}")

    speaker = TextToSpeech.from_env()
    if not speaker.is_available():
        print(
            "Warning: ELEVENLABS_API_KEY is not set. Recognition will still "
            "run, but nothing will be spoken.",
            file=sys.stderr,
        )

    session = IsolatedDemoSession(
        recognizer,
        speaker=speaker,
        inference_hz=args.inference_hz,
    )
    try:
        with HolisticTracker() as tracker:
            return run_loop(camera, tracker, session)
    except HolisticTrackerError as exc:
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
