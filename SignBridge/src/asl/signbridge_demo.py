"""SignBridge live demo: temporal ASL → English → ElevenLabs.

    python -m src.asl.signbridge_demo

SignBridge is a hackathon prototype demonstrating temporal ASL recognition
for a limited vocabulary. It is not a complete ASL translation system.

SPACE clears the sentence. T speaks it. Q quits. Signs are accepted from
the webcam automatically — no keypress per sign.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2

from src import SEQUENCE_LABELS_PATH, SEQUENCE_MODEL_PATH
from src.asl.sequence_features import positions_from_hands
from src.asl.sequence_model import load_sequence_model
from src.asl.sequence_recognition import (
    LiveSequenceRecognizer,
    RecognitionSession,
    demo_settings,
)
from src.speech.text_to_speech import TextToSpeech, load_local_env
from src.vision.camera import CAMERA_LOST_ERROR, open_camera
from src.vision.hand_tracker import HandTracker, HandTrackerError
from src.vision.overlay import AMBER, GREEN, RED, WHITE, draw_text_lines

WINDOW_NAME = "SignBridge"
MAX_CONSECUTIVE_READ_FAILURES = 30
KEY_SPACE = 32


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Limited-vocabulary temporal ASL recognition. "
            "Not a complete ASL translation system."
        )
    )
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--labels", default=None)
    return parser.parse_args(argv)


def overlay_color(status: str) -> tuple[int, int, int]:
    if status == "SPEAKING...":
        return AMBER
    if status.startswith("WAITING"):
        return WHITE
    if status == "ERROR":
        return RED
    return GREEN


def run_loop(camera, tracker: HandTracker, session: RecognitionSession) -> int:
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
        hands = tracker.process(frame)
        tracker.draw(frame, hands)
        session.push_positions(positions_from_hands(hands))
        session.maybe_infer()
        session.poll()

        draw_text_lines(
            frame,
            session.overlay_lines(),
            color=overlay_color(session.status),
            scale=0.62,
            line_height=26,
        )
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return 0
        if key == KEY_SPACE:
            session.clear_sentence()
        elif key in (ord("t"), ord("T")):
            session.speak_sentence()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_local_env()
    settings = demo_settings()
    print("SignBridge is a hackathon prototype demonstrating temporal ASL")
    print("recognition for a limited vocabulary. It is not a complete")
    print("ASL translation system.")
    print(
        f"Demo mode: {settings['demo_mode']}   "
        f"confidence threshold: {settings['threshold']:.2f}"
    )

    model_path = args.model or SEQUENCE_MODEL_PATH
    labels_path = args.labels or SEQUENCE_LABELS_PATH
    if not Path(model_path).is_file():
        print(
            "Error: no trained sequence model found.\n"
            "Collect clips, then train:\n"
            "  python -m src.asl.sequence_data_collector\n"
            "  python -m src.asl.train_sequence_model",
            file=sys.stderr,
        )
        return 1

    try:
        model, classes, meta = load_sequence_model(model_path, labels_path)
    except Exception as exc:
        print(f"Error: could not load the sequence model: {exc}", file=sys.stderr)
        return 1

    val_acc = meta.get("best_val_accuracy")
    if isinstance(val_acc, (int, float)):
        print(f"Loaded {len(classes)} classes. Stored validation accuracy: {val_acc:.3f}")
    else:
        print(f"Loaded {len(classes)} classes.")

    opened = open_camera(args.camera)
    if opened is None:
        return 1
    camera, camera_name = opened
    print(f"Camera: {camera_name}")

    speaker = TextToSpeech.from_env()
    if not speaker.is_available():
        print(
            "Warning: ELEVENLABS_API_KEY is not set. Translations will still "
            "appear; nothing will be spoken.",
            file=sys.stderr,
        )

    recognizer = LiveSequenceRecognizer(
        model, classes, threshold=settings["threshold"]
    )
    executor = ThreadPoolExecutor(max_workers=1)
    session = RecognitionSession(
        recognizer, speaker=speaker, settings=settings, executor=executor
    )
    try:
        with HandTracker() as tracker:
            return run_loop(camera, tracker, session)
    except HandTrackerError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        executor.shutdown(wait=False)
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
