"""SignBridge in the browser.

Run it with:

    python -m src.web.server

then open http://127.0.0.1:5000. The browser owns the webcam and posts JPEG
frames here. This server runs the same temporal BiGRU path as
`python -m src.asl.signbridge_demo`.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from src import SEQUENCE_LABELS_PATH, SEQUENCE_MODEL_PATH
from src.asl import SIGNS
from src.asl.sequence_features import positions_from_hands
from src.asl.sequence_model import load_sequence_model
from src.asl.sequence_recognition import (
    LiveSequenceRecognizer,
    RecognitionSession,
    demo_settings,
)
from src.vision.hand_tracker import HandTracker, HandTrackerError

MAX_FRAME_BYTES = 4 * 1024 * 1024
UNKNOWN_SIGN = "unknown"


class SequenceModelError(RuntimeError):
    """The temporal model is missing or cannot be loaded."""


class RecognitionService:
    """Owns the tracker, rolling landmark buffer, and sequence model.

    MediaPipe's video mode keeps state between frames and is not thread-safe,
    so every call goes through a single lock.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        labels_path: Path | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._model_path = Path(model_path) if model_path else SEQUENCE_MODEL_PATH
        self._labels_path = Path(labels_path) if labels_path else SEQUENCE_LABELS_PATH
        self._settings = demo_settings()
        self._tracker = HandTracker(max_hands=2)
        self._recognizer: LiveSequenceRecognizer | None = None
        self._session: RecognitionSession | None = None
        self.model_error: str | None = None
        self.reload_model()

    def reload_model(self) -> None:
        with self._lock:
            try:
                self._recognizer, classes = self._load_recognizer()
                self._session = RecognitionSession(
                    self._recognizer, settings=self._settings
                )
                self.model_error = None
                self._classes = classes
            except SequenceModelError as exc:
                self._recognizer = None
                self._session = None
                self._classes = list(SIGNS)
                self.model_error = str(exc)

    def _load_recognizer(self) -> tuple[LiveSequenceRecognizer, list[str]]:
        if not self._model_path.is_file():
            raise SequenceModelError(
                f"No trained sequence model at {self._model_path}.\n"
                "Collect clips, then train:\n"
                "  python -m src.asl.sequence_data_collector\n"
                "  python -m src.asl.train_sequence_model"
            )
        try:
            model, classes, _meta = load_sequence_model(
                self._model_path, self._labels_path
            )
        except Exception as exc:
            raise SequenceModelError(
                f"Could not load the sequence model at {self._model_path}: {exc}\n"
                "Retrain it with:  python -m src.asl.train_sequence_model"
            ) from exc
        recognizer = LiveSequenceRecognizer(
            model, classes, threshold=float(self._settings["threshold"])
        )
        return recognizer, list(classes)

    def status(self) -> dict[str, Any]:
        return {
            "model_loaded": self._recognizer is not None,
            "classes": list(getattr(self, "_classes", SIGNS)),
            "default_threshold": float(self._settings["threshold"]),
            "error": self.model_error,
        }

    def process(self, jpeg: bytes, threshold: float) -> dict[str, Any]:
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode the frame as an image.")
        frame = cv2.flip(frame, 1)

        with self._lock:
            hands = self._tracker.process(frame)
            sign, confidence = UNKNOWN_SIGN, 0.0
            session = self._session
            recognizer = self._recognizer
            if session is not None and recognizer is not None:
                recognizer.threshold = threshold
                session.smoother.threshold = threshold
                session.push_positions(positions_from_hands(hands))
                prediction = session.maybe_infer()
                session.poll()
                confidence = float(session.confidence)
                if prediction is not None and prediction.label:
                    sign = prediction.label

        return {
            "hands": [
                {
                    "label": hand.label,
                    "confidence": hand.confidence,
                    "points": [[x, y] for x, y, _z in hand.points()],
                }
                for hand in hands
            ],
            "sign": sign,
            "confidence": confidence,
            "model_loaded": self._recognizer is not None,
        }

    def close(self) -> None:
        self._tracker.close()


def create_app(service: RecognitionService) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_FRAME_BYTES

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/api/status")
    def status():
        return jsonify(service.status())

    @app.post("/api/reload")
    def reload_model():
        service.reload_model()
        return jsonify(service.status())

    @app.post("/api/frame")
    def frame():
        threshold = request.args.get(
            "threshold", float(demo_settings()["threshold"]), type=float
        )
        threshold = min(max(threshold, 0.0), 1.0)
        try:
            return jsonify(service.process(request.get_data(), threshold))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except HandTrackerError as exc:
            return jsonify({"error": str(exc)}), 500

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the SignBridge web app.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--model", type=Path, default=None, help="trained sequence model path")
    parser.add_argument("--labels", type=Path, default=None, help="sequence labels JSON path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        service = RecognitionService(args.model, args.labels)
    except HandTrackerError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if service.model_error:
        print(f"Warning: {service.model_error}\nServing hand tracking only.", file=sys.stderr)

    print(f"SignBridge is running at http://{args.host}:{args.port}")
    try:
        create_app(service).run(host=args.host, port=args.port, threaded=True)
    finally:
        service.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
