"""SignBridge in the browser.

Run it with:

    python -m src.web.server

then open http://127.0.0.1:5000. The browser owns the webcam and posts JPEG
frames here; this server runs the same `HandTracker` and `SignRecognizer` as
`src.asl.live_recognition`, so predictions match the desktop app exactly.
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

from src.asl import SIGNS
from src.asl.features import FeatureError, extract_features
from src.asl.recognizer import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    UNKNOWN_SIGN,
    RecognizerError,
    SignRecognizer,
)
from src.vision.hand_tracker import HandTracker, HandTrackerError

MAX_FRAME_BYTES = 4 * 1024 * 1024


class RecognitionService:
    """Owns the tracker and the model; one frame at a time.

    MediaPipe's video mode keeps state between frames and is not thread-safe,
    so every call goes through a single lock.
    """

    def __init__(self, model_path: Path | None) -> None:
        self._lock = threading.Lock()
        self._model_path = model_path
        self._tracker = HandTracker(max_hands=2)
        self._recognizer: SignRecognizer | None = None
        self.model_error: str | None = None
        self.reload_model()

    def reload_model(self) -> None:
        with self._lock:
            try:
                self._recognizer = SignRecognizer(self._model_path)
                self.model_error = None
            except RecognizerError as exc:
                self._recognizer = None
                self.model_error = str(exc)

    def status(self) -> dict[str, Any]:
        recognizer = self._recognizer
        return {
            "model_loaded": recognizer is not None,
            "classes": recognizer.classes if recognizer else list(SIGNS),
            "default_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
            "error": self.model_error,
        }

    def process(self, jpeg: bytes, threshold: float) -> dict[str, Any]:
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode the frame as an image.")
        frame = cv2.flip(frame, 1)  # selfie view, and what handedness assumes

        with self._lock:
            hands = self._tracker.process(frame)
            sign, confidence = UNKNOWN_SIGN, 0.0
            if hands and self._recognizer is not None:
                self._recognizer.confidence_threshold = threshold
                try:
                    sign, confidence = self._recognizer.predict(extract_features(hands[0]))
                except FeatureError:
                    pass

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
        threshold = request.args.get("threshold", DEFAULT_CONFIDENCE_THRESHOLD, type=float)
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
    parser.add_argument("--model", type=Path, default=None, help="trained model path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        service = RecognitionService(args.model)
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
