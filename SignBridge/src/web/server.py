"""SignBridge in the browser.

Run it with:

    python -m src.web.server

then open http://127.0.0.1:5000. The browser owns the webcam and posts JPEG
frames here. This server runs the same temporal BiGRU path as
`python -m src.asl.signbridge_demo`.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from src import SEQUENCE_LABELS_PATH, SEQUENCE_MODEL_PATH
from src.asl import SIGNS
from src.asl.sequence_features import positions_from_hands
from src.asl.sequence_model import load_sequence_model
from src.asl.sequence_recognition import (
    LiveSequenceRecognizer,
    RecognitionSession,
    demo_settings,
)
from src.speech.text_to_speech import TextToSpeech
from src.vision.hand_tracker import HandTracker, HandTrackerError

MAX_FRAME_BYTES = 4 * 1024 * 1024
UNKNOWN_SIGN = "unknown"
TRACK_MAX_WIDTH = 256


def web_settings() -> dict[str, Any]:
    """Live web UI settings — a bit less strict than the desktop demo defaults."""
    settings = demo_settings()
    # Only soften the threshold when the user has not set an explicit override.
    if "SIGNBRIDGE_CONFIDENCE_THRESHOLD" not in os.environ:
        settings["threshold"] = min(float(settings["threshold"]), 0.70)
    settings["window"] = 3
    settings["min_agree"] = 2
    settings["motion"] = min(float(settings["motion"]), 0.010)
    settings["inference_hz"] = max(float(settings["inference_hz"]), 8.0)
    settings["cooldown"] = min(float(settings["cooldown"]), 1.0)
    return settings


class _LandmarkPoint:
    """Minimal stand-in for a MediaPipe landmark (x, y, z)."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class _LandmarkList:
    __slots__ = ("landmark",)

    def __init__(self, points: list[_LandmarkPoint]) -> None:
        self.landmark = points


class _PayloadHand:
    """DetectedHand-compatible object built from a browser JSON payload."""

    __slots__ = ("label", "confidence", "landmarks")

    def __init__(self, label: str, confidence: float, points: list[list[float]]) -> None:
        self.label = label
        self.confidence = float(confidence)
        self.landmarks = _LandmarkList(
            [
                _LandmarkPoint(
                    point[0],
                    point[1],
                    point[2] if len(point) > 2 else 0.0,
                )
                for point in points
            ]
        )

    def points(self) -> list[tuple[float, float, float]]:
        return [(lm.x, lm.y, lm.z) for lm in self.landmarks.landmark]


def hands_from_payload(payload: Any) -> list[_PayloadHand]:
    """Parse browser landmark JSON into tracker-compatible hand objects."""
    if not isinstance(payload, list):
        raise ValueError("hands must be a list.")
    hands: list[_PayloadHand] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each hand must be an object.")
        points = item.get("points")
        if not isinstance(points, list) or len(points) != 21:
            raise ValueError("Each hand needs exactly 21 landmark points.")
        label = str(item.get("label") or "Unknown")
        confidence = float(item.get("confidence") or 0.0)
        hands.append(_PayloadHand(label, confidence, points))
    return hands


class SignHold:
    """Stop the banner flipping between a sign and unknown.

    A label is shown only after it wins several inferences in a row. Once it
    is showing, it stays up for `hold_seconds` after the model disagrees.
    Frames that skip inference leave the banner alone.
    """

    def __init__(
        self,
        min_hits: int = 3,
        hold_seconds: float = 1.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.min_hits = min_hits
        self.hold_seconds = hold_seconds
        self._clock = clock or time.perf_counter
        self.sign = UNKNOWN_SIGN
        self.confidence = 0.0
        self._pending: str | None = None
        self._hits = 0
        self._hold_until = 0.0

    def update(
        self,
        observed: str | None,
        confidence: float,
        *,
        inferred: bool,
    ) -> str:
        if not inferred:
            return self.sign
        now = self._clock()
        if observed and observed != UNKNOWN_SIGN:
            if observed == self._pending:
                self._hits += 1
            else:
                self._pending = observed
                self._hits = 1
            stable = self._hits >= self.min_hits or observed == self.sign
            if stable and (observed == self.sign or self._hits >= self.min_hits):
                self.sign = observed
                self.confidence = confidence
                self._hold_until = now + self.hold_seconds
            return self.sign
        self._pending = None
        self._hits = 0
        if now >= self._hold_until:
            self.sign = UNKNOWN_SIGN
            self.confidence = 0.0
        return self.sign


def shrink_for_tracking(frame: np.ndarray, max_width: int = TRACK_MAX_WIDTH) -> np.ndarray:
    """Downscale a webcam frame before MediaPipe. Landmarks stay in 0–1 space."""
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    return cv2.resize(
        frame,
        (max_width, max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


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
        self._settings = web_settings()
        self._tracker = HandTracker(max_hands=2)
        self._recognizer: LiveSequenceRecognizer | None = None
        self._session: RecognitionSession | None = None
        self._hold = SignHold(min_hits=2, hold_seconds=1.2)
        self.model_error: str | None = None
        self.reload_model()

    def reload_model(self) -> None:
        with self._lock:
            try:
                self._recognizer, classes = self._load_recognizer()
                self._session = RecognitionSession(
                    self._recognizer, settings=self._settings
                )
                self._hold = SignHold(min_hits=2, hold_seconds=1.2)
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

    def _recognize(
        self,
        hands: list[Any],
        threshold: float,
    ) -> tuple[str, float]:
        """Update the temporal buffer and return the held sign label."""
        sign, confidence = UNKNOWN_SIGN, 0.0
        session = self._session
        recognizer = self._recognizer
        if session is None or recognizer is None:
            return sign, confidence

        recognizer.threshold = threshold
        session.smoother.threshold = threshold
        session.push_positions(positions_from_hands(hands))
        prediction = session.maybe_infer()
        session.poll()
        inferred = prediction is not None or str(session.status).startswith("WAITING")
        if prediction is not None and prediction.label:
            observed: str | None = prediction.label
            observed_confidence = float(prediction.confidence)
        elif inferred:
            observed = UNKNOWN_SIGN
            observed_confidence = 0.0
        else:
            observed = None
            observed_confidence = float(session.confidence)
        sign = self._hold.update(observed, observed_confidence, inferred=inferred)
        confidence = self._hold.confidence if sign != UNKNOWN_SIGN else 0.0
        return sign, confidence

    def _response(self, hands: list[Any], sign: str, confidence: float) -> dict[str, Any]:
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

    def process(self, jpeg: bytes, threshold: float) -> dict[str, Any]:
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode the frame as an image.")
        frame = shrink_for_tracking(cv2.flip(frame, 1))

        with self._lock:
            hands = self._tracker.process(frame)
            sign, confidence = self._recognize(hands, threshold)
        return self._response(hands, sign, confidence)

    def process_hands(self, hands_payload: Any, threshold: float) -> dict[str, Any]:
        """Recognize from browser-side landmarks (skips JPEG + server MediaPipe)."""
        hands = hands_from_payload(hands_payload)
        with self._lock:
            sign, confidence = self._recognize(hands, threshold)
        return self._response(hands, sign, confidence)

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

    @app.post("/api/speak")
    def speak():
        payload = request.get_json(silent=True) or {}
        text = payload.get("text", "") if isinstance(payload, dict) else ""
        tts = TextToSpeech.from_env()
        wav = tts.synthesize_wav(str(text))
        if wav is None:
            return jsonify({"error": tts.last_error or "Speech failed."}), 400
        return Response(wav, mimetype="audio/wav")

    @app.post("/api/frame")
    def frame():
        threshold = request.args.get(
            "threshold", float(web_settings()["threshold"]), type=float
        )
        threshold = min(max(threshold, 0.0), 1.0)
        try:
            if request.is_json:
                payload = request.get_json(silent=True) or {}
                if isinstance(payload, dict) and "threshold" in payload:
                    threshold = min(max(float(payload["threshold"]), 0.0), 1.0)
                hands_payload = (
                    payload.get("hands", payload) if isinstance(payload, dict) else payload
                )
                return jsonify(service.process_hands(hands_payload, threshold))
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
