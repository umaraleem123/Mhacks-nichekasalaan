"""Load the trained classifier and turn features into a predicted sign.

Deliberately free of OpenCV and MediaPipe: it takes a feature vector and
returns a label, so it can be reused by any front end later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src import ASL_MODEL_PATH
from src.asl.features import FEATURE_DIM, validate_feature_vector

UNKNOWN_SIGN = "unknown"
DEFAULT_CONFIDENCE_THRESHOLD = 0.6


class RecognizerError(RuntimeError):
    """The trained model is missing or cannot be used."""


class SignRecognizer:
    """Predicts a sign from a feature vector, or reports `unknown`.

    Predictions below `confidence_threshold` return `unknown` rather than a
    forced guess, which keeps the live display honest between signs.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.model_path = Path(model_path) if model_path else ASL_MODEL_PATH
        self.confidence_threshold = confidence_threshold

        if not self.model_path.is_file():
            raise RecognizerError(
                f"No trained model at {self.model_path}.\n"
                "Collect data and train one first:\n"
                "  python -m src.asl.data_collector\n"
                "  python -m src.asl.train"
            )

        try:
            bundle: dict[str, Any] = joblib.load(self.model_path)
            self._model = bundle["model"]
            self.classes: list[str] = list(bundle["classes"])
            stored_dim = int(bundle.get("feature_dim", FEATURE_DIM))
        except Exception as exc:
            raise RecognizerError(
                f"Could not load the model at {self.model_path}: {exc}\n"
                "Retrain it with:  python -m src.asl.train"
            ) from exc

        if stored_dim != FEATURE_DIM:
            raise RecognizerError(
                f"The model expects {stored_dim} features but this build "
                f"produces {FEATURE_DIM}. Retrain with the current code:\n"
                "  python -m src.asl.train"
            )

    def predict(self, features: Any) -> tuple[str, float]:
        """Return (sign, confidence). The sign is `unknown` below threshold."""
        vector = validate_feature_vector(features).reshape(1, -1)

        if hasattr(self._model, "predict_proba"):
            probabilities = self._model.predict_proba(vector)[0]
            best = int(np.argmax(probabilities))
            label = str(self._model.classes_[best])
            confidence = float(probabilities[best])
        else:
            label = str(self._model.predict(vector)[0])
            confidence = 1.0

        if confidence < self.confidence_threshold:
            return UNKNOWN_SIGN, confidence
        return label, confidence
