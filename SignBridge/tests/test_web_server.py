"""Web server uses the temporal sequence model, not the old pickle classifier."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.web.server import UNKNOWN_SIGN, RecognitionService, SignHold


class WebServerModelTests(unittest.TestCase):
    def test_missing_sequence_model_explains_how_to_train(self) -> None:
        with patch("src.web.server.HandTracker"):
            service = RecognitionService(model_path=Path("/no/such/asl_sequence_model.pt"))
        status = service.status()
        self.assertFalse(status["model_loaded"])
        self.assertIn("sequence_data_collector", service.model_error or "")
        self.assertIn("train_sequence_model", service.model_error or "")
        self.assertNotIn("asl_classifier.pkl", service.model_error or "")
        self.assertNotIn("src.asl.data_collector", service.model_error or "")
        self.assertNotIn("src.asl.train\n", service.model_error or "")

    def test_existing_sequence_checkpoint_is_loaded(self) -> None:
        handle = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        path = Path(handle.name)
        handle.close()
        self.addCleanup(path.unlink)
        fake_model = MagicMock()
        with patch("src.web.server.HandTracker"):
            with patch(
                "src.web.server.load_sequence_model",
                return_value=(fake_model, ["hello", "goodbye"], {"best_val_accuracy": 0.9}),
            ):
                service = RecognitionService(model_path=path, labels_path=path)
        status = service.status()
        self.assertTrue(status["model_loaded"])
        self.assertEqual(status["classes"], ["hello", "goodbye"])
        self.assertIsNone(status["error"])


class SignHoldTests(unittest.TestCase):
    def test_sign_stays_up_through_unknown_frames(self) -> None:
        now = {"t": 0.0}
        hold = SignHold(min_hits=3, hold_seconds=1.0, clock=lambda: now["t"])

        for _ in range(2):
            self.assertEqual(hold.update("hello", 0.9, inferred=True), UNKNOWN_SIGN)
        self.assertEqual(hold.update("hello", 0.91, inferred=True), "hello")

        now["t"] = 0.4
        self.assertEqual(hold.update(None, 0.0, inferred=False), "hello")
        self.assertEqual(hold.update(UNKNOWN_SIGN, 0.0, inferred=True), "hello")

        now["t"] = 1.5
        self.assertEqual(hold.update(UNKNOWN_SIGN, 0.0, inferred=True), UNKNOWN_SIGN)


if __name__ == "__main__":
    unittest.main()
