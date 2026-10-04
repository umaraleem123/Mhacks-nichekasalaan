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

from src.web.server import UNKNOWN_SIGN, RecognitionService, SignHold, create_app


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


class SpeakEndpointTests(unittest.TestCase):
    def test_speak_returns_elevenlabs_wav(self) -> None:
        handle = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        path = Path(handle.name)
        handle.close()
        self.addCleanup(path.unlink)
        fake_wav = b"RIFF....WAVE"
        tts = MagicMock()
        tts.synthesize_wav.return_value = fake_wav
        tts.last_error = None
        with patch("src.web.server.HandTracker"):
            with patch(
                "src.web.server.load_sequence_model",
                return_value=(MagicMock(), ["hello"], {}),
            ):
                service = RecognitionService(model_path=path, labels_path=path)
        app = create_app(service)
        with patch("src.web.server.TextToSpeech.from_env", return_value=tts):
            response = app.test_client().post("/api/speak", json={"text": "Hello"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "audio/wav")
        self.assertEqual(response.data, fake_wav)
        tts.synthesize_wav.assert_called_once_with("Hello")

    def test_legacy_elevenlabs_voice_id_is_not_in_source(self) -> None:
        root = Path(__file__).resolve().parents[1]
        forbidden = "Tcm4TlvDq8ikWAM"
        scanned = list((root / "src").rglob("*"))
        scanned += [root / "README.md", root / ".env.example"]
        hits = []
        for path in scanned:
            if not path.is_file() or path.suffix not in {".py", ".js", ".html", ".md", ".example", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if forbidden in text or "speechSynthesis" in text:
                hits.append(str(path))
        self.assertEqual(hits, [])


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
