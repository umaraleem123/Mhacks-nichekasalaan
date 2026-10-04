"""Pretrained isolated ASL pipeline tests — no Hugging Face download, no ElevenLabs."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.asl.pretrained_features import (
    FEATURE_DIM,
    LANDMARK_COUNT,
    POSITION_DIM,
    SEQ_LEN,
    VELOCITY_DIM,
    extract_75_landmarks,
    interpolate_positions,
    landmark_list_to_array,
    model_input_from_positions,
    pad_or_interpolate,
    position_vector,
    position_velocity,
    sequence_features,
)
from src.asl.pretrained_recognizer import (
    CHECKPOINT_FILENAME,
    HF_REPO_ID,
    LABELS_FILENAME,
    IsolatedSignBiGRU,
    LandmarkBuffer,
    PretrainedRecognizer,
    PredictionStabilizer,
    SentenceBuffer,
    download_pretrained_files,
    label_to_spoken_english,
    load_id_to_label,
    resolve_pretrained_paths,
    softmax_probs,
)
from src.asl.pretrained_demo import IsolatedDemoSession, Status
from src.speech.text_to_speech import TextToSpeech, clean_text, speak


class Landmark:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class LandmarkList:
    def __init__(self, count: int, fill: float) -> None:
        self.landmark = [Landmark(fill, fill + 0.01, fill + 0.02) for _ in range(count)]


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class ImmediateExecutor:
    def submit(self, fn, *args, **kwargs):
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, wait: bool = False) -> None:
        return None


class StubSpeaker:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str) -> bool:
        self.spoken.append(text)
        return True


class StubModel(torch.nn.Module):
    """Returns a one-hot-ish logit vector without touching a real checkpoint."""

    def __init__(self, class_id: int = 105, num_classes: int = 200, logit: float = 8.0) -> None:
        super().__init__()
        self.class_id = class_id
        self.num_classes = num_classes
        self.logit = logit
        self.calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        batch = inputs.shape[0]
        logits = torch.zeros(batch, self.num_classes)
        logits[:, self.class_id] = self.logit
        return logits


def fake_hub_download(repo_id: str, filename: str, local_dir: str, **_kwargs: object) -> str:
    dest = Path(local_dir) / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("placeholder", encoding="utf-8")
    return str(dest)


class FeatureTests(unittest.TestCase):
    def test_75_landmark_extraction_and_missing_hands(self) -> None:
        pose = LandmarkList(33, 0.4)
        left = LandmarkList(21, 0.2)
        right = LandmarkList(21, 0.8)
        points = extract_75_landmarks(pose, left, right)
        self.assertEqual(points.shape, (LANDMARK_COUNT, 3))
        missing = extract_75_landmarks(pose, None, None)
        self.assertEqual(missing.shape, (75, 3))
        self.assertTrue(np.allclose(missing[33:], 0.0))

    def test_position_and_velocity_and_450_features(self) -> None:
        pose = LandmarkList(33, 0.5)
        left = LandmarkList(21, 0.1)
        right = LandmarkList(21, 0.9)
        one = position_vector(pose, left, right)
        self.assertEqual(one.shape, (POSITION_DIM,))
        self.assertEqual(POSITION_DIM, 225)
        self.assertEqual(VELOCITY_DIM, 225)
        self.assertEqual(FEATURE_DIM, 450)

        simple = np.zeros((3, POSITION_DIM), dtype=np.float32)
        simple[1] = 0.05
        simple[2] = 0.10
        simple_velocity = position_velocity(simple)
        self.assertTrue(np.allclose(simple_velocity[0], 0.0))
        self.assertTrue(np.allclose(simple_velocity[1], 0.05))
        self.assertTrue(np.allclose(simple_velocity[2], 0.05))

        features = sequence_features(simple)
        self.assertEqual(features.shape, (3, FEATURE_DIM))
        self.assertTrue(np.allclose(features[:, :POSITION_DIM], simple))
        self.assertTrue(np.allclose(features[:, POSITION_DIM:], simple_velocity))

    def test_interpolate_and_pad_to_200(self) -> None:
        short = np.linspace(0.0, 1.0, 12 * POSITION_DIM, dtype=np.float32).reshape(12, POSITION_DIM)
        resampled = interpolate_positions(short, length=SEQ_LEN)
        self.assertEqual(resampled.shape, (SEQ_LEN, POSITION_DIM))
        self.assertTrue(np.allclose(resampled[0], short[0]))
        self.assertTrue(np.allclose(resampled[-1], short[-1]))

        empty = pad_or_interpolate(np.zeros((0, POSITION_DIM), dtype=np.float32))
        self.assertEqual(empty.shape, (200, 225))

        single = interpolate_positions(short[:1], length=200)
        self.assertEqual(single.shape, (200, 225))
        self.assertTrue(np.allclose(single[0], single[-1]))

        model_in = model_input_from_positions(short)
        self.assertEqual(model_in.shape, (SEQ_LEN, FEATURE_DIM))

    def test_landmark_list_to_array_zeros_when_missing(self) -> None:
        zeros = landmark_list_to_array(None, 21)
        self.assertEqual(zeros.shape, (21, 3))
        self.assertTrue(np.allclose(zeros, 0.0))


class BufferAndPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp(prefix="signbridge_pretrained_"))

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_huggingface_path_resolution_uses_local_files(self) -> None:
        checkpoint = self._tmp / CHECKPOINT_FILENAME
        labels = self._tmp / LABELS_FILENAME
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        labels.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"ckpt")
        labels.write_text("{}", encoding="utf-8")
        paths = resolve_pretrained_paths(cache_dir=self._tmp, download=False)
        self.assertEqual(paths.checkpoint, checkpoint)
        self.assertEqual(paths.labels, labels)

    def test_download_writes_expected_filenames(self) -> None:
        calls: list[tuple[str, str]] = []

        def hub(repo_id: str, filename: str, local_dir: str, **_kwargs: object) -> str:
            calls.append((repo_id, filename))
            return fake_hub_download(repo_id, filename, local_dir)

        paths = download_pretrained_files(cache_dir=self._tmp, hub_download=hub)
        self.assertEqual([item[0] for item in calls], [HF_REPO_ID, HF_REPO_ID])
        self.assertEqual(calls[0][1], CHECKPOINT_FILENAME)
        self.assertEqual(calls[1][1], LABELS_FILENAME)
        self.assertTrue(paths.checkpoint.is_file())
        self.assertTrue(paths.labels.is_file())

    def test_label_mapping_loading(self) -> None:
        path = self._tmp / "id_to_label.json"
        path.write_text(json.dumps({"105": "HELLO", "0": "ACTION"}), encoding="utf-8")
        mapping = load_id_to_label(path)
        self.assertEqual(mapping[105], "HELLO")
        self.assertEqual(mapping[0], "ACTION")
        self.assertEqual(len(mapping), 2)

    def test_temporal_buffer_resizes_to_200x450(self) -> None:
        buffer = LandmarkBuffer(max_frames=40)
        frame = np.linspace(0.0, 1.0, POSITION_DIM, dtype=np.float32)
        for offset in range(20):
            buffer.append(frame + offset * 0.01)
        self.assertEqual(len(buffer), 20)
        model_in = buffer.to_model_input()
        self.assertEqual(model_in.shape, (200, 450))


class ModelAndConfidenceTests(unittest.TestCase):
    def test_model_output_shape(self) -> None:
        model = IsolatedSignBiGRU()
        model.eval()
        dummy = torch.zeros(2, SEQ_LEN, FEATURE_DIM)
        with torch.no_grad():
            logits = model(dummy)
        self.assertEqual(tuple(logits.shape), (2, 200))

    def test_confidence_softmax_and_unknown_threshold(self) -> None:
        logits = np.array([1.0, 3.0, 0.0], dtype=np.float32)
        probs = softmax_probs(logits)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=6)
        self.assertEqual(int(np.argmax(probs)), 1)

        mapping = {105: "HELLO", 0: "ACTION"}
        model = StubModel(class_id=105, logit=0.2)
        recognizer = PretrainedRecognizer(model, mapping, device="cpu", threshold=0.70)
        features = np.zeros((SEQ_LEN, FEATURE_DIM), dtype=np.float32)
        prediction = recognizer.predict_tensor(features)
        self.assertIsNone(prediction.label)
        self.assertEqual(prediction.display_label, "Unknown")
        self.assertLess(prediction.confidence, 0.70)

        confident = StubModel(class_id=105, logit=8.0)
        recognizer = PretrainedRecognizer(confident, mapping, device="cpu", threshold=0.70)
        prediction = recognizer.predict_tensor(features)
        self.assertEqual(prediction.label, "HELLO")
        self.assertGreaterEqual(prediction.confidence, 0.70)

    def test_prediction_stabilization(self) -> None:
        stabilizer = PredictionStabilizer(consecutive=3, threshold=0.70)
        results = [
            stabilizer.update("HELLO", 0.91, "HELLO"),
            stabilizer.update("HELLO", 0.93, "HELLO"),
            stabilizer.update("HELLO", 0.94, "HELLO"),
        ]
        self.assertFalse(results[0].should_speak)
        self.assertFalse(results[1].should_speak)
        self.assertTrue(results[2].should_speak)
        self.assertEqual(results[2].accepted_label, "HELLO")

        fourth = stabilizer.update("HELLO", 0.92, "HELLO")
        self.assertFalse(fourth.should_speak)

    def test_duplicate_tts_prevention_until_change(self) -> None:
        stabilizer = PredictionStabilizer(consecutive=2, threshold=0.70)
        first = stabilizer.update("HELLO", 0.9, "HELLO")
        accepted = stabilizer.update("HELLO", 0.91, "HELLO")
        self.assertFalse(first.should_speak)
        self.assertTrue(accepted.should_speak)

        again = stabilizer.update("HELLO", 0.95, "HELLO")
        self.assertFalse(again.should_speak)

        unknown = stabilizer.update(None, 0.2, "Unknown")
        self.assertFalse(unknown.should_speak)
        self.assertEqual(unknown.display_label, "Unknown")

        hello_again = stabilizer.update("HELLO", 0.9, "HELLO")
        spoken = stabilizer.update("HELLO", 0.91, "HELLO")
        self.assertFalse(hello_again.should_speak)
        self.assertTrue(spoken.should_speak)

    def test_sentence_buffer_is_just_a_list(self) -> None:
        buffer = SentenceBuffer()
        buffer.add("HELLO")
        buffer.add("HOW")
        buffer.add("YOU")
        self.assertEqual(buffer.display(), "HELLO + HOW + YOU")
        buffer.add("YOU")
        self.assertEqual(buffer.display(), "HELLO + HOW + YOU")

    def test_label_to_spoken_english(self) -> None:
        self.assertEqual(label_to_spoken_english("HELLO"), "Hello")
        self.assertEqual(label_to_spoken_english("WHAT1"), "What")


class DemoAndSpeechTests(unittest.TestCase):
    def test_demo_speaks_once_after_stable_inferences(self) -> None:
        mapping = {105: "HELLO"}
        recognizer = PretrainedRecognizer(
            StubModel(class_id=105, logit=8.0),
            mapping,
            device="cpu",
            threshold=0.70,
        )
        speaker = StubSpeaker()
        clock = FakeClock()
        session = IsolatedDemoSession(
            recognizer,
            speaker=speaker,
            executor=ImmediateExecutor(),
            clock=clock,
            inference_hz=4.0,
            min_frames=3,
            stable_count=3,
        )
        frame = np.linspace(0.0, 1.0, POSITION_DIM, dtype=np.float32)
        for index in range(12):
            session.push_positions(frame + index * 0.01)
            session.maybe_infer()
            session.poll()
            clock.advance(0.3)

        self.assertEqual(speaker.spoken, ["Hello"])
        self.assertEqual(session.detected_sign, "HELLO")
        self.assertEqual(session.status, Status.READY)
        self.assertIn("HELLO", session.sentence.display())

        for index in range(8):
            session.push_positions(frame)
            session.maybe_infer()
            session.poll()
            clock.advance(0.3)
        self.assertEqual(speaker.spoken, ["Hello"])

    def test_speak_skips_empty_and_missing_key(self) -> None:
        tts = TextToSpeech(api_key=None)
        self.assertFalse(tts.speak(""))
        self.assertEqual(tts.last_error, "Nothing to speak.")
        self.assertFalse(tts.speak("hello"))
        self.assertIn("ELEVENLABS_API_KEY", tts.last_error or "")
        self.assertNotIn("sk_", tts.last_error or "")

    def test_speak_mocks_elevenlabs(self) -> None:
        tts = TextToSpeech(api_key="test-key", voice_id="voice")
        pcm = (np.zeros(22050, dtype=np.int16)).tobytes()
        with patch.object(TextToSpeech, "_synthesize", return_value=pcm) as synth:
            with patch.object(TextToSpeech, "_play_pcm16") as play:
                self.assertTrue(tts.speak("Hello"))
        synth.assert_called_once()
        play.assert_called_once()
        self.assertEqual(tts.last_backend, "elevenlabs")
        args, _kwargs = synth.call_args
        self.assertEqual(args[0], "Hello")

    def test_module_speak_uses_from_env(self) -> None:
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"}, clear=False):
            with patch.object(TextToSpeech, "speak", return_value=True) as mocked:
                self.assertTrue(speak("Hi"))
        mocked.assert_called()

    def test_clean_text(self) -> None:
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text("Hello   there"), "Hello there")

    def test_default_voice_id_is_current(self) -> None:
        from src.speech.text_to_speech import DEFAULT_VOICE_ID

        self.assertEqual(DEFAULT_VOICE_ID, "rWZM1pGWKmpGt3Hvergd")

    def test_load_local_env_overwrites_existing_voice_id(self) -> None:
        from src.speech.text_to_speech import load_local_env

        root = Path(tempfile.mkdtemp())
        (root / ".env").write_text(
            "ELEVENLABS_VOICE_ID=rWZM1pGWKmpGt3Hvergd\n",
            encoding="utf-8",
        )
        with patch.dict(os.environ, {"ELEVENLABS_VOICE_ID": "old-voice"}, clear=False):
            load_local_env(root)
            self.assertEqual(os.environ["ELEVENLABS_VOICE_ID"], "rWZM1pGWKmpGt3Hvergd")

    def test_synthesize_wav_wraps_pcm(self) -> None:
        tts = TextToSpeech(api_key="test-key", voice_id="voice")
        pcm = (np.zeros(22050, dtype=np.int16)).tobytes()
        with patch.object(TextToSpeech, "_synthesize", return_value=pcm):
            wav = tts.synthesize_wav("Hello")
        self.assertIsNotNone(wav)
        self.assertTrue((wav or b"").startswith(b"RIFF"))
        self.assertEqual(tts.last_backend, "elevenlabs")


if __name__ == "__main__":
    unittest.main()
