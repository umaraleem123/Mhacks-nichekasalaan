"""Local temporal ASL pipeline tests — no webcam, no ElevenLabs, no downloads."""

from __future__ import annotations

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

from src.asl.phrase_mapper import format_sentence, label_to_english
from src.asl.sequence_data_collector import SequenceRecorder
from src.asl.sequence_dataset import (
    SequenceDataError,
    SequenceExample,
    augment_sequence,
    load_all_sequences,
    load_sequence,
    mirror_example,
    pad_batch,
    save_sequence,
    split_by_sequence,
    with_mirrored_copies,
)
from src.asl.sequence_features import (
    FEATURE_DIM,
    HAND_DIM,
    POSITION_DIM,
    VELOCITY_DIM,
    features_from_position_sequence,
    frame_features,
    landmark_list_to_points,
    mirror_positions,
    mirror_sequence_features,
    motion_energy,
    normalize_hand,
    position_velocity,
    positions_from_hands,
    two_hand_positions,
)
from src.asl.sequence_model import SequenceBiGRU, softmax_probs
from src.asl.sequence_recognition import (
    Cooldown,
    LiveSequenceRecognizer,
    MotionGate,
    PredictionSmoother,
    RecognitionSession,
    SentenceBuffer,
)
from src.asl.train_sequence_model import train
from src.speech.text_to_speech import TextToSpeech, speak


class Landmark:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x, self.y, self.z = x, y, z


class LandmarkList:
    def __init__(self, offset: float = 0.2) -> None:
        self.landmark = [
            Landmark(offset, offset + i * 0.01, offset * 0.1) for i in range(21)
        ]


class Hand:
    def __init__(self, label: str, offset: float) -> None:
        self.label = label
        self.landmarks = LandmarkList(offset)


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
    def __init__(self, class_id: int = 0, num_classes: int = 2, logit: float = 8.0) -> None:
        super().__init__()
        self.class_id = class_id
        self.num_classes = num_classes
        self.logit = logit

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch = inputs.shape[0]
        logits = torch.zeros(batch, self.num_classes)
        logits[:, self.class_id] = self.logit
        return logits


def moving_positions(frames: int = 20, phase: float = 0.0) -> np.ndarray:
    clip = np.zeros((frames, POSITION_DIM), dtype=np.float32)
    for index in range(frames):
        clip[index, :HAND_DIM] = 0.12 * np.sin(index * 0.5 + phase)
        clip[index, HAND_DIM:] = 0.10 * np.cos(index * 0.4 + phase)
    return clip


class FeatureTests(unittest.TestCase):
    def test_two_hand_and_missing_hand(self) -> None:
        left = np.full((21, 3), 0.2, dtype=np.float32)
        left[:, 1] += np.linspace(0.0, 0.3, 21, dtype=np.float32)
        right = np.full((21, 3), 0.7, dtype=np.float32)
        right[:, 1] += np.linspace(0.0, 0.25, 21, dtype=np.float32)
        both = two_hand_positions(left, right)
        self.assertEqual(both.shape, (POSITION_DIM,))
        missing_left = two_hand_positions(None, right)
        self.assertTrue(np.allclose(missing_left[:HAND_DIM], 0.0))
        self.assertFalse(np.allclose(missing_left[HAND_DIM:], 0.0))
        missing_right = two_hand_positions(left, None)
        self.assertTrue(np.allclose(missing_right[HAND_DIM:], 0.0))

    def test_positions_from_hands_list(self) -> None:
        vector = positions_from_hands([Hand("Left", 0.2), Hand("Right", 0.8)])
        self.assertEqual(vector.shape, (126,))
        none = positions_from_hands([])
        self.assertTrue(np.allclose(none, 0.0))

    def test_landmark_normalization_is_translation_invariant(self) -> None:
        points = np.zeros((21, 3), dtype=np.float32)
        points[:, 0] = np.linspace(0.1, 0.4, 21)
        points[9] = points[0] + np.array([0.2, 0.0, 0.0], dtype=np.float32)
        shifted = points + np.array([0.3, -0.2, 0.05], dtype=np.float32)
        a = normalize_hand(points)
        b = normalize_hand(shifted)
        self.assertTrue(np.allclose(a, b, atol=1e-5))
        self.assertTrue(np.allclose(a[0], 0.0))

    def test_velocity_and_252_features(self) -> None:
        self.assertEqual(POSITION_DIM, 126)
        self.assertEqual(VELOCITY_DIM, 126)
        self.assertEqual(FEATURE_DIM, 252)
        current = np.ones(POSITION_DIM, dtype=np.float32)
        previous = np.zeros(POSITION_DIM, dtype=np.float32)
        self.assertTrue(np.allclose(position_velocity(current, None), 0.0))
        self.assertTrue(np.allclose(position_velocity(current, previous), 1.0))
        frame = frame_features(current, previous)
        self.assertEqual(frame.shape, (252,))
        self.assertTrue(np.allclose(frame[:126], 1.0))
        self.assertTrue(np.allclose(frame[126:], 1.0))

        clip = moving_positions(8)
        features = features_from_position_sequence(clip)
        self.assertEqual(features.shape, (8, 252))
        self.assertTrue(np.allclose(features[0, 126:], 0.0))
        self.assertGreater(motion_energy(features), 0.0)


class DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="signbridge_seq_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_sequence_roundtrip_and_variable_lengths(self) -> None:
        short = np.zeros((9, FEATURE_DIM), dtype=np.float32)
        long = np.ones((21, FEATURE_DIM), dtype=np.float32)
        p1 = save_sequence(self.tmp / "hello" / "a.npz", short, "hello")
        p2 = save_sequence(self.tmp / "yes" / "b.npz", long, "yes")
        loaded = load_sequence(p1)
        self.assertEqual(loaded.label, "hello")
        self.assertEqual(loaded.sequence_length, 9)
        self.assertEqual(loaded.features.shape, (9, 252))
        self.assertEqual(load_sequence(p2).sequence_length, 21)

    def test_padding(self) -> None:
        a = np.zeros((5, FEATURE_DIM), dtype=np.float32)
        b = np.ones((12, FEATURE_DIM), dtype=np.float32)
        padded, lengths = pad_batch([a, b])
        self.assertEqual(padded.shape, (2, 12, 252))
        self.assertEqual(list(lengths), [5, 12])
        self.assertTrue(np.allclose(padded[0, 5:], 0.0))

    def test_split_is_by_sequence_and_covers_every_class(self) -> None:
        examples = []
        for label in ("hello", "yes"):
            for index in range(5):
                path = save_sequence(
                    self.tmp / label / f"{index}.npz",
                    np.zeros((6 + index, FEATURE_DIM), np.float32),
                    label,
                )
                examples.append(load_sequence(path))
        train, val = split_by_sequence(examples, val_ratio=0.2, seed=1)
        self.assertEqual(len(train) + len(val), 10)
        self.assertTrue({item.label for item in train} >= {"hello", "yes"})
        self.assertEqual({item.label for item in val}, {"hello", "yes"})

    def test_split_rejects_single_example_class(self) -> None:
        path = save_sequence(
            self.tmp / "hello" / "only.npz",
            np.zeros((8, FEATURE_DIM), np.float32),
            "hello",
        )
        with self.assertRaises(SequenceDataError):
            split_by_sequence([load_sequence(path)])

    def test_augment_changes_train_clip_not_shape_family(self) -> None:
        rng = np.random.default_rng(0)
        original = features_from_position_sequence(moving_positions(16))
        jittered = augment_sequence(original, rng=rng)
        self.assertEqual(jittered.shape[1], FEATURE_DIM)
        self.assertGreaterEqual(jittered.shape[0], 4)


def _asymmetric_positions(frames: int = 6) -> np.ndarray:
    """Left and right hands differ so a swap is observable."""
    clip = np.zeros((frames, POSITION_DIM), dtype=np.float32)
    for index in range(frames):
        # Left hand: x grows with landmark index; y with time.
        for landmark in range(21):
            clip[index, landmark * 3 + 0] = 0.05 * landmark
            clip[index, landmark * 3 + 1] = 0.10 + 0.02 * index
            clip[index, landmark * 3 + 2] = 0.03
        # Right hand: different x/y/z pattern.
        for landmark in range(21):
            base = HAND_DIM + landmark * 3
            clip[index, base + 0] = -0.04 * landmark - 0.01 * index
            clip[index, base + 1] = 0.20 + 0.01 * index
            clip[index, base + 2] = -0.07
    return clip


class MirrorAugmentationTests(unittest.TestCase):
    def test_mirror_twice_returns_original(self) -> None:
        original = features_from_position_sequence(_asymmetric_positions(8))
        once = mirror_sequence_features(original)
        twice = mirror_sequence_features(once)
        self.assertTrue(np.allclose(twice, original, atol=1e-5))

    def test_x_mirrored_y_z_unchanged_and_hands_swapped(self) -> None:
        positions = _asymmetric_positions(5)
        mirrored = mirror_positions(positions)
        orig_left = positions[:, :HAND_DIM].reshape(-1, 21, 3)
        orig_right = positions[:, HAND_DIM:].reshape(-1, 21, 3)
        mir_left = mirrored[:, :HAND_DIM].reshape(-1, 21, 3)
        mir_right = mirrored[:, HAND_DIM:].reshape(-1, 21, 3)
        # Image-space x' = 1-x, then wrist X is restored, so each hand is
        # reflected through its wrist: x' = 2*wrist_x - x. Then hands swap.
        right_wrist = orig_right[:, 0, 0][:, None]
        left_wrist = orig_left[:, 0, 0][:, None]
        self.assertTrue(
            np.allclose(mir_left[..., 0], 2.0 * right_wrist - orig_right[..., 0], atol=1e-5)
        )
        self.assertTrue(
            np.allclose(mir_right[..., 0], 2.0 * left_wrist - orig_left[..., 0], atol=1e-5)
        )
        self.assertTrue(np.allclose(mir_left[..., 1], orig_right[..., 1], atol=1e-5))
        self.assertTrue(np.allclose(mir_right[..., 1], orig_left[..., 1], atol=1e-5))
        self.assertTrue(np.allclose(mir_left[..., 2], orig_right[..., 2], atol=1e-5))
        self.assertTrue(np.allclose(mir_right[..., 2], orig_left[..., 2], atol=1e-5))

    def test_velocity_recomputed_after_mirror_length_and_label(self) -> None:
        positions = _asymmetric_positions(7)
        original = features_from_position_sequence(positions)
        mirrored = mirror_sequence_features(original)
        self.assertEqual(mirrored.shape, original.shape)
        mir_pos = mirrored[:, :POSITION_DIM]
        mir_vel = mirrored[:, POSITION_DIM:]
        self.assertTrue(np.allclose(mir_vel[0], 0.0))
        self.assertTrue(np.allclose(mir_vel[1:], mir_pos[1:] - mir_pos[:-1], atol=1e-5))
        # Not a naive copy/flip of the original velocity block.
        self.assertFalse(np.allclose(mir_vel, original[:, POSITION_DIM:], atol=1e-4))

        example = SequenceExample(
            path=Path("hello.npz"),
            features=original,
            label="hello",
            sequence_length=original.shape[0],
        )
        flipped = mirror_example(example)
        self.assertEqual(flipped.label, "hello")
        self.assertEqual(flipped.sequence_length, example.sequence_length)

    def test_dataset_doubles_train_only(self) -> None:
        originals = [
            SequenceExample(
                path=Path(f"{label}_{index}.npz"),
                features=features_from_position_sequence(_asymmetric_positions(5 + index)),
                label=label,
                sequence_length=5 + index,
            )
            for label in ("hello", "yes")
            for index in range(3)
        ]
        expanded, n_orig, n_mir = with_mirrored_copies(originals)
        self.assertEqual(n_orig, 6)
        self.assertEqual(n_mir, 6)
        self.assertEqual(len(expanded), 12)
        self.assertEqual([item.label for item in expanded[:6]], [item.label for item in originals])
        self.assertEqual([item.label for item in expanded[6:]], [item.label for item in originals])

    def test_training_skips_mirrors_when_disabled_and_never_mirrors_val(self) -> None:
        data = Path(tempfile.mkdtemp(prefix="signbridge_nomirror_"))
        try:
            for label, phase in (("hello", 0.0), ("yes", 1.2)):
                for index in range(4):
                    save_sequence(
                        data / label / f"{index}.npz",
                        features_from_position_sequence(_asymmetric_positions(8 + index)),
                        label,
                    )
            with patch.dict(os.environ, {"SIGNBRIDGE_MIRROR_AUGMENTATION": "false"}):
                result = train(
                    data_dir=data,
                    epochs=1,
                    batch_size=4,
                    seed=0,
                    model_path=data / "model.pt",
                    labels_path=data / "labels.json",
                )
            self.assertEqual(result["mirrored_train_sequences"], 0)
            self.assertEqual(
                result["train_sequences"], result["original_train_sequences"]
            )
            self.assertEqual(
                result["original_train_sequences"] + result["val_sequences"], 8
            )
        finally:
            shutil.rmtree(data, ignore_errors=True)


class CollectorAndPhraseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="signbridge_col_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_recorder_saves_one_sequence(self) -> None:
        clock = FakeClock()
        recorder = SequenceRecorder(
            self.tmp, max_seconds=2.5, min_frames=5, clock=clock
        )
        recorder.start("hello")
        pos = moving_positions(12)
        for index, row in enumerate(pos):
            recorder.add_positions(row)
            clock.advance(0.05)
        path = recorder.stop(save=True)
        self.assertIsNotNone(path)
        loaded = load_sequence(path)
        self.assertEqual(loaded.label, "hello")
        self.assertEqual(loaded.features.shape[1], 252)
        self.assertGreaterEqual(loaded.sequence_length, 5)

    def test_phrase_mapping_and_sentence(self) -> None:
        self.assertEqual(label_to_english("HELLO"), "Hello")
        self.assertEqual(label_to_english("thank_you"), "Thank you")
        self.assertEqual(label_to_english("HOW_ARE_YOU"), "How are you?")
        self.assertEqual(
            format_sentence(["hello", "how_are_you"]),
            "Hello, how are you?",
        )
        buffer = SentenceBuffer()
        self.assertTrue(buffer.add("hello"))
        self.assertFalse(buffer.add("hello"))
        self.assertTrue(buffer.add("how_are_you"))
        self.assertEqual(buffer.english(), "Hello, how are you?")
        buffer.clear()
        self.assertEqual(buffer.english(), "")


class ModelAndLiveTests(unittest.TestCase):
    def test_model_forward_pass_variable_lengths(self) -> None:
        model = SequenceBiGRU(num_classes=10)
        model.eval()
        padded, lengths = pad_batch(
            [
                np.zeros((7, FEATURE_DIM), np.float32),
                np.ones((15, FEATURE_DIM), np.float32),
            ]
        )
        with torch.no_grad():
            logits = model(torch.from_numpy(padded), torch.from_numpy(lengths))
        self.assertEqual(tuple(logits.shape), (2, 10))

    def test_confidence_threshold_unknown(self) -> None:
        recognizer = LiveSequenceRecognizer(
            StubModel(class_id=0, logit=0.2),
            classes=["hello", "yes"],
            threshold=0.75,
        )
        pred = recognizer.predict(np.zeros((10, FEATURE_DIM), np.float32))
        self.assertIsNone(pred.label)
        self.assertEqual(pred.display_label, "Unknown")
        confident = LiveSequenceRecognizer(
            StubModel(class_id=0, logit=8.0),
            classes=["hello", "yes"],
            threshold=0.75,
        )
        pred = confident.predict(np.zeros((10, FEATURE_DIM), np.float32))
        self.assertEqual(pred.label, "hello")
        self.assertGreaterEqual(pred.confidence, 0.75)

    def test_temporal_smoothing(self) -> None:
        smoother = PredictionSmoother(window=5, min_agree=3, threshold=0.75)
        self.assertIsNone(smoother.update("hello", 0.88))
        self.assertIsNone(smoother.update("hello", 0.91))
        self.assertEqual(smoother.update("hello", 0.94), "hello")
        mixed = PredictionSmoother(window=5, min_agree=3, threshold=0.75)
        mixed.update("hello", 0.51)
        mixed.update("yes", 0.49)
        mixed.update("hello", 0.52)
        self.assertIsNone(mixed.update("hello", 0.50))

    def test_motion_gate(self) -> None:
        gate = MotionGate(threshold=0.01)
        still = np.zeros((12, FEATURE_DIM), np.float32)
        moving = features_from_position_sequence(moving_positions(12))
        self.assertFalse(gate.is_signing(still))
        self.assertTrue(gate.is_signing(moving))

    def test_cooldown_and_duplicate_suppression(self) -> None:
        clock = FakeClock()
        cool = Cooldown(seconds=1.0, clock=clock)
        self.assertFalse(cool.active())
        cool.trigger()
        self.assertTrue(cool.active())
        clock.advance(1.1)
        self.assertFalse(cool.active())

        recognizer = LiveSequenceRecognizer(
            StubModel(class_id=0, logit=8.0),
            classes=["hello", "yes"],
            threshold=0.75,
        )
        speaker = StubSpeaker()
        session = RecognitionSession(
            recognizer,
            speaker=speaker,
            clock=clock,
            executor=ImmediateExecutor(),
            settings={
                "demo_mode": False,
                "threshold": 0.75,
                "window": 5,
                "min_agree": 3,
                "cooldown": 0.8,
                "motion": 0.001,
                "inference_hz": 20,
                "buffer_seconds": 2.5,
                "min_frames": 4,
            },
        )
        clip = moving_positions(30)
        accepted = 0
        for index, row in enumerate(clip):
            session.push_positions(row, timestamp=clock.now)
            session.maybe_infer()
            session.poll()
            clock.advance(0.05)
            if speaker.spoken:
                accepted = len(speaker.spoken)
        self.assertEqual(speaker.spoken, ["Hello"])
        self.assertEqual(session.sentence.labels, ["hello"])

        clock.advance(2.0)
        still = np.zeros(POSITION_DIM, np.float32)
        for _ in range(8):
            session.push_positions(still, timestamp=clock.now)
            session.maybe_infer()
            session.poll()
            clock.advance(0.05)

        for index, row in enumerate(clip):
            session.push_positions(row, timestamp=clock.now)
            session.maybe_infer()
            session.poll()
            clock.advance(0.05)
        self.assertEqual(speaker.spoken, ["Hello"])
        self.assertEqual(session.sentence.labels, ["hello"])

    def test_tts_is_mocked(self) -> None:
        tts = TextToSpeech(api_key="test-key")
        pcm = np.zeros(1000, dtype=np.int16).tobytes()
        with patch.object(TextToSpeech, "_synthesize", return_value=pcm) as synth:
            with patch.object(TextToSpeech, "_play_pcm16"):
                self.assertTrue(tts.speak("Hello"))
        synth.assert_called_once()
        self.assertFalse(TextToSpeech(api_key=None).speak("Hello"))
        with patch.object(TextToSpeech, "speak", return_value=True) as mocked:
            with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "x"}, clear=False):
                self.assertTrue(speak("Hi"))
        mocked.assert_called()


class TrainingSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="signbridge_train_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_train_loads_sequences_and_writes_model(self) -> None:
        data = self.tmp / "data"
        for label, phase in (("hello", 0.0), ("yes", 1.5)):
            for index in range(4):
                features = features_from_position_sequence(
                    moving_positions(10 + index, phase=phase + index)
                )
                save_sequence(data / label / f"{index}.npz", features, label)

        result = train(
            data_dir=data,
            epochs=1,
            batch_size=4,
            seed=0,
            model_path=self.tmp / "asl_sequence_model.pt",
            labels_path=self.tmp / "asl_sequence_labels.json",
        )
        self.assertEqual(len(result["classes"]), 2)
        self.assertEqual(
            result["original_train_sequences"] + result["val_sequences"], 8
        )
        self.assertEqual(
            result["mirrored_train_sequences"], result["original_train_sequences"]
        )
        self.assertEqual(
            result["train_sequences"], 2 * result["original_train_sequences"]
        )
        self.assertTrue(Path(result["model_path"]).is_file())
        self.assertIn("best_val_accuracy", result)
        all_examples = load_all_sequences(data)
        self.assertEqual(len(all_examples), 8)

        from src.asl.sequence_model import load_sequence_model

        model, classes, meta = load_sequence_model(
            result["model_path"], result["labels_path"]
        )
        recognizer = LiveSequenceRecognizer(model, classes, threshold=0.0)
        pred = recognizer.predict(all_examples[0].features)
        self.assertIn(pred.class_id, range(len(classes)))
        self.assertEqual(tuple(softmax_probs(np.zeros(3)).shape), (3,))


if __name__ == "__main__":
    unittest.main()
