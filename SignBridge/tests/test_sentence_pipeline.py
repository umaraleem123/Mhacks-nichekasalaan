"""Sentence-level ASL capture, Gemini parsing, and TTS — no camera, no API."""

from __future__ import annotations

import io
import json
import os
import sys
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.asl.gemini_recognizer import (
    DEFAULT_MODEL,
    GeminiConfig,
    GeminiJsonError,
    GeminiRecognizerError,
    GeminiRequestError,
    GeminiSignRecognizer,
    SentenceTranslation,
    gemini_timeout_seconds,
    incomplete_reason,
    interaction_diagnostics,
    parse_gemini_json,
    preview_model_text,
    redact_secrets,
)
from src.asl.sequence_capture import (
    SENTENCE_HARD_MAX_SECONDS,
    SENTENCE_MAX_SECONDS,
    SENTENCE_SAMPLE_FPS,
    CaptureConfig,
    FrameSequence,
    SequenceCapture,
    SequenceCaptureError,
    deduplicate_sequence,
    sentence_max_seconds,
)
from src.speech.text_to_speech import TextToSpeech, clean_text


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def solid_frame(value: int, width: int = 60, height: int = 40) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def make_sequence(count: int = 5, step: float = 0.2) -> FrameSequence:
    capture = SequenceCapture(
        CaptureConfig.for_sentence(),
        clock=FakeClock(),
    )
    clock = capture._clock
    capture.start()
    for index in range(count):
        capture.add_frame(solid_frame(20 + index * 10))
        clock.advance(step)
    return capture.sequence()


class TextItem:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class ModelStep:
    def __init__(self, text: str) -> None:
        self.type = "model_output"
        self.content = [TextItem(text)]


class FakeError:
    def __init__(self, message: str, code: str = "failed") -> None:
        self.message = message
        self.code = code


class FakeUsage:
    def __init__(
        self,
        total_input_tokens: int | None = None,
        total_output_tokens: int | None = None,
        total_thought_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        self.total_input_tokens = total_input_tokens
        self.total_output_tokens = total_output_tokens
        self.total_thought_tokens = total_thought_tokens
        self.total_tokens = total_tokens


class FakeInteraction:
    def __init__(
        self,
        payload: dict | None = None,
        status: str = "completed",
        text: str | None = None,
        interaction_id: str | None = "ix-1",
        errors: list | None = None,
        usage: FakeUsage | None = None,
        continuation_token: str | None = None,
    ) -> None:
        self.status = status
        self.id = interaction_id
        if text is not None:
            self.steps = [ModelStep(text)] if text else []
        elif payload is not None:
            self.steps = [ModelStep(json.dumps(payload))]
        else:
            self.steps = []
        self.errors = errors or []
        self.usage = usage
        self.continuation_token = continuation_token


class FakeInteractions:
    def __init__(self, client: "FakeClient") -> None:
        self._client = client

    def create(self, **kwargs):
        self._client.calls.append(kwargs)
        if self._client.error is not None:
            raise self._client.error
        return self._client.next_interaction()

    def get(self, interaction_id: str, **kwargs):
        self._client.get_calls.append(interaction_id)
        if self._client.get_error is not None:
            raise self._client.get_error
        return self._client.next_interaction()


class FakeGenerateResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeModels:
    def __init__(self, client: "FakeClient") -> None:
        self._client = client

    def generate_content(self, **kwargs):
        self._client.calls.append(kwargs)
        if self._client.error is not None:
            raise self._client.error
        return FakeGenerateResponse(self._client.next_text())


class FakeClient:
    def __init__(
        self,
        payload: dict | None = None,
        error: Exception | None = None,
        queue: list | None = None,
        get_error: Exception | None = None,
    ) -> None:
        self.payload = payload or {
            "recognized": True,
            "english_translation": "Hello, how are you?",
            "confidence": 0.91,
            "notes": "",
        }
        self.error = error
        self.get_error = get_error
        self.queue = list(queue or [])
        self.calls: list[dict] = []
        self.get_calls: list[str] = []
        self.models = FakeModels(self)
        self.interactions = FakeInteractions(self)

    def next_text(self) -> str:
        if self.queue:
            item = self.queue.pop(0)
            if isinstance(item, str):
                return item
            if isinstance(item, FakeInteraction):
                if item.steps:
                    return "".join(
                        getattr(part, "text", "")
                        for step in item.steps
                        for part in (getattr(step, "content", None) or [])
                    )
                return ""
            if isinstance(item, dict):
                return json.dumps(item)
        return json.dumps(self.payload)

    def next_interaction(self) -> FakeInteraction:
        if self.queue:
            item = self.queue.pop(0)
            if isinstance(item, FakeInteraction):
                return item
        return FakeInteraction(self.payload)


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
    def __init__(self, succeed: bool = True, explode: bool = False) -> None:
        self.spoken: list[str] = []
        self.succeed = succeed
        self.explode = explode

    def speak(self, text: str) -> bool:
        if self.explode:
            raise RuntimeError("synthesizer exploded")
        self.spoken.append(text)
        return self.succeed


class BusyApiError(Exception):
    def __init__(self, code: int = 429) -> None:
        self.code = code
        self.status = "RESOURCE_EXHAUSTED"
        self.message = "quota exceeded for key AIzaSyTestKeyABCDEFG12345"


class TimeoutApiError(Exception):
    def __init__(self) -> None:
        self.code = 408
        self.status = "DEADLINE_EXCEEDED"
        self.message = "deadline exceeded"
        super().__init__("deadline exceeded")


class SentenceCaptureTests(unittest.TestCase):
    def test_space_starts_and_stops_recording(self) -> None:
        from src.asl.gemini_demo import DemoSession, Status

        clock = FakeClock()
        session = DemoSession(
            GeminiSignRecognizer(
                client=FakeClient(),
                config=GeminiConfig(send_as_video=False),
            ),
            CaptureConfig.for_sentence(),
            speaker=StubSpeaker(),
            executor=ImmediateExecutor(),
            clock=clock,
        )
        self.assertEqual(session.status, Status.READY)
        self.assertTrue(session.handle_space())
        self.assertEqual(session.status, Status.CAPTURING)

        for value in (10, 40, 80, 120, 160):
            session.offer_frame(solid_frame(value))
            clock.advance(0.25)

        self.assertGreaterEqual(session.frames_captured, 3)
        self.assertTrue(session.handle_space())
        session.poll()
        self.assertNotEqual(session.status, Status.CAPTURING)

    def test_multiple_frames_preserve_order(self) -> None:
        clock = FakeClock()
        capture = SequenceCapture(CaptureConfig.for_sentence(), clock=clock)
        capture.start()
        values = [11, 33, 55, 77, 99]
        for value in values:
            capture.add_frame(solid_frame(value))
            clock.advance(0.4)

        sequence = capture.stop()
        self.assertGreaterEqual(len(sequence), 4)
        stored = [int(frame.image[0, 0, 0]) for frame in sequence]
        self.assertEqual(stored, sorted(stored))
        self.assertEqual(stored, values[: len(stored)])
        self.assertEqual(sequence.timestamps, sorted(sequence.timestamps))

    def test_sentence_config_bounds(self) -> None:
        config = CaptureConfig.for_sentence()
        self.assertEqual(config.duration_seconds, SENTENCE_MAX_SECONDS)
        self.assertEqual(config.sample_fps, SENTENCE_SAMPLE_FPS)
        expected = int(round(SENTENCE_MAX_SECONDS * SENTENCE_SAMPLE_FPS))
        self.assertEqual(expected, 18)
        self.assertGreaterEqual(config.max_frames, expected)
        self.assertEqual(config.max_frame_width, 512)
        with self.assertRaises(SequenceCaptureError):
            CaptureConfig.for_sentence(max_seconds=SENTENCE_HARD_MAX_SECONDS + 1)

    def test_recording_duration_from_env(self) -> None:
        with patch.dict(os.environ, {"SIGNBRIDGE_MAX_RECORDING_SECONDS": "8"}):
            self.assertEqual(sentence_max_seconds(), 8.0)
            config = CaptureConfig.for_sentence()
            self.assertEqual(config.duration_seconds, 8.0)

    def test_frame_sampling_not_every_camera_frame(self) -> None:
        clock = FakeClock()
        capture = SequenceCapture(CaptureConfig.for_sentence(), clock=clock)
        capture.start()
        camera_fps = 30
        for index in range(int(6 * camera_fps)):
            capture.add_frame(solid_frame(index % 200))
            clock.advance(1 / camera_fps)
        sampled = capture.sequence()
        self.assertGreater(capture._offered, 100)
        self.assertLessEqual(len(sampled), 20)
        self.assertGreaterEqual(len(sampled), 12)
        self.assertEqual(sampled.timestamps, sorted(sampled.timestamps))

    def test_downscale_preserves_aspect(self) -> None:
        clock = FakeClock()
        capture = SequenceCapture(CaptureConfig.for_sentence(), clock=clock)
        capture.start()
        wide = np.full((360, 1280, 3), 80, dtype=np.uint8)
        capture.add_frame(wide)
        stored = capture.sequence().frames[0].image
        self.assertEqual(stored.shape[1], 512)
        self.assertEqual(stored.shape[0], 144)

    def test_duplicate_frames_are_removed(self) -> None:
        clock = FakeClock()
        capture = SequenceCapture(CaptureConfig.for_sentence(), clock=clock)
        capture.start()
        for _ in range(6):
            capture.add_frame(solid_frame(90))
            clock.advance(0.4)
        sampled = capture.sequence()
        self.assertGreater(len(sampled), 2)
        deduped = deduplicate_sequence(sampled)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped.timestamps, sorted(deduped.timestamps))


class GeminiSentenceTests(unittest.TestCase):
    def test_one_request_for_complete_sentence_not_per_frame(self) -> None:
        from src.asl.gemini_demo import DemoSession, Status

        client = FakeClient()
        clock = FakeClock()
        session = DemoSession(
            GeminiSignRecognizer(
                client=client,
                config=GeminiConfig(send_as_video=False),
            ),
            CaptureConfig.for_sentence(),
            speaker=StubSpeaker(),
            executor=ImmediateExecutor(),
            clock=clock,
        )
        session.handle_space()
        for value in range(8):
            session.offer_frame(solid_frame(20 + value))
            clock.advance(0.25)
            self.assertEqual(len(client.calls), 0)

        self.assertEqual(session.status, Status.CAPTURING)
        session.handle_space()
        session.poll()
        self.assertEqual(len(client.calls), 1)
        self.assertIn(session.status, {Status.SPEAKING, Status.RESULT, Status.READY})

    def test_parse_successful_sentence(self) -> None:
        client = FakeClient(
            {
                "recognized": True,
                "english_translation": "Hello, how are you?",
                "confidence": 0.91,
                "notes": "",
            }
        )
        recognizer = GeminiSignRecognizer(
            client=client, config=GeminiConfig(send_as_video=False)
        )
        result = recognizer.translate_sequence(make_sequence())
        self.assertTrue(result.recognized)
        self.assertEqual(result.english_translation, "Hello, how are you?")
        self.assertAlmostEqual(result.confidence, 0.91)
        self.assertEqual(result.speakable_text, "Hello, how are you?")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["model"], DEFAULT_MODEL)

    def test_ambiguous_response_is_not_forced(self) -> None:
        client = FakeClient(
            {
                "recognized": False,
                "english_translation": "",
                "confidence": 0.35,
                "notes": "The signing sequence was ambiguous.",
            }
        )
        result = GeminiSignRecognizer(
            client=client, config=GeminiConfig(send_as_video=False)
        ).translate_sequence(make_sequence())
        self.assertTrue(result.is_unknown)
        self.assertEqual(result.english_translation, "")
        self.assertEqual(result.speakable_text, "")
        self.assertIn("ambiguous", result.notes)

    def test_recognized_without_text_is_unknown(self) -> None:
        result = GeminiSignRecognizer(
            client=FakeClient(
                {
                    "recognized": True,
                    "english_translation": "   ",
                    "confidence": 0.9,
                    "notes": "",
                }
            ),
            config=GeminiConfig(send_as_video=False),
        ).translate_sequence(make_sequence())
        self.assertTrue(result.is_unknown)

    def test_busy_errors_do_not_retry(self) -> None:
        client = FakeClient(error=BusyApiError(429))
        recognizer = GeminiSignRecognizer(
            client=client, config=GeminiConfig(send_as_video=False)
        )
        with self.assertRaises(GeminiRequestError) as raised:
            recognizer.translate_sequence(make_sequence())
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(raised.exception.is_busy)
        self.assertEqual(
            raised.exception.user_message,
            GeminiRequestError.BUSY_MESSAGE,
        )

        client_503 = FakeClient(error=BusyApiError(503))
        with self.assertRaises(GeminiRequestError) as raised_503:
            GeminiSignRecognizer(
                client=client_503, config=GeminiConfig(send_as_video=False)
            ).translate_sequence(make_sequence())
        self.assertEqual(len(client_503.calls), 1)
        self.assertEqual(
            raised_503.exception.user_message,
            GeminiRequestError.BUSY_MESSAGE,
        )

    def test_api_keys_never_appear_in_errors(self) -> None:
        secret = "supersecretkeyvalue99"
        leaked = f"Authorization: Bearer {secret} and AIzaSyLeakedTokenABCDEFG"
        error = GeminiRequestError.from_exception(
            Exception(leaked), extra_secrets=(secret,)
        )
        blob = "\n".join(
            [error.diagnostics(), error.user_message, error.sanitized_message]
        )
        self.assertNotIn(secret, blob)
        self.assertNotIn("AIzaSyLeakedTokenABCDEFG", blob)
        self.assertIn("***REDACTED***", blob)

    def test_redact_env_key(self) -> None:
        key = "ENVONLYSECRETKEY88"
        with patch.dict(os.environ, {"GEMINI_API_KEY": key}):
            cleaned = redact_secrets(f"failed with {key}")
        self.assertNotIn(key, cleaned)

    def test_single_frame_is_rejected(self) -> None:
        recognizer = GeminiSignRecognizer(client=FakeClient())
        with self.assertRaises(GeminiRecognizerError):
            recognizer.translate_sequence(solid_frame(1))  # type: ignore[arg-type]
        too_short = FrameSequence([])
        result = recognizer.translate_sequence(too_short)
        self.assertTrue(result.is_unknown)


class SpeechAndUiTests(unittest.TestCase):
    def test_empty_translation_does_not_speak(self) -> None:
        from src.asl.gemini_demo import DemoSession, Status

        speaker = StubSpeaker()
        client = FakeClient(
            {
                "recognized": False,
                "english_translation": "",
                "confidence": 0.2,
                "notes": "Too unclear.",
            }
        )
        clock = FakeClock()
        session = DemoSession(
            GeminiSignRecognizer(
                client=client, config=GeminiConfig(send_as_video=False)
            ),
            CaptureConfig.for_sentence(),
            speaker=speaker,
            executor=ImmediateExecutor(),
            clock=clock,
        )
        session.handle_space()
        for value in range(4):
            session.offer_frame(solid_frame(50 + value))
            clock.advance(0.25)
        session.handle_space()
        session.poll()
        self.assertEqual(speaker.spoken, [])
        self.assertEqual(session.status, Status.RESULT)
        self.assertTrue(session.result.is_unknown)

    def test_successful_translation_speaks_once(self) -> None:
        from src.asl.gemini_demo import DemoSession, Status

        speaker = StubSpeaker()
        clock = FakeClock()
        session = DemoSession(
            GeminiSignRecognizer(
                client=FakeClient(),
                config=GeminiConfig(send_as_video=False),
            ),
            CaptureConfig.for_sentence(),
            speaker=speaker,
            executor=ImmediateExecutor(),
            clock=clock,
        )
        session.handle_space()
        for value in range(4):
            session.offer_frame(solid_frame(50 + value))
            clock.advance(0.25)
        session.handle_space()
        session.poll()
        self.assertEqual(speaker.spoken, ["Hello, how are you?"])
        session.poll()
        self.assertEqual(speaker.spoken, ["Hello, how are you?"])
        self.assertEqual(session.status, Status.READY)

    def test_tts_failure_does_not_crash(self) -> None:
        from src.asl.gemini_demo import DemoSession, Status

        speaker = StubSpeaker(explode=True)
        clock = FakeClock()
        session = DemoSession(
            GeminiSignRecognizer(
                client=FakeClient(),
                config=GeminiConfig(send_as_video=False),
            ),
            CaptureConfig.for_sentence(),
            speaker=speaker,
            executor=ImmediateExecutor(),
            clock=clock,
        )
        session.handle_space()
        for value in range(4):
            session.offer_frame(solid_frame(50 + value))
            clock.advance(0.25)
        session.handle_space()
        session.poll()
        self.assertEqual(session.status, Status.RESULT)
        self.assertEqual(session.result.english_translation, "Hello, how are you?")

    def test_clean_text_rejects_empty(self) -> None:
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text("   "), "")
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text("Hello,  how\nare you?"), "Hello, how are you?")

    def test_speak_skips_empty_without_backend(self) -> None:
        tts = TextToSpeech()
        self.assertFalse(tts.speak(""))
        self.assertFalse(tts.speak("   "))
        self.assertEqual(tts.last_error, "Nothing to speak.")

    def test_model_is_configurable_not_scattered(self) -> None:
        self.assertEqual(DEFAULT_MODEL, "gemini-3.7-flash")
        self.assertEqual(GeminiConfig().model, DEFAULT_MODEL)
        self.assertEqual(GeminiConfig(model="other-model").model, "other-model")

    def test_busy_ui_message(self) -> None:
        from src.asl.gemini_demo import DemoSession, Status

        clock = FakeClock()
        session = DemoSession(
            GeminiSignRecognizer(
                client=FakeClient(error=BusyApiError(429)),
                config=GeminiConfig(send_as_video=False),
            ),
            CaptureConfig.for_sentence(),
            speaker=StubSpeaker(),
            executor=ImmediateExecutor(),
            clock=clock,
        )
        session.handle_space()
        for value in range(4):
            session.offer_frame(solid_frame(10 + value))
            clock.advance(0.25)
        stderr = io.StringIO()
        with patch.object(sys, "stderr", stderr):
            session.handle_space()
            session.poll()
        self.assertEqual(session.status, Status.ERROR)
        self.assertEqual(session.error, GeminiRequestError.BUSY_MESSAGE)
        self.assertNotIn("AIzaSyTestKeyABCDEFG12345", stderr.getvalue())
        self.assertNotIn("AIzaSyTestKeyABCDEFG12345", session.error)


SUCCESS_PAYLOAD = {
    "recognized": True,
    "english_translation": "Hello, how are you?",
    "confidence": 0.91,
    "notes": "",
}


class GenerateContentTests(unittest.TestCase):
    def _recognizer(self, client, **config_kw):
        defaults = dict(send_as_video=False, request_timeout_seconds=22.0)
        defaults.update(config_kw)
        return GeminiSignRecognizer(
            client=client, config=GeminiConfig(**defaults)
        )

    def test_successful_gemini_response(self) -> None:
        client = FakeClient(SUCCESS_PAYLOAD)
        result = self._recognizer(client).translate_sequence(make_sequence())
        self.assertTrue(result.recognized)
        self.assertEqual(result.english_translation, "Hello, how are you?")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["model"], DEFAULT_MODEL)
        self.assertGreater(len(client.calls[0]["contents"]), 2)

    def test_gemini_timeout(self) -> None:
        client = FakeClient(error=TimeoutApiError())
        with self.assertRaises(GeminiRequestError) as raised:
            self._recognizer(client).translate_sequence(make_sequence())
        self.assertTrue(raised.exception.is_timeout)
        self.assertEqual(raised.exception.user_message, GeminiRequestError.TIMEOUT_MESSAGE)
        self.assertEqual(len(client.calls), 1)

    def test_timeout_returns_ui_to_ready(self) -> None:
        from src.asl.gemini_demo import DemoSession, Status

        clock = FakeClock()
        session = DemoSession(
            GeminiSignRecognizer(
                client=FakeClient(error=TimeoutApiError()),
                config=GeminiConfig(send_as_video=False),
            ),
            CaptureConfig.for_sentence(),
            speaker=StubSpeaker(),
            executor=ImmediateExecutor(),
            clock=clock,
        )
        session.handle_space()
        for value in range(4):
            session.offer_frame(solid_frame(10 + value))
            clock.advance(0.4)
        session.handle_space()
        session.poll()
        self.assertEqual(session.status, Status.READY)
        self.assertEqual(session.error, GeminiRequestError.TIMEOUT_MESSAGE)

    def test_empty_gemini_response(self) -> None:
        client = FakeClient(queue=[""])
        result = self._recognizer(client).translate_sequence(make_sequence())
        self.assertTrue(result.is_unknown)
        self.assertEqual(len(client.calls), 1)

    def test_malformed_response(self) -> None:
        client = FakeClient(queue=["not-json {"])
        result = self._recognizer(client).translate_sequence(make_sequence())
        self.assertTrue(result.is_unknown)
        self.assertIn("JSON", result.notes)
        self.assertEqual(len(client.calls), 1)

    def test_timeout_env(self) -> None:
        with patch.dict(os.environ, {"SIGNBRIDGE_GEMINI_TIMEOUT_SECONDS": "25"}):
            self.assertEqual(gemini_timeout_seconds(), 25.0)

    def test_diagnostics_redact_secrets(self) -> None:
        secret = "supersecretkeyvalue99"
        interaction = FakeInteraction(
            status="failed",
            text="",
            errors=[FakeError(f"Authorization: Bearer {secret}")],
        )
        blob = interaction_diagnostics(interaction, extra_secrets=(secret,))
        self.assertNotIn(secret, blob)
        self.assertIn("***REDACTED***", blob)
        reason = incomplete_reason(
            FakeInteraction(
                status="incomplete",
                text="",
                usage=FakeUsage(total_output_tokens=1024, total_thought_tokens=0),
            ),
            1024,
        )
        self.assertIn("max_output_tokens=1024", reason)


class ParseGeminiJsonTests(unittest.TestCase):
    def test_valid_object(self) -> None:
        payload = parse_gemini_json(
            '{"recognized": true, "english_translation": "Hello", '
            '"confidence": 0.93, "notes": ""}'
        )
        self.assertEqual(payload["english_translation"], "Hello")
        self.assertAlmostEqual(payload["confidence"], 0.93)
        self.assertTrue(payload["recognized"])
        self.assertEqual(payload["notes"], "")

    def test_markdown_fences(self) -> None:
        payload = parse_gemini_json(
            "```json\n"
            '{"recognized": true, "english_translation": "Hello", '
            '"confidence": 0.93, "notes": ""}\n'
            "```"
        )
        self.assertEqual(payload["english_translation"], "Hello")
        self.assertAlmostEqual(payload["confidence"], 0.93)

    def test_surrounding_prose(self) -> None:
        payload = parse_gemini_json(
            "Here is the result:\n"
            '{"recognized": true, "english_translation": "Hello", '
            '"confidence": 0.93, "notes": ""}\n'
            "Thanks."
        )
        self.assertEqual(payload["english_translation"], "Hello")

    def test_malformed_confidence_leading_dot(self) -> None:
        payload = parse_gemini_json(
            "{\n"
            '  "recognized": true,\n'
            '  "english_translation": "Hello, how are you?",\n'
            '  "confidence": .0.93,\n'
            '  "notes": ""\n'
            "}"
        )
        self.assertAlmostEqual(payload["confidence"], 0.93)
        self.assertEqual(payload["english_translation"], "Hello, how are you?")

    def test_malformed_confidence_double_decimal(self) -> None:
        payload = parse_gemini_json(
            '{"recognized": true, "english_translation": "Hello", '
            '"confidence": 0.93.0, "notes": ""}'
        )
        self.assertAlmostEqual(payload["confidence"], 0.93)

    def test_truncated_json_is_unknown_not_crash(self) -> None:
        with self.assertRaises(GeminiJsonError):
            parse_gemini_json('{"recognized": true, "english_translation": "Hello"')

        client = FakeClient(
            queue=[
                FakeInteraction(
                    status="completed",
                    text=(
                        "{\n"
                        '  "recognized": true,\n'
                        '  "english_translation": "Hello, how are you?",\n'
                        '  "confidence": .0.93'
                    ),
                )
            ]
        )
        # Completely truncated (no closing brace AND unrepairable leftover)
        # still must not raise out of translate_sequence.
        result = GeminiSignRecognizer(
            client=client,
            config=GeminiConfig(send_as_video=False),
        ).translate_sequence(make_sequence())
        # The leading-dot confidence is repairable even without a closing brace
        # if the extractor returns the remainder; if not, we get unknown.
        self.assertIsInstance(result, SentenceTranslation)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.get_calls, [])

    def test_parse_failure_does_not_retry(self) -> None:
        client = FakeClient(
            queue=[FakeInteraction(status="completed", text="definitely not json")]
        )
        result = GeminiSignRecognizer(
            client=client,
            config=GeminiConfig(send_as_video=False),
        ).translate_sequence(make_sequence())
        self.assertTrue(result.is_unknown)
        self.assertIn("JSON", result.notes)
        self.assertEqual(len(client.calls), 1)
        self.assertIsNotNone(result.raw)
        self.assertIn("raw_text", result.raw or {})

    def test_preview_redacts_keys(self) -> None:
        secret = "supersecretkeyvalue99"
        preview = preview_model_text(
            f'{{"notes": "key={secret}"}}', extra_secrets=(secret,)
        )
        self.assertNotIn(secret, preview)
        self.assertIn("***REDACTED***", preview)

    def test_missing_optional_notes_default(self) -> None:
        payload = parse_gemini_json(
            '{"recognized": true, "english_translation": "Hello", "confidence": 0.5}'
        )
        self.assertEqual(payload["notes"], "")
        self.assertAlmostEqual(payload["confidence"], 0.5)

    def test_confidence_clamped(self) -> None:
        payload = parse_gemini_json(
            '{"recognized": false, "english_translation": "", '
            '"confidence": 4.2, "notes": ""}'
        )
        self.assertAlmostEqual(payload["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
