"""Mocked ElevenLabs diagnostic tests — no network, no audio device."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.speech.test_elevenlabs_audio import (
    DiagnosticError,
    categorize_sdk_error,
    generate_audio,
    materialize_audio,
    play_wav_file,
    require_credentials,
    run_diagnostic,
    sniff_format,
    write_audio_file,
)


class FakeApiError(Exception):
    def __init__(self, status_code: int, body: object) -> None:
        super().__init__("api")
        self.status_code = status_code
        self.body = body
        self.headers: dict[str, str] = {}


def _silent_wav_bytes(frames: int = 2205, rate: int = 22050) -> bytes:
    buffer = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    path = Path(buffer.name)
    buffer.close()
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(np.zeros(frames, dtype=np.int16).tobytes())
    data = path.read_bytes()
    path.unlink(missing_ok=True)
    return data


class CredentialTests(unittest.TestCase):
    def test_missing_api_key(self) -> None:
        with patch("src.speech.test_elevenlabs_audio.load_local_env"):
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(DiagnosticError) as caught:
                    require_credentials()
        self.assertEqual(caught.exception.category, "API KEY ERROR")
        self.assertNotIn("sk_", str(caught.exception))

    def test_placeholder_key_is_treated_as_missing(self) -> None:
        env = {"ELEVENLABS_API_KEY": "your-key-here", "ELEVENLABS_VOICE_ID": "abc"}
        with patch("src.speech.test_elevenlabs_audio.load_local_env"):
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(DiagnosticError) as caught:
                    require_credentials()
        self.assertEqual(caught.exception.category, "API KEY ERROR")

    def test_missing_voice_id(self) -> None:
        env = {"ELEVENLABS_API_KEY": "sk_test_not_a_real_key"}
        with patch("src.speech.test_elevenlabs_audio.load_local_env"):
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(DiagnosticError) as caught:
                    require_credentials()
        self.assertEqual(caught.exception.category, "VOICE ID ERROR")
        self.assertNotIn("sk_test_not_a_real_key", str(caught.exception.detail))

    def test_credentials_ok_when_both_set(self) -> None:
        env = {
            "ELEVENLABS_API_KEY": "sk_test_not_a_real_key",
            "ELEVENLABS_VOICE_ID": "voice123",
        }
        with patch("src.speech.test_elevenlabs_audio.load_local_env"):
            with patch.dict(os.environ, env, clear=True):
                key, voice = require_credentials()
        self.assertEqual(key, "sk_test_not_a_real_key")
        self.assertEqual(voice, "voice123")


class AudioMaterializeTests(unittest.TestCase):
    def test_bytes_passthrough(self) -> None:
        data, name = materialize_audio(b"RIFF....WAVE")
        self.assertEqual(data, b"RIFF....WAVE")
        self.assertEqual(name, "bytes")

    def test_iterator_is_consumed(self) -> None:
        def chunks():
            yield b"RIFF"
            yield b"xxxx"
            yield b"WAVE"

        data, name = materialize_audio(chunks())
        self.assertEqual(data, b"RIFFxxxxWAVE")
        self.assertIn(name.lower(), {"generator", "generator"})

    def test_empty_bytes_are_generation_error(self) -> None:
        with self.assertRaises(DiagnosticError) as caught:
            materialize_audio(b"")
        self.assertEqual(caught.exception.category, "AUDIO GENERATION ERROR")

    def test_stream_api_error_is_not_generation_error(self) -> None:
        def chunks():
            yield b"RIFF"
            raise FakeApiError(
                402,
                {
                    "detail": {
                        "type": "payment_required",
                        "code": "paid_plan_required",
                        "message": (
                            "Free users cannot use library voices via the API. "
                            "Please upgrade your subscription to use this voice."
                        ),
                    }
                },
            )

        with self.assertRaises(DiagnosticError) as caught:
            materialize_audio(chunks(), api_key="sk_test_not_a_real_key")
        self.assertEqual(caught.exception.category, "API ERROR")
        self.assertIn("HTTP 402", caught.exception.detail)
        self.assertIn("library voices", caught.exception.detail)
        self.assertNotIn("sk_test_not_a_real_key", caught.exception.detail)

    def test_sniff_wav_and_pcm(self) -> None:
        wav = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 8
        self.assertEqual(sniff_format(wav), "wav")
        self.assertEqual(sniff_format(b"\x00\x01\x02"), "pcm")

    def test_write_wav_file(self) -> None:
        payload = _silent_wav_bytes()
        path = write_audio_file(payload, "wav")
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 44)
        self.assertEqual(path.suffix, ".wav")


class DiagnosticRunTests(unittest.TestCase):
    def test_full_success_path_mocks_sdk_and_playback(self) -> None:
        wav = _silent_wav_bytes()

        def fake_convert(**_kwargs):
            yield wav[:20]
            yield wav[20:]

        client = MagicMock()
        client.text_to_speech.convert.side_effect = lambda **kwargs: fake_convert()
        env = {
            "ELEVENLABS_API_KEY": "sk_test_not_a_real_key",
            "ELEVENLABS_VOICE_ID": "voice123",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("src.speech.test_elevenlabs_audio.make_client", return_value=client):
                with patch("src.speech.test_elevenlabs_audio.play_wav_file") as play:
                    code = run_diagnostic(play=True)
        self.assertEqual(code, 0)
        play.assert_called_once()
        client.text_to_speech.convert.assert_called_once()
        kwargs = client.text_to_speech.convert.call_args.kwargs
        self.assertEqual(kwargs["text"], "Hello from SignBridge.")
        self.assertEqual(kwargs["voice_id"], "voice123")
        self.assertEqual(kwargs["output_format"], "wav_22050")

    def test_api_failure_is_categorized_and_key_is_hidden(self) -> None:
        client = MagicMock()
        client.text_to_speech.convert.side_effect = RuntimeError(
            "401 unauthorized for sk_test_not_a_real_key"
        )
        with self.assertRaises(DiagnosticError) as caught:
            generate_audio(client, "voice123", "sk_test_not_a_real_key")
        self.assertEqual(caught.exception.category, "API KEY ERROR")
        self.assertNotIn("sk_test_not_a_real_key", caught.exception.detail)

    def test_payment_required_is_api_error(self) -> None:
        err = FakeApiError(
            402,
            {"detail": {"message": "Free users cannot use library voices via the API."}},
        )
        categorized = categorize_sdk_error(err, "sk_test_not_a_real_key")
        self.assertEqual(categorized.category, "API ERROR")
        self.assertIn("HTTP 402", categorized.detail)

    def test_playback_error_category(self) -> None:
        with self.assertRaises(DiagnosticError) as caught:
            play_wav_file(Path("/no/such/file.wav"))
        self.assertEqual(caught.exception.category, "PLAYBACK ERROR")

    def test_run_without_key_prints_category(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.speech.test_elevenlabs_audio.load_local_env"):
                code = run_diagnostic(play=False)
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
