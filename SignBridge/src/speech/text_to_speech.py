"""Speak English text aloud through ElevenLabs.

Credentials come from the environment:

    ELEVENLABS_API_KEY
    ELEVENLABS_VOICE_ID

Neither value is logged, printed, or included in error messages. Playback
uses a WAV-sized PCM stream from the ElevenLabs HTTP API and a portable
audio device library, so the same code runs on macOS and Windows.

Callers that need a live UI should run `speak` on a worker thread; it
blocks until the utterance finishes.
"""

from __future__ import annotations

import io
import json
import os
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAX_SPOKEN_CHARACTERS = 500
API_KEY_ENV = "ELEVENLABS_API_KEY"
VOICE_ID_ENV = "ELEVENLABS_VOICE_ID"
MODEL_ENV = "ELEVENLABS_MODEL_ID"
DEFAULT_MODEL_ID = "eleven_turbo_v2_5"
DEFAULT_VOICE_ID = "rWZM1pGWKmpGt3Hvergd"
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_TIMEOUT_SECONDS = 20.0
TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"


class TextToSpeechError(RuntimeError):
    """Speech could not be produced."""


@dataclass(frozen=True)
class SpeechConfig:
    """Voice tunables. None means the ElevenLabs default."""

    model_id: str = DEFAULT_MODEL_ID
    sample_rate: int = DEFAULT_SAMPLE_RATE
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


def clean_text(text: object) -> str:
    """Normalize text for speaking, or return "" if there is nothing to say."""
    if not isinstance(text, str):
        return ""
    collapsed = " ".join(text.split())
    printable = "".join(ch for ch in collapsed if ch.isprintable())
    return printable[:MAX_SPOKEN_CHARACTERS].strip()


def load_local_env(root: Path | None = None) -> None:
    """Load SignBridge/.env into os.environ, overwriting matching keys.

    `.env` is gitignored. This helper never prints values.
    """
    env_path = (root or Path(__file__).resolve().parents[2]) / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ[key] = value


class TextToSpeech:
    """Speaks short English sentences through ElevenLabs.

        tts = TextToSpeech.from_env()
        tts.speak("Hello")

    `speak` returns False rather than raising when there is nothing to say
    or the request fails, so a failed utterance cannot take the app down.
    """

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str | None = None,
        config: SpeechConfig | None = None,
    ) -> None:
        self.api_key = (api_key or "").strip() or None
        self.voice_id = (voice_id or "").strip() or DEFAULT_VOICE_ID
        self.config = config or SpeechConfig()
        self.last_error: str | None = None
        self.last_backend: str | None = None

    @classmethod
    def from_env(cls, config: SpeechConfig | None = None) -> TextToSpeech:
        load_local_env()
        model_id = os.environ.get(MODEL_ENV, "").strip() or DEFAULT_MODEL_ID
        speech_config = config or SpeechConfig(model_id=model_id)
        return cls(
            api_key=os.environ.get(API_KEY_ENV),
            voice_id=os.environ.get(VOICE_ID_ENV) or DEFAULT_VOICE_ID,
            config=speech_config,
        )

    def is_available(self) -> bool:
        """Whether an API key is configured. Does not call the network."""
        return bool(self.api_key)

    def speak(self, text: str) -> bool:
        """Say `text` aloud. Returns True if audio was fetched and played."""
        spoken = clean_text(text)
        self.last_error = None
        self.last_backend = None

        if not spoken:
            self.last_error = "Nothing to speak."
            return False

        if not self.api_key:
            self.last_error = f"{API_KEY_ENV} is not set."
            return False

        wav = self.synthesize_wav(spoken)
        if wav is None:
            return False
        try:
            self._play_pcm16(self._wav_pcm(wav), self.config.sample_rate)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False

        self.last_backend = "elevenlabs"
        return True

    def synthesize_wav(self, text: str) -> bytes | None:
        """Return a WAV file for `text`, or None. Does not play audio."""
        spoken = clean_text(text)
        self.last_error = None
        self.last_backend = None
        if not spoken:
            self.last_error = "Nothing to speak."
            return None
        if not self.api_key:
            self.last_error = f"{API_KEY_ENV} is not set."
            return None
        try:
            pcm = self._synthesize(spoken)
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(self.config.sample_rate)
                wav.writeframes(pcm)
            self.last_backend = "elevenlabs"
            return buffer.getvalue()
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    @staticmethod
    def _wav_pcm(wav_bytes: bytes) -> bytes:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            return wav.readframes(wav.getnframes())

    def _synthesize(self, text: str) -> bytes:
        """POST text to ElevenLabs and return 16-bit mono PCM."""
        url = (
            TTS_URL.format(voice_id=self.voice_id)
            + f"?output_format=pcm_{self.config.sample_rate}"
        )
        payload = json.dumps(
            {"text": text, "model_id": self.config.model_id}
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={
                "xi-api-key": self.api_key or "",
                "Content-Type": "application/json",
                "Accept": "application/octet-stream",
            },
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as response:
                audio = response.read()
        except urllib.error.HTTPError as exc:
            raise TextToSpeechError(
                f"ElevenLabs HTTP {exc.code}"
            ) from exc
        except urllib.error.URLError as exc:
            raise TextToSpeechError("ElevenLabs request failed") from exc

        if not audio:
            raise TextToSpeechError("ElevenLabs returned empty audio.")
        return audio

    def _play_pcm16(self, pcm: bytes, sample_rate: int) -> None:
        """Play signed 16-bit little-endian mono PCM through the default device."""
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise TextToSpeechError(
                "sounddevice is required to play ElevenLabs audio. "
                "Install dependencies with python -m pip install -r requirements.txt."
            ) from exc

        samples = np.frombuffer(pcm, dtype="<i2")
        if samples.size == 0:
            raise TextToSpeechError("ElevenLabs returned empty PCM.")
        audio = samples.astype(np.float32) / 32768.0
        sd.play(audio, samplerate=sample_rate, blocking=True)


def speak(text: str) -> bool:
    """Module-level helper used by the demo and tests."""
    return TextToSpeech.from_env().speak(text)
