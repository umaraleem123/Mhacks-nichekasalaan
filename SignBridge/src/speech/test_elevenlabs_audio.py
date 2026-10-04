"""Standalone ElevenLabs generate-and-play diagnostic.

    python -m src.speech.test_elevenlabs_audio

Does not touch ASL recognition. Credentials are never printed.
"""

from __future__ import annotations

import importlib.metadata
import os
import sys
import tempfile
import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from src.speech.text_to_speech import VOICE_ID_ENV, load_local_env

TEST_PHRASE = "Hello from SignBridge."
OUTPUT_FORMAT = "wav_22050"
SAMPLE_RATE = 22050
MODEL_ID = "eleven_turbo_v2_5"
API_KEY_ENV = "ELEVENLABS_API_KEY"


class DiagnosticError(RuntimeError):
    def __init__(self, category: str, detail: str = "") -> None:
        self.category = category
        self.detail = detail
        super().__init__(detail or category)


def sdk_version() -> str:
    try:
        return importlib.metadata.version("elevenlabs")
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _redact(text: str, secret: str | None) -> str:
    if not secret:
        return text
    return text.replace(secret, "[REDACTED]")


def format_sdk_error(exc: BaseException, api_key: str | None) -> str:
    """Human-readable SDK/API failure. Never includes the API key."""
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    message = ""
    if isinstance(body, dict):
        detail = body.get("detail", body)
        if isinstance(detail, dict):
            message = str(detail.get("message") or detail)
        else:
            message = str(detail)
    elif body is not None:
        message = str(body)
    else:
        message = str(exc)
    prefix = f"HTTP {status}: " if status is not None else f"{type(exc).__name__}: "
    return _redact(prefix + message, api_key)


def categorize_sdk_error(
    exc: BaseException,
    api_key: str | None,
    *,
    default_category: str = "API ERROR",
) -> DiagnosticError:
    """Map an ElevenLabs SDK exception to a diagnostic category."""
    message = format_sdk_error(exc, api_key)
    lowered = message.lower()
    status = getattr(exc, "status_code", None)
    if status == 401 or "unauthorized" in lowered or "invalid api" in lowered:
        return DiagnosticError("API KEY ERROR", message)
    if status == 404 or ("voice" in lowered and "not found" in lowered):
        return DiagnosticError("VOICE ID ERROR", message)
    if status is not None or type(exc).__name__ in {"ApiError", "UnprocessableEntityError"}:
        return DiagnosticError("API ERROR", message)
    return DiagnosticError(default_category, message)


def require_credentials() -> tuple[str, str]:
    load_local_env()
    api_key = os.environ.get(API_KEY_ENV, "").strip()
    if not api_key or api_key in {"your-key-here", "your-key"}:
        raise DiagnosticError("API KEY ERROR", f"{API_KEY_ENV} is not set.")
    voice_id = os.environ.get(VOICE_ID_ENV, "").strip()
    if not voice_id or voice_id in {"your-voice-id"}:
        raise DiagnosticError("VOICE ID ERROR", f"{VOICE_ID_ENV} is not set.")
    return api_key, voice_id


def make_client(api_key: str):
    try:
        from elevenlabs.client import ElevenLabs
    except Exception as exc:
        raise DiagnosticError(
            "API ERROR",
            f"Could not import ElevenLabs SDK ({sdk_version()}): {type(exc).__name__}",
        ) from exc
    try:
        return ElevenLabs(api_key=api_key)
    except Exception as exc:
        raise DiagnosticError(
            "API ERROR",
            _redact(f"{type(exc).__name__}: {exc}", api_key),
        ) from exc


def materialize_audio(audio: object, api_key: str | None = None) -> tuple[bytes, str]:
    """Turn whatever the SDK returned into raw bytes. Never assume the type."""
    type_name = type(audio).__name__
    if audio is None:
        raise DiagnosticError("AUDIO GENERATION ERROR", "SDK returned None.")
    if isinstance(audio, (bytes, bytearray, memoryview)):
        data = bytes(audio)
        if not data:
            raise DiagnosticError("AUDIO GENERATION ERROR", "SDK returned empty bytes.")
        return data, type_name
    if isinstance(audio, str):
        raise DiagnosticError(
            "AUDIO GENERATION ERROR",
            "SDK returned a string instead of audio bytes.",
        )
    if isinstance(audio, Iterator) or hasattr(audio, "__iter__"):
        chunks: list[bytes] = []
        try:
            for chunk in audio:  # type: ignore[union-attr]
                if chunk is None:
                    continue
                if isinstance(chunk, (bytes, bytearray, memoryview)):
                    chunks.append(bytes(chunk))
                    continue
                raise DiagnosticError(
                    "AUDIO GENERATION ERROR",
                    f"Audio stream yielded {type(chunk).__name__}, not bytes.",
                )
        except DiagnosticError:
            raise
        except Exception as exc:
            if getattr(exc, "status_code", None) is not None or type(exc).__name__ in {
                "ApiError",
                "UnprocessableEntityError",
            }:
                raise categorize_sdk_error(exc, api_key) from exc
            raise DiagnosticError(
                "AUDIO GENERATION ERROR",
                _redact(
                    f"Failed while consuming audio stream: {type(exc).__name__}: {exc}",
                    api_key,
                ),
            ) from exc
        data = b"".join(chunks)
        if not data:
            raise DiagnosticError("AUDIO GENERATION ERROR", "Audio stream was empty.")
        return data, type_name
    raise DiagnosticError(
        "AUDIO GENERATION ERROR",
        f"Unsupported audio return type: {type_name}.",
    )


def sniff_format(data: bytes) -> str:
    if data.startswith(b"RIFF") and b"WAVE" in data[:16]:
        return "wav"
    if data.startswith(b"ID3") or data[:2] in {b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"}:
        return "mp3"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "mp4"
    return "pcm"


def write_audio_file(data: bytes, fmt: str) -> Path:
    try:
        suffix = ".wav" if fmt in {"wav", "pcm"} else f".{fmt}"
        handle = tempfile.NamedTemporaryFile(prefix="signbridge_tts_", suffix=suffix, delete=False)
        path = Path(handle.name)
        if fmt == "pcm":
            handle.close()
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(SAMPLE_RATE)
                wav.writeframes(data)
            return path
        handle.write(data)
        handle.close()
        return path
    except DiagnosticError:
        raise
    except Exception as exc:
        raise DiagnosticError(
            "AUDIO FILE ERROR",
            f"{type(exc).__name__}: could not write temp audio file.",
        ) from exc


def play_wav_file(path: Path) -> None:
    """Play a WAV file through sounddevice (macOS and Windows)."""
    try:
        import sounddevice as sd
    except Exception as exc:
        raise DiagnosticError(
            "PLAYBACK ERROR",
            f"sounddevice is not available: {type(exc).__name__}",
        ) from exc

    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            width = wav.getsampwidth()
            frames = wav.readframes(wav.getnframes())
    except Exception as exc:
        raise DiagnosticError(
            "PLAYBACK ERROR",
            f"Could not read WAV file: {type(exc).__name__}",
        ) from exc

    if width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise DiagnosticError("PLAYBACK ERROR", f"Unsupported sample width {width}.")

    if channels > 1:
        samples = samples.reshape(-1, channels)
    if samples.size == 0:
        raise DiagnosticError("PLAYBACK ERROR", "WAV file contained no samples.")

    try:
        sd.play(samples, samplerate=sample_rate, blocking=True)
    except Exception as exc:
        raise DiagnosticError(
            "PLAYBACK ERROR",
            f"sounddevice failed: {type(exc).__name__}: {exc}",
        ) from exc


def generate_audio(client: object, voice_id: str, api_key: str) -> object:
    try:
        convert = client.text_to_speech.convert  # type: ignore[attr-defined]
        return convert(
            voice_id=voice_id,
            text=TEST_PHRASE,
            model_id=MODEL_ID,
            output_format=OUTPUT_FORMAT,
        )
    except DiagnosticError:
        raise
    except Exception as exc:
        raise categorize_sdk_error(exc, api_key) from exc


def run_diagnostic(play: bool = True) -> int:
    print(f"ElevenLabs SDK: {sdk_version()}")
    print(f"Requested output_format: {OUTPUT_FORMAT}")
    print(f"Playback: sounddevice + stdlib wave")
    try:
        api_key, voice_id = require_credentials()
        client = make_client(api_key)
        print("ElevenLabs API: OK")

        raw = generate_audio(client, voice_id, api_key)
        data, type_name = materialize_audio(raw, api_key=api_key)
        print("Audio generated: OK")
        print(f"Returned audio type: {type_name}")

        fmt = sniff_format(data)
        path = write_audio_file(data, fmt)
        written_fmt = "wav" if fmt == "pcm" else fmt
        print(f"Audio file: {path}")
        print(f"Audio format: {written_fmt}")
        print(f"Audio size: {path.stat().st_size}")

        if play:
            if written_fmt != "wav":
                raise DiagnosticError(
                    "PLAYBACK ERROR",
                    f"Refusing to play non-WAV diagnostic file ({written_fmt}).",
                )
            play_wav_file(path)
            print("Playback attempted: OK")
        return 0
    except DiagnosticError as exc:
        print(exc.category)
        if exc.detail:
            print(exc.detail)
        return 1


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    play = "--no-play" not in args
    return run_diagnostic(play=play)


if __name__ == "__main__":
    sys.exit(main())
