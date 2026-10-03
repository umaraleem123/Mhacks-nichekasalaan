"""Recognize an ASL sign from a short frame sequence using Gemini.

Uses the Interactions API (`client.interactions.create`) of the google-genai
SDK with Gemini 3.8 Flash. All Gemini access is confined to this module so the
rest of SignBridge stays unaware of the API.

The input is a `FrameSequence` from `sequence_capture`, never a single frame,
because an ASL sign is defined partly by movement. The whole ordered sequence
goes out as one interaction: a flat list of content items alternating a text
label and the JPEG for each frame, so the chronology is explicit.

Nothing here runs at import time, and no network call happens until
`recognize_sequence` is called. The API key is read from the environment and is
never logged or included in an error message.
"""

from __future__ import annotations

import base64
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.asl import SIGNS
from src.asl.sequence_capture import FrameSequence

API_KEY_ENV_VAR = "GEMINI_API_KEY"
DEFAULT_MODEL = "gemini-3.8-flash"
JPEG_MIME_TYPE = "image/jpeg"
VIDEO_MIME_TYPE = "video/mp4"
RESPONSE_MIME_TYPE = "application/json"

# Latency settings. A hackathon demo needs an answer in a few seconds, so
# reasoning depth is traded away deliberately.
DEFAULT_THINKING_LEVEL = "low"
DEFAULT_MAX_OUTPUT_TOKENS = 200
DEFAULT_REQUEST_TIMEOUT_SECONDS = 20.0

# H.264 first for the smallest payload, MPEG-4 Part 2 as the portable
# fallback. Both were confirmed to encode through OpenCV's ffmpeg build.
VIDEO_CODECS: tuple[str, ...] = ("avc1", "mp4v")
MIN_VIDEO_FPS = 2.0

# Interaction statuses that mean no usable answer came back.
_FAILED_STATUSES = frozenset(
    {"failed", "cancelled", "budget_exceeded", "incomplete", "requires_action"}
)
DEFAULT_JPEG_QUALITY = 80
DEFAULT_CONFIDENCE_THRESHOLD = 0.6

UNKNOWN_SIGN = "unknown"

# What Gemini is allowed to answer: the MVP vocabulary, plus a way to decline.
SUPPORTED_SIGNS: tuple[str, ...] = tuple(SIGNS)
ALLOWED_ANSWERS: tuple[str, ...] = SUPPORTED_SIGNS + (UNKNOWN_SIGN,)

SYSTEM_PROMPT = f"""\
You recognize American Sign Language (ASL) signs for an assistive tool. ASL is
a real language, not a set of static poses. Accuracy matters more than
answering.

INPUT: a short clip of ONE signing event, about 1-2 seconds, given either as a
video or as frames in chronological order. Interpret it temporally.

The change between frames is the evidence. Movement is often the only thing
separating two signs, so never classify from a single frame: a frame is one
slice through a motion, and unrelated signs pass through identical handshapes
at different instants. The hand's location, orientation, and configuration can
all change mid-sign, and a mid-sign pose may be only a transition. Judge the
whole clip.

Weigh all five parameters: handshape, palm orientation, location, movement
(path, direction, repetition), and any visible non-manual signals. One or both
hands may be involved; do not assume one-handed just because a frame shows one.

VOCABULARY. Answer only with: {", ".join(SUPPORTED_SIGNS)}, or "unknown".
This is a limited-vocabulary prototype, not full ASL translation.
- "hello": flat hand at forehead or temple, moving outward and away.
- "yes": fist, palm forward, bobbing at the wrist like a nod.
- "no": index and middle fingers snap down onto the thumb, once and quickly.

ANSWER "unknown" whenever evidence is insufficient: a sign outside the
vocabulary, a hand moving into or out of position, a still or resting hand,
blur or occlusion hiding a parameter, motion fitting more than one sign, or no
hand visible. "unknown" is a correct answer and is preferred over a guess.
Never invent a sign, and never pick the nearest supported one for a gesture
that is not in the list.

CONFIDENCE: your genuine probability from 0.0 to 1.0. Keep it low when the
evidence is thin.

OUTPUT: the JSON object only. No reasoning, no step-by-step explanation, no
prose outside the fields. Keep `description` to at most one short clause naming
the movement you saw, or omit it.
"""

# Matches the SignInterpretation fields below.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "recognized": {
            "type": "boolean",
            "description": "True only if a supported sign was identified.",
        },
        "sign": {
            "type": "string",
            "enum": list(ALLOWED_ANSWERS),
            "description": "The sign, or 'unknown' when evidence is insufficient.",
        },
        "confidence": {
            "type": "number",
            "description": "Probability from 0.0 to 1.0 that the sign is correct.",
        },
        "description": {
            "type": "string",
            "description": "Optional. At most one short clause; may be omitted.",
        },
    },
    # `description` is deliberately not required: the recognition result does
    # not depend on it, and demanding prose costs output tokens and latency.
    "required": ["recognized", "sign", "confidence"],
}


REDACTED = "***REDACTED***"
MAX_DIAGNOSTIC_MESSAGE = 600

# Applied to anything from the provider before it is printed. The first pattern
# catches a Google API key by shape, the rest catch credentials carried in
# headers or query strings.
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"AIza[0-9A-Za-z_\-]{10,}"), REDACTED),
    (re.compile(r"(?i)([?&]key=)[^&\s\"']+"), r"\1" + REDACTED),
    # Swallows an optional auth scheme, so "Authorization: Bearer <token>"
    # redacts the token and not just the word "Bearer".
    (
        re.compile(
            r"(?i)((?:x-goog-api-key|authorization|api[_-]?key)[\"']?\s*[:=]\s*[\"']?)"
            r"(?:bearer\s+|basic\s+|token\s+)?[^\s\"',&}]+"
        ),
        r"\1" + REDACTED,
    ),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}"), r"\1" + REDACTED),
)


def redact_secrets(text: Any, extra_secrets: tuple[str, ...] = ()) -> str:
    """Strip credentials out of provider text so it is safe to print.

    Removes the configured key by exact match, then anything that merely looks
    like a credential, so an unexpected token shape is still caught.
    """
    if text is None:
        return ""

    cleaned = str(text)
    candidates = (*extra_secrets, os.environ.get(API_KEY_ENV_VAR, ""))
    for secret in candidates:
        secret = (secret or "").strip()
        if len(secret) >= 8:  # ignore placeholders too short to be real
            cleaned = cleaned.replace(secret, REDACTED)

    for pattern, replacement in _SECRET_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned


class GeminiRecognizerError(RuntimeError):
    """Gemini could not be configured or did not return a usable answer."""


class GeminiRequestError(GeminiRecognizerError):
    """A Gemini request failed, with safe diagnostics attached.

    Holds the exception type, HTTP status, and API status code so local
    debugging has something to work with, while every string has been through
    `redact_secrets`.
    """

    BUSY_MESSAGE = "Gemini is currently busy. Please try again."
    TIMEOUT_MESSAGE = "Gemini request timed out."

    # 429 too many requests, 503 unavailable: the service is overloaded.
    BUSY_STATUS_CODES = frozenset({429, 503})

    def __init__(
        self,
        error_type: str,
        status_code: int | None = None,
        api_status: str | None = None,
        message: str = "",
    ) -> None:
        self.error_type = error_type
        self.status_code = status_code
        self.api_status = api_status
        self.sanitized_message = message
        super().__init__(self.short_message)

    @property
    def is_busy(self) -> bool:
        """Transient overload, worth retrying by hand but not automatically."""
        if self.status_code in self.BUSY_STATUS_CODES:
            return True
        status = (self.api_status or "").upper()
        return status in {"RESOURCE_EXHAUSTED", "UNAVAILABLE"}

    @property
    def is_timeout(self) -> bool:
        if self.status_code in (408, 504):
            return True
        return "timeout" in self.error_type.lower()

    @property
    def user_message(self) -> str:
        """What the window shows: plain, actionable, no provider text."""
        if self.is_timeout:
            return self.TIMEOUT_MESSAGE
        if self.is_busy:
            return self.BUSY_MESSAGE
        return self.short_message

    @classmethod
    def from_exception(
        cls, exc: BaseException, extra_secrets: tuple[str, ...] = ()
    ) -> GeminiRequestError:
        """Pull whatever detail the exception carries, redacting as we go.

        `google.genai.errors.APIError` exposes `code` (HTTP status), `status`
        (for example INVALID_ARGUMENT), and `message`. Network and timeout
        errors have none of those, so each is read defensively.
        """
        status_code = getattr(exc, "code", None)
        if not isinstance(status_code, int):
            status_code = None

        api_status = getattr(exc, "status", None)
        api_status = str(api_status) if api_status else None

        raw = getattr(exc, "message", None) or str(exc)
        message = redact_secrets(raw, extra_secrets)
        if len(message) > MAX_DIAGNOSTIC_MESSAGE:
            message = message[:MAX_DIAGNOSTIC_MESSAGE] + " ...(truncated)"

        return cls(type(exc).__name__, status_code, api_status, message)

    @property
    def short_message(self) -> str:
        """One line for the on-screen display."""
        detail = " ".join(
            part for part in (
                f"HTTP {self.status_code}" if self.status_code else "",
                self.api_status or "",
            ) if part
        )
        suffix = f" ({detail})" if detail else f" ({self.error_type})"
        return f"The Gemini request failed{suffix}. See the terminal for details."

    def diagnostics(self) -> str:
        """The multi-line block printed to the terminal. Contains no secrets."""
        lines = ["Gemini request failed", f"Error type: {self.error_type}"]
        if self.status_code is not None:
            lines.append(f"Status code: {self.status_code}")
        if self.api_status:
            lines.append(f"API status: {self.api_status}")
        lines.append(f"Error: {self.sanitized_message or '(no message provided)'}")
        return "\n".join(lines)


@dataclass(frozen=True)
class GeminiConfig:
    """Tunables for the Gemini call."""

    model: str = DEFAULT_MODEL
    temperature: float = 0.0  # recognition should be repeatable, not creative
    jpeg_quality: int = DEFAULT_JPEG_QUALITY
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS

    # Latency levers.
    thinking_level: str = DEFAULT_THINKING_LEVEL
    max_output_tokens: int | None = DEFAULT_MAX_OUTPUT_TOKENS
    send_as_video: bool = True
    media_resolution: str | None = None  # None lets the API choose

    # 1 means a single attempt. The SDK never retries unless retry options are
    # supplied, and anything above 1 adds exponential backoff the user waits
    # through, so this stays at 1 to keep the demo responsive.
    retry_attempts: int = 1


@dataclass(frozen=True)
class SignInterpretation:
    """A structured answer from Gemini."""

    recognized: bool
    sign: str
    confidence: float
    description: str
    frame_count: int = 0
    raw: dict[str, Any] | None = None

    @property
    def is_unknown(self) -> bool:
        return not self.recognized or self.sign == UNKNOWN_SIGN

    @classmethod
    def unknown(cls, description: str, frame_count: int = 0) -> SignInterpretation:
        return cls(False, UNKNOWN_SIGN, 0.0, description, frame_count)


def api_key_from_env(env_var: str = API_KEY_ENV_VAR) -> str:
    """Read the API key, or explain how to set it. Never echoes the value."""
    key = (os.environ.get(env_var) or "").strip()
    if not key:
        raise GeminiRecognizerError(
            f"{env_var} is not set.\n"
            "Create a key at https://aistudio.google.com/apikey, then:\n"
            f"  macOS:   export {env_var}='your-key-here'\n"
            f"  Windows: $env:{env_var} = 'your-key-here'\n"
            "See the README section 'Gemini ASL Recognition'."
        )
    return key


def interaction_output_text(interaction: Any) -> str:
    """Pull the model's text out of a completed interaction.

    The Interactions API returns a list of steps; the answer lives in the text
    content of the `model_output` step. `Interaction.output_text` exists but is
    deprecated, so the steps are walked instead.
    """
    chunks: list[str] = []
    for step in getattr(interaction, "steps", None) or []:
        if getattr(step, "type", None) != "model_output":
            continue
        for item in getattr(step, "content", None) or []:
            if getattr(item, "type", None) == "text" and getattr(item, "text", None):
                chunks.append(str(item.text))
    return "".join(chunks)


def sequence_fps(sequence: FrameSequence) -> float:
    """Playback rate implied by the frame timestamps.

    Used both for encoding and for telling Gemini how to sample the clip, so
    the video plays back at the speed the movement actually happened.
    """
    if len(sequence) < 2 or sequence.duration <= 0:
        return MIN_VIDEO_FPS
    return max(MIN_VIDEO_FPS, (len(sequence) - 1) / sequence.duration)


def encode_sequence_to_video(
    sequence: FrameSequence, codecs: tuple[str, ...] = VIDEO_CODECS
) -> tuple[bytes, float] | None:
    """Encode the frames into a short MP4, returning (bytes, fps).

    Returns None when no codec in `codecs` is usable, which lets the caller
    fall back to sending individual frames rather than failing the request.

    OpenCV's writer needs a path, so this goes through a temporary directory
    that is removed immediately; the clip is never left on disk.
    """
    if not sequence.shows_movement:
        return None

    first = sequence.frames[0].image
    height, width = first.shape[:2]
    fps = sequence_fps(sequence)

    with tempfile.TemporaryDirectory(prefix="signbridge-") as directory:
        path = Path(directory) / "clip.mp4"
        for codec in codecs:
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height)
            )
            if not writer.isOpened():
                writer.release()
                continue
            try:
                for frame in sequence:
                    writer.write(frame.image)
            finally:
                writer.release()

            if path.is_file() and path.stat().st_size > 0:
                return path.read_bytes(), fps
            path.unlink(missing_ok=True)
    return None


def encode_frames_to_jpeg(
    sequence: FrameSequence, quality: int = DEFAULT_JPEG_QUALITY
) -> list[bytes]:
    """Encode the sequence to JPEG bytes, preserving order."""
    encoded: list[bytes] = []
    for position, frame in enumerate(sequence):
        ok, buffer = cv2.imencode(
            ".jpg", frame.image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        )
        if not ok:
            raise GeminiRecognizerError(f"Could not encode frame {position} as JPEG.")
        encoded.append(buffer.tobytes())
    return encoded


class GeminiSignRecognizer:
    """Sends a frame sequence to Gemini and returns a structured result.

    Pass `client` to supply a stub in tests; the real client is built lazily so
    importing this module never needs credentials or a network connection.
    """

    def __init__(
        self,
        api_key: str | None = None,
        config: GeminiConfig | None = None,
        client: Any | None = None,
    ) -> None:
        self.config = config or GeminiConfig()
        self._client = client
        self._api_key = api_key

    @property
    def supported_signs(self) -> tuple[str, ...]:
        return SUPPORTED_SIGNS

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            from google import genai
        except ImportError as exc:
            raise GeminiRecognizerError(
                "The google-genai package is not installed. Install it with:\n"
                "  python -m pip install -r requirements.txt"
            ) from exc

        self._client = genai.Client(
            api_key=self._api_key or api_key_from_env(),
            http_options=self._http_options(),
        )
        return self._client

    def _http_options(self) -> Any | None:
        """Retry policy, or None to keep the SDK default of no retries.

        The SDK retries nothing unless retry options are supplied, so leaving
        this at None means one SPACE press is exactly one HTTP attempt. Raising
        `retry_attempts` opts into the SDK's own exponential backoff rather
        than adding a second retry loop here.
        """
        if self.config.retry_attempts <= 1:
            return None

        from google.genai import types

        return types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=self.config.retry_attempts)
        )

    def _raise_for_status(self, interaction: Any) -> None:
        """Turn a non-completed interaction into a diagnosable error."""
        status = getattr(interaction, "status", None)
        if status is None or str(status) not in _FAILED_STATUSES:
            return

        details = "; ".join(
            str(getattr(err, "message", None) or err)
            for err in (getattr(interaction, "errors", None) or [])
        )
        raise GeminiRequestError(
            "InteractionNotCompleted",
            api_status=str(status),
            message=redact_secrets(
                details or f"The interaction finished with status {status}.",
                (self._api_key or "",),
            ),
        )

    def recognize_sequence(self, sequence: FrameSequence) -> SignInterpretation:
        """Interpret a short frame sequence. Makes one Gemini call."""
        if not isinstance(sequence, FrameSequence):
            raise GeminiRecognizerError(
                "recognize_sequence expects a FrameSequence from "
                "sequence_capture, not a single frame."
            )
        if not sequence.shows_movement:
            return SignInterpretation.unknown(
                "Too few frames to show movement; a sign needs a sequence.",
                len(sequence),
            )

        client = self._ensure_client()
        model_input = self._build_input(sequence)

        try:
            from google.genai import interactions

            interaction = client.interactions.create(
                model=self.config.model,
                input=model_input,
                system_instruction=SYSTEM_PROMPT,
                response_format=RESPONSE_SCHEMA,
                response_mime_type=RESPONSE_MIME_TYPE,
                generation_config=interactions.GenerationConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_output_tokens,
                    # Classifying three signs needs recognition, not
                    # deliberation; deep thinking is the main latency cost.
                    thinking_level=self.config.thinking_level,
                    thinking_summaries="none",
                ),
                # Seconds here, unlike the older http_options milliseconds.
                timeout=self.config.request_timeout_seconds,
            )
        except GeminiRecognizerError:
            raise
        except Exception as exc:
            # Keeps the status and provider message for debugging, but only
            # after redact_secrets has been over them.
            raise GeminiRequestError.from_exception(
                exc, extra_secrets=(self._api_key or "",)
            ) from exc

        return self._parse_response(interaction, len(sequence))

    def _build_input(self, sequence: FrameSequence) -> list[Any]:
        """Build the interaction input for the whole clip, in one request.

        Prefers a short MP4, which carries the same frames in a fraction of
        the payload of the equivalent JPEGs. Falls back to the labeled frame
        sequence if no video codec is available, so the temporal information
        survives either way.
        """
        if self.config.send_as_video:
            encoded = encode_sequence_to_video(sequence)
            if encoded is not None:
                return self._build_video_input(sequence, *encoded)
        return self._build_frames_input(sequence)

    def _build_video_input(
        self, sequence: FrameSequence, video: bytes, fps: float
    ) -> list[Any]:
        """One video part, processed statically at the clip's own frame rate."""
        from google.genai import interactions

        content = interactions.VideoContent(
            type="video",
            mime_type=VIDEO_MIME_TYPE,
            data=base64.b64encode(video).decode("ascii"),
            # Static processing: the whole clip is 1-2 seconds, so there is
            # nothing for agentic search to explore. Passing fps explicitly
            # matters -- the default sampling would thin a clip this short down
            # to roughly one frame and destroy the movement.
            processing=interactions.StaticMediaProcessing(type="static", fps=fps),
        )
        if self.config.media_resolution:
            content.resolution = self.config.media_resolution

        return [
            interactions.TextContent(
                type="text",
                text=(
                    f"This is one continuous {sequence.duration:.2f} second clip "
                    f"of a single ASL signing event, {len(sequence)} frames at "
                    f"{fps:.1f} fps. Identify the sign produced across the whole "
                    "clip, or answer unknown."
                ),
            ),
            content,
        ]

    def _build_frames_input(self, sequence: FrameSequence) -> list[Any]:
        """Fallback: each frame labeled with its position and timestamp.

        The chronology survives in the labels rather than relying on list
        order alone. Still one interaction, never one request per frame.
        """
        from google.genai import interactions

        jpegs = encode_frames_to_jpeg(sequence, self.config.jpeg_quality)
        total = len(jpegs)

        items: list[Any] = [
            interactions.TextContent(
                type="text",
                text=(
                    f"The following {total} frames are one continuous clip of "
                    f"{sequence.duration:.2f} seconds, in chronological order. "
                    "Identify the single ASL sign being produced across the "
                    "whole clip, or answer unknown."
                ),
            )
        ]
        for position, (jpeg, timestamp) in enumerate(zip(jpegs, sequence.timestamps)):
            items.append(
                interactions.TextContent(
                    type="text",
                    text=f"Frame {position + 1} of {total}, t = {timestamp:.2f}s:",
                )
            )
            items.append(
                interactions.ImageContent(
                    type="image",
                    mime_type=JPEG_MIME_TYPE,
                    # The field takes base64 text, not raw bytes.
                    data=base64.b64encode(jpeg).decode("ascii"),
                )
            )
        return items

    def _parse_response(self, interaction: Any, frame_count: int) -> SignInterpretation:
        """Validate Gemini's JSON and refuse answers outside the vocabulary."""
        self._raise_for_status(interaction)

        text = interaction_output_text(interaction).strip()
        if not text:
            return SignInterpretation.unknown(
                "Gemini returned an empty response.", frame_count
            )

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise GeminiRecognizerError(
                "Gemini did not return valid JSON: "
                f"{redact_secrets(exc, (self._api_key or '',))}"
            ) from exc
        if not isinstance(payload, dict):
            raise GeminiRecognizerError(
                f"Expected a JSON object from Gemini, got {type(payload).__name__}."
            )

        sign = str(payload.get("sign", UNKNOWN_SIGN)).strip().lower()
        description = str(payload.get("description", "")).strip()
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))

        # A model can still answer outside the enum; treat that as unknown
        # rather than passing an unsupported label to the rest of the app.
        if sign not in ALLOWED_ANSWERS:
            return SignInterpretation(
                False, UNKNOWN_SIGN, confidence,
                f"Gemini reported an unsupported sign {sign!r}. {description}".strip(),
                frame_count, payload,
            )

        recognized = bool(payload.get("recognized", False)) and sign != UNKNOWN_SIGN
        if recognized and confidence < self.config.confidence_threshold:
            return SignInterpretation(
                False, UNKNOWN_SIGN, confidence,
                f"Below the {self.config.confidence_threshold:.0%} confidence "
                f"threshold for {sign!r}. {description}".strip(),
                frame_count, payload,
            )

        return SignInterpretation(
            recognized,
            sign if recognized else UNKNOWN_SIGN,
            confidence,
            description,
            frame_count,
            payload,
        )
