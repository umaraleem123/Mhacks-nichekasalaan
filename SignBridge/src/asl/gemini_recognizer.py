"""UNUSED / EXPERIMENTAL — not part of the live SignBridge demo.

Kept for reference. The live path does not call Gemini.

Translate a signed ASL sentence into English using Gemini.

Uses the standard generateContent API (`client.models.generate_content`) of
the google-genai SDK. That call is synchronous: one HTTP request, one
response, no interaction polling. All Gemini access is confined to this
module so the rest of SignBridge stays unaware of the API.

The input is a `FrameSequence` from `sequence_capture` covering a complete
signing event: a small chronological list of JPEG frames, never a single
frame and never a full-rate video. ASL grammar lives in movement, so the
whole sampled sequence goes out as one request and comes back as one
English sentence.

Nothing here runs at import time, and no network call happens until
`translate_sequence` is called. The API key is read from the environment and is
never logged or included in an error message.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from src.asl.sequence_capture import FrameSequence

API_KEY_ENV_VAR = "GEMINI_API_KEY"
TIMEOUT_ENV = "SIGNBRIDGE_GEMINI_TIMEOUT_SECONDS"
DEFAULT_MODEL = "gemini-3.7-flash"
JPEG_MIME_TYPE = "image/jpeg"
VIDEO_MIME_TYPE = "video/mp4"
RESPONSE_MIME_TYPE = "application/json"

# Latency settings. A hackathon demo needs an answer in a few seconds, so
# reasoning depth is traded away deliberately.
DEFAULT_THINKING_LEVEL = "low"
DEFAULT_MAX_OUTPUT_TOKENS = 512
DEFAULT_REQUEST_TIMEOUT_SECONDS = 22.0
DEFAULT_POLL_INTERVAL_SECONDS = 0.75

# Official Interaction.status values from google-genai 2.28.
_NON_TERMINAL_STATUSES = frozenset({"queued", "in_progress"})
_TERMINAL_FAILURE_STATUSES = frozenset(
    {"failed", "cancelled", "budget_exceeded", "requires_action"}
)
_COMPLETED_STATUS = "completed"
_INCOMPLETE_STATUS = "incomplete"

# H.264 first for the smallest payload, MPEG-4 Part 2 as the portable
# fallback. Both were confirmed to encode through OpenCV's ffmpeg build.
VIDEO_CODECS: tuple[str, ...] = ("avc1", "mp4v")
MIN_VIDEO_FPS = 2.0

DEFAULT_JPEG_QUALITY = 70
DEFAULT_CONFIDENCE_THRESHOLD = 0.6

SYSTEM_PROMPT = """\
You are interpreting a short temporal sequence of images showing American Sign
Language. Analyze the sequence as a whole. Do not interpret each frame
independently. Use the hand movements across time to infer the intended ASL
message.

ASL is a complete natural language. Do not map one frame to one English word.
Translate the meaning of the whole utterance into one natural English sentence.

If the sequence is too ambiguous, set recognized to false and leave
english_translation empty.

Return ONLY this JSON object, with double quotes, no markdown, no explanation:

{"recognized": true, "english_translation": "Hello, how are you?", "confidence": 0.93, "notes": ""}

confidence must be a JSON number from 0.0 to 1.0, written like 0.93 — never
.0.93, 0.93.0, or a string. notes is a short clause or "".
"""

# Matches the SentenceTranslation fields below.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "recognized": {
            "type": "boolean",
            "description": "True only if the signing could be translated.",
        },
        "english_translation": {
            "type": "string",
            "description": "One natural English sentence, or empty if not recognized.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": (
                "JSON number from 0.0 to 1.0 inclusive. Example: 0.93. "
                "Never .0.93, 0.93.0, a percentage, or a string."
            ),
        },
        "notes": {
            "type": "string",
            "description": "Optional. One short clause, or an empty string.",
        },
    },
    "required": ["recognized", "english_translation", "confidence", "notes"],
}

# Preview of model text printed to the terminal. Never includes the API key.
MAX_RAW_PREVIEW = 400
_FENCE_BLOCK = re.compile(
    r"^```(?:json|JSON)?\s*\r?\n?(.*?)\r?\n?```\s*$",
    re.DOTALL,
)
_CONFIDENCE_FIELD = re.compile(
    r'("confidence"\s*:\s*)([^\n,}]*)',
    re.IGNORECASE,
)
_NUMBER_PIECE = re.compile(r"\d+(?:\.\d+)?|\.\d+")


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


class GeminiJsonError(GeminiRecognizerError):
    """The model text could not be turned into a JSON object."""


class GeminiRequestError(GeminiRecognizerError):
    """A Gemini request failed, with safe diagnostics attached.

    Holds the exception type, HTTP status, and API status code so local
    debugging has something to work with, while every string has been through
    `redact_secrets`.
    """

    BUSY_MESSAGE = "Gemini is currently busy. Please try again."
    TIMEOUT_MESSAGE = "Gemini timed out."
    INTERACTION_TIMEOUT_MESSAGE = "Gemini timed out."

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
        return "timeout" in self.error_type.lower() or self.error_type == "InteractionTimeout"

    @property
    def user_message(self) -> str:
        """What the window shows: plain, actionable, no provider text."""
        if self.error_type == "InteractionTimeout":
            return self.INTERACTION_TIMEOUT_MESSAGE
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
    request_timeout_seconds: float = field(default_factory=lambda: gemini_timeout_seconds())
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS

    # Latency levers.
    thinking_level: str = DEFAULT_THINKING_LEVEL
    max_output_tokens: int | None = DEFAULT_MAX_OUTPUT_TOKENS
    send_as_video: bool = False
    media_resolution: str | None = None  # None lets the API choose

    # 1 means a single attempt. The SDK never retries unless retry options are
    # supplied, and anything above 1 adds exponential backoff the user waits
    # through, so this stays at 1 to keep the demo responsive.
    retry_attempts: int = 1


@dataclass(frozen=True)
class SentenceTranslation:
    """A structured English translation of one signed sentence."""

    recognized: bool
    english_translation: str
    confidence: float
    notes: str = ""
    frame_count: int = 0
    raw: dict[str, Any] | None = None

    @property
    def is_unknown(self) -> bool:
        """True when there is no translation worth showing or speaking."""
        return not self.recognized or not self.english_translation.strip()

    @property
    def speakable_text(self) -> str:
        """Only the sentence. Never confidence, notes, or debug output."""
        return "" if self.is_unknown else self.english_translation.strip()

    @classmethod
    def unknown(
        cls,
        notes: str,
        frame_count: int = 0,
        raw: dict[str, Any] | None = None,
    ) -> SentenceTranslation:
        return cls(False, "", 0.0, notes, frame_count, raw)


def gemini_timeout_seconds() -> float:
    """Hard request timeout, from SIGNBRIDGE_GEMINI_TIMEOUT_SECONDS or 22s."""
    raw = (os.environ.get(TIMEOUT_ENV) or "").strip()
    if not raw:
        return DEFAULT_REQUEST_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_REQUEST_TIMEOUT_SECONDS
    return min(60.0, max(5.0, value))


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


def preview_model_text(
    text: Any, extra_secrets: tuple[str, ...] = (), limit: int = MAX_RAW_PREVIEW
) -> str:
    """Redacted, truncated model text, safe to print."""
    cleaned = redact_secrets(text, extra_secrets).replace("\r\n", "\n")
    if len(cleaned) > limit:
        return cleaned[:limit] + " ...(truncated)"
    return cleaned


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    fenced = _FENCE_BLOCK.match(stripped)
    if fenced:
        return fenced.group(1).strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json|JSON)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _extract_json_object(text: str) -> str | None:
    """Return the first brace-balanced object, or None if there is no '{'."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def _confidence_from_token(token: str) -> float | None:
    """Pull a 0–1 probability out of a messy confidence token."""
    pieces = _NUMBER_PIECE.findall(token or "")
    if not pieces:
        return None
    values: list[float] = []
    for piece in pieces:
        try:
            values.append(float(piece))
        except ValueError:
            continue
    if not values:
        return None
    in_unit = [value for value in values if 0.0 <= value <= 1.0]
    if in_unit:
        # Prefer a real fraction over a lone 0 created by a leading ".0.93".
        nonzero = [value for value in in_unit if value > 0.0]
        return nonzero[-1] if nonzero else in_unit[0]
    value = values[0]
    if 1.0 < value <= 100.0:
        return value / 100.0
    return min(1.0, max(0.0, value))


def _repair_confidence_field(text: str) -> str:
    """Fix the known malformed confidence spellings before json.loads."""

    def replace(match: re.Match[str]) -> str:
        prefix, raw = match.group(1), match.group(2)
        value = _confidence_from_token(raw)
        if value is None:
            return match.group(0)
        return f"{prefix}{json.dumps(value)}"

    return _CONFIDENCE_FIELD.sub(replace, text, count=1)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    if isinstance(value, (int, float)):
        return value != 0
    return False


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    translation = payload.get("english_translation", "")
    if translation is None:
        translation = ""
    notes = payload.get("notes", "")
    if notes is None:
        notes = ""
    confidence = payload.get("confidence", 0.0)
    if isinstance(confidence, str):
        parsed = _confidence_from_token(confidence)
        confidence = 0.0 if parsed is None else parsed
    else:
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
    return {
        "recognized": _coerce_bool(payload.get("recognized", False)),
        "english_translation": str(translation).strip(),
        "confidence": min(1.0, max(0.0, float(confidence))),
        "notes": str(notes).strip(),
    }


def parse_gemini_json(raw_text: object) -> dict[str, Any]:
    """Turn model text into a normalized sentence payload.

    Accepts a bare object, markdown fences, or surrounding prose. Repairs the
    confidence spellings Gemini has been emitting (`.0.93`, `0.93.0`).
    Never uses eval.
    """
    if raw_text is None:
        raise GeminiJsonError("Gemini returned an empty response.")
    text = _strip_fences(str(raw_text))
    if not text.strip():
        raise GeminiJsonError("Gemini returned an empty response.")

    candidate = _extract_json_object(text) or text
    attempts = (candidate, _repair_confidence_field(candidate))
    last_error = "Gemini did not return valid JSON."
    for attempt in attempts:
        try:
            payload = json.loads(attempt)
        except json.JSONDecodeError as exc:
            last_error = f"Gemini did not return valid JSON: {exc.msg}"
            continue
        if not isinstance(payload, dict):
            raise GeminiJsonError(
                f"Expected a JSON object from Gemini, got {type(payload).__name__}."
            )
        return _normalize_payload(payload)
    raise GeminiJsonError(last_error)


def structured_response_format() -> Any:
    """SDK object that asks the Interactions API for schema-constrained JSON."""
    from google.genai import interactions

    return interactions.TextResponseFormat(
        type="text",
        mime_type=RESPONSE_MIME_TYPE,
        schema_=RESPONSE_SCHEMA,
    )


def interaction_status(interaction: Any) -> str:
    raw = getattr(interaction, "status", None)
    return str(raw).strip().lower() if raw is not None else ""


def interaction_diagnostics(
    interaction: Any, extra_secrets: tuple[str, ...] = ()
) -> str:
    """Safe local dump of why an interaction is not completed.

    Never includes credentials, request headers, or raw URLs.
    """
    status = interaction_status(interaction) or "(none)"
    if status in _NON_TERMINAL_STATUSES:
        lifecycle = "still processing (non-terminal)"
    elif status == _COMPLETED_STATUS:
        lifecycle = "terminated (completed)"
    elif status == _INCOMPLETE_STATUS:
        lifecycle = "terminated with incomplete results"
    elif status in _TERMINAL_FAILURE_STATUSES:
        lifecycle = "terminated (failure)"
    else:
        lifecycle = "unknown"

    lines = [
        "Gemini interaction diagnostics",
        f"Interaction id: {getattr(interaction, 'id', None) or '(none)'}",
        f"Status: {status}",
        f"Lifecycle: {lifecycle}",
        (
            "Continuation token: present"
            if getattr(interaction, "continuation_token", None)
            else "Continuation token: absent"
        ),
    ]

    errors = getattr(interaction, "errors", None) or []
    lines.append(f"Error field: {'yes' if errors else 'no'}")
    for err in errors:
        code = getattr(err, "code", "") or ""
        message = redact_secrets(getattr(err, "message", None) or err, extra_secrets)
        lines.append(f"Error: {code} {message}".strip())

    usage = getattr(interaction, "usage", None)
    if usage is not None:
        lines.append(
            "Usage:"
            f" input={getattr(usage, 'total_input_tokens', None)}"
            f" output={getattr(usage, 'total_output_tokens', None)}"
            f" thought={getattr(usage, 'total_thought_tokens', None)}"
            f" total={getattr(usage, 'total_tokens', None)}"
        )
    else:
        lines.append("Usage: (none)")

    steps = getattr(interaction, "steps", None) or []
    step_types = [str(getattr(step, "type", type(step).__name__)) for step in steps]
    lines.append(f"Step types: {', '.join(step_types) if step_types else '(none)'}")

    output_types: list[str] = []
    for step in steps:
        if getattr(step, "type", None) != "model_output":
            continue
        step_error = getattr(step, "error", None)
        if step_error is not None:
            lines.append(
                "Model output error: "
                + redact_secrets(
                    getattr(step_error, "message", None) or step_error,
                    extra_secrets,
                )
            )
        for item in getattr(step, "content", None) or []:
            output_types.append(str(getattr(item, "type", type(item).__name__)))
    lines.append(
        f"Output item types: {', '.join(output_types) if output_types else '(none)'}"
    )
    lines.append(
        f"Output text characters: {len(interaction_output_text(interaction))}"
    )
    return "\n".join(lines)


def incomplete_reason(
    interaction: Any,
    max_output_tokens: int | None,
    extra_secrets: tuple[str, ...] = (),
) -> str:
    """Explain an `incomplete` status from fields the SDK actually exposes."""
    errors = getattr(interaction, "errors", None) or []
    if errors:
        parts = [
            redact_secrets(getattr(err, "message", None) or err, extra_secrets)
            for err in errors
        ]
        return "; ".join(part for part in parts if part)

    usage = getattr(interaction, "usage", None)
    output = getattr(usage, "total_output_tokens", None) if usage is not None else None
    thought = getattr(usage, "total_thought_tokens", None) if usage is not None else None
    billed = 0
    if isinstance(output, int):
        billed += output
    if isinstance(thought, int):
        billed += thought
    if max_output_tokens and billed >= max_output_tokens:
        return (
            f"Generation stopped at max_output_tokens={max_output_tokens} "
            f"(output={output}, thought={thought})."
        )
    if getattr(interaction, "continuation_token", None):
        return (
            "Decode stopped early; a continuation token is present. "
            "This usually means the output token budget was reached."
        )
    return (
        "The interaction finished with status incomplete "
        "(officially: completed but incomplete results, e.g. hitting max_tokens)."
    )


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
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self.config = config or GeminiConfig()
        self._client = client
        self._api_key = api_key
        self._clock = clock or time.monotonic
        self._sleep = sleeper or time.sleep

    @property
    def model(self) -> str:
        return self.config.model

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

    def _http_options(self) -> Any:
        """Hard timeout in milliseconds, and never more than one HTTP attempt."""
        from google.genai import types

        attempts = 1 if self.config.retry_attempts <= 1 else self.config.retry_attempts
        return types.HttpOptions(
            timeout=int(self.config.request_timeout_seconds * 1000),
            retry_options=types.HttpRetryOptions(attempts=attempts),
        )

    def _thinking_config(self) -> Any:
        from google.genai import types

        names = {
            "minimal": types.ThinkingLevel.MINIMAL,
            "low": types.ThinkingLevel.LOW,
            "medium": types.ThinkingLevel.MEDIUM,
            "high": types.ThinkingLevel.HIGH,
        }
        level = names.get(self.config.thinking_level.lower(), types.ThinkingLevel.LOW)
        return types.ThinkingConfig(thinking_level=level, include_thoughts=False)

    def _secrets(self) -> tuple[str, ...]:
        return (self._api_key or "",)

    def _await_terminal(self, client: Any, interaction: Any) -> Any:
        """Poll only while the interaction is queued or in progress.

        `create` is foreground by default and should already return a
        terminal status. `get` exists for the cases it does not
        (`queued`, `in_progress`). `incomplete` is terminal and is not polled.
        """
        deadline = self._clock() + self.config.request_timeout_seconds
        current = interaction

        while interaction_status(current) in _NON_TERMINAL_STATUSES:
            remaining = deadline - self._clock()
            if remaining <= 0:
                print(interaction_diagnostics(current, self._secrets()), file=sys.stderr)
                raise GeminiRequestError(
                    "InteractionTimeout",
                    api_status=interaction_status(current) or None,
                    message=GeminiRequestError.INTERACTION_TIMEOUT_MESSAGE,
                )

            interaction_id = getattr(current, "id", None)
            if not interaction_id:
                print(interaction_diagnostics(current, self._secrets()), file=sys.stderr)
                raise GeminiRequestError(
                    "InteractionNotCompleted",
                    api_status=interaction_status(current) or None,
                    message=redact_secrets(
                        "The interaction is still processing and has no id to poll.",
                        self._secrets(),
                    ),
                )

            self._sleep(min(self.config.poll_interval_seconds, remaining))
            if self._clock() >= deadline:
                print(interaction_diagnostics(current, self._secrets()), file=sys.stderr)
                raise GeminiRequestError(
                    "InteractionTimeout",
                    api_status=interaction_status(current) or None,
                    message=GeminiRequestError.INTERACTION_TIMEOUT_MESSAGE,
                )

            getter = getattr(getattr(client, "interactions", None), "get", None)
            if getter is None:
                print(interaction_diagnostics(current, self._secrets()), file=sys.stderr)
                raise GeminiRequestError(
                    "InteractionNotCompleted",
                    api_status=interaction_status(current) or None,
                    message="The interaction is still processing and get() is unavailable.",
                )
            try:
                current = getter(
                    str(interaction_id),
                    timeout=min(self.config.request_timeout_seconds, max(1.0, remaining)),
                )
            except GeminiRecognizerError:
                raise
            except Exception as exc:
                raise GeminiRequestError.from_exception(
                    exc, extra_secrets=self._secrets()
                ) from exc

        return current

    def _raise_for_status(self, interaction: Any) -> None:
        """Turn a failed or empty incomplete interaction into an error."""
        status = interaction_status(interaction)
        if not status or status == _COMPLETED_STATUS:
            return

        print(interaction_diagnostics(interaction, self._secrets()), file=sys.stderr)

        if status in _NON_TERMINAL_STATUSES:
            raise GeminiRequestError(
                "InteractionNotCompleted",
                api_status=status,
                message=redact_secrets(
                    f"The interaction is still {status}.",
                    self._secrets(),
                ),
            )

        if status == _INCOMPLETE_STATUS:
            if interaction_output_text(interaction).strip():
                # Officially terminal: results may still be usable JSON.
                return
            raise GeminiRequestError(
                "InteractionNotCompleted",
                api_status=status,
                message=redact_secrets(
                    incomplete_reason(
                        interaction,
                        self.config.max_output_tokens,
                        self._secrets(),
                    ),
                    self._secrets(),
                ),
            )

        if status in _TERMINAL_FAILURE_STATUSES:
            details = "; ".join(
                redact_secrets(getattr(err, "message", None) or err, self._secrets())
                for err in (getattr(interaction, "errors", None) or [])
            )
            raise GeminiRequestError(
                "InteractionNotCompleted",
                api_status=status,
                message=details or f"The interaction finished with status {status}.",
            )

    def translate_sequence(self, sequence: FrameSequence) -> SentenceTranslation:
        """Translate one signed sentence. Exactly one generate_content call."""
        if not isinstance(sequence, FrameSequence):
            raise GeminiRecognizerError(
                "translate_sequence expects a FrameSequence from "
                "sequence_capture, not a single frame."
            )
        if not sequence.shows_movement:
            return SentenceTranslation.unknown(
                "Too few frames to show movement; signing needs a sequence.",
                len(sequence),
            )

        client = self._ensure_client()
        contents = self._build_generate_contents(sequence)
        print(
            f"Gemini request started... ({len(sequence)} frames, "
            f"{self.config.request_timeout_seconds:g}s timeout)",
            flush=True,
        )
        started = time.perf_counter()
        try:
            from google.genai import types

            response = client.models.generate_content(
                model=self.config.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_output_tokens,
                    response_mime_type=RESPONSE_MIME_TYPE,
                    response_schema=RESPONSE_SCHEMA,
                    thinking_config=self._thinking_config(),
                    http_options=self._http_options(),
                ),
            )
        except GeminiRecognizerError:
            raise
        except Exception as exc:
            raise GeminiRequestError.from_exception(
                exc, extra_secrets=self._secrets()
            ) from exc

        elapsed = time.perf_counter() - started
        print(f"Gemini completed in {elapsed:.1f} sec", flush=True)
        text = getattr(response, "text", None) or ""
        return self._parse_model_text(str(text), len(sequence))

    def _build_generate_contents(self, sequence: FrameSequence) -> list[Any]:
        """One chronological JPEG list. Never a full-rate video, never one frame."""
        from google.genai import types

        jpegs = encode_frames_to_jpeg(sequence, self.config.jpeg_quality)
        total = len(jpegs)
        parts: list[Any] = [
            types.Part.from_text(
                text=(
                    f"You are interpreting a short temporal sequence of {total} "
                    f"images covering {sequence.duration:.2f} seconds of American "
                    "Sign Language. Analyze the sequence as a whole. Do not "
                    "interpret each frame independently. Use the hand movements "
                    "across time to infer the intended ASL message. Return the "
                    "JSON object only."
                )
            )
        ]
        for position, (jpeg, timestamp) in enumerate(zip(jpegs, sequence.timestamps)):
            parts.append(
                types.Part.from_text(
                    text=f"Frame {position + 1} of {total}, t = {timestamp:.2f}s:"
                )
            )
            parts.append(
                types.Part.from_bytes(data=jpeg, mime_type=JPEG_MIME_TYPE)
            )
        return parts

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
            # Static processing: the clip is a few seconds of one sentence,
            # so there is nothing for agentic search to explore. Passing fps
            # explicitly matters -- default video sampling would thin the clip
            # and destroy the movement between signs.
            processing=interactions.StaticMediaProcessing(type="static", fps=fps),
        )
        if self.config.media_resolution:
            content.resolution = self.config.media_resolution

        return [
            interactions.TextContent(
                type="text",
                text=(
                    f"This is one continuous {sequence.duration:.2f} second "
                    f"recording of a complete ASL signing event, "
                    f"{len(sequence)} frames at {fps:.1f} fps, in "
                    "chronological order. Interpret the entire sequence and "
                    "translate its meaning into one natural English sentence. "
                    "Do not emit one word per frame. If the signing is too "
                    "ambiguous, leave english_translation empty."
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
                    f"The following {total} frames are one continuous "
                    f"{sequence.duration:.2f} second recording of a complete "
                    "ASL signing event, in chronological order. Interpret the "
                    "entire sequence and translate its meaning into one "
                    "natural English sentence. Do not emit one word per "
                    "frame. If the signing is too ambiguous, leave "
                    "english_translation empty."
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

    def _parse_response(self, interaction: Any, frame_count: int) -> SentenceTranslation:
        """Validate Gemini's JSON and refuse to pass on a shaky translation."""
        self._raise_for_status(interaction)
        return self._parse_model_text(interaction_output_text(interaction), frame_count)

    def _parse_model_text(self, text: str, frame_count: int) -> SentenceTranslation:
        preview = preview_model_text(text, self._secrets())
        print(f"Gemini raw model text (sanitized): {preview}", file=sys.stderr)

        try:
            payload = parse_gemini_json(text)
        except GeminiJsonError as exc:
            message = redact_secrets(exc, self._secrets())
            print(f"Gemini JSON parse failed: {message}", file=sys.stderr)
            return SentenceTranslation.unknown(
                message,
                frame_count,
                {"raw_text": preview, "parse_error": message},
            )

        translation = payload["english_translation"]
        notes = payload["notes"]
        confidence = payload["confidence"]
        recognized = payload["recognized"] and bool(translation)
        if recognized and confidence < self.config.confidence_threshold:
            return SentenceTranslation(
                False,
                "",
                confidence,
                (
                    f"Below the {self.config.confidence_threshold:.0%} "
                    f"confidence threshold. {notes}"
                ).strip(),
                frame_count,
                payload,
            )
        if not recognized:
            return SentenceTranslation(
                False, "", confidence, notes, frame_count, payload
            )
        return SentenceTranslation(
            True, translation, confidence, notes, frame_count, payload
        )
