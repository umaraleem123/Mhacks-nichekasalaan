"""Recognize an ASL sign from a short frame sequence using Gemini.

All Gemini access is confined to this module so the rest of SignBridge stays
unaware of the API. The input is a `FrameSequence` from `sequence_capture`,
never a single frame, because an ASL sign is defined partly by movement.

Nothing here runs at import time, and no network call happens until
`recognize_sequence` is called. The API key is read from the environment and is
never logged or included in an error message.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from src.asl import SIGNS
from src.asl.sequence_capture import FrameSequence

API_KEY_ENV_VAR = "GEMINI_API_KEY"
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_JPEG_QUALITY = 80
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0

UNKNOWN_SIGN = "unknown"

# What Gemini is allowed to answer: the MVP vocabulary, plus a way to decline.
SUPPORTED_SIGNS: tuple[str, ...] = tuple(SIGNS)
ALLOWED_ANSWERS: tuple[str, ...] = SUPPORTED_SIGNS + (UNKNOWN_SIGN,)

SYSTEM_PROMPT = f"""\
You are an American Sign Language (ASL) recognition component inside an
assistive communication tool. Your judgments affect how a Deaf or
hard-of-hearing person is understood, so being accurate matters far more than
being decisive.

ASL is a complete natural language with its own grammar and phonology. It is
not English spelled out with hands, and it is not a set of static gestures.

THE INPUT IS A TEMPORAL SEQUENCE, NOT A PHOTOGRAPH
You receive several still frames sampled in chronological order from a short
video clip of one ASL signing event, roughly one to three seconds long. Each
image is labeled with its position in the sequence and the time in seconds
since the clip began. Read them as consecutive moments of one continuous
motion, and interpret the clip temporally.

The change between consecutive frames is itself the evidence. Movement is not
noise to look past; it is frequently the only thing that separates one sign
from another.

Never treat any single frame as the whole sign, and never classify from one
frame alone. One frame is a slice through a movement. A sign and a completely
unrelated sign can pass through identical handshapes at different instants, so
a pose you recognize in frame three is evidence about frame three only.

Expect the hand to change during the sign. Its location, its orientation, and
its configuration can all differ between the start and the end of the clip, and
a handshape that appears mid-sign may be a transition rather than the sign
itself. Examine every frame and decide what happened across the whole clip
before you answer.

WHAT TO ANALYZE
ASL signs are distinguished by five parameters, and you should weigh all of
them:
1. Handshape, including how it changes during the sign.
2. Palm orientation and how it rotates.
3. Location relative to the body, head, and neutral signing space.
4. Movement: its path, direction, repetition, and size. This is the parameter a
   single frame cannot show, and often the one that decides the answer.
5. Non-manual signals such as head movement, when visible.

One or two hands may be involved. Note whether one hand is dominant and whether
the other is a stationary base, and do not assume a sign is one-handed just
because only one hand is clearly visible in some frames.

VOCABULARY YOU MAY REPORT
You may only answer with one of: {", ".join(SUPPORTED_SIGNS)}.

These are the only signs this prototype supports. For reference:
- "hello": a flat hand near the forehead or temple moving outward and away, like
  a salute that relaxes into a wave.
- "yes": a fist, palm facing forward, bobbing at the wrist like a nodding head.
- "no": the index and middle fingers close down onto the thumb in a single quick
  snapping motion.

ANSWERING HONESTLY
Answer "unknown" whenever the clip does not give you sufficient evidence for one
of those signs. "unknown" is a correct, useful answer and is strongly preferred
over a guess. Specifically answer "unknown" when:
- the signer is clearly producing some sign outside the supported vocabulary;
- the hand is moving into or out of position rather than signing;
- the hand is still, idle, or resting;
- motion blur, framing, or occlusion hides a parameter you would need;
- the movement is consistent with more than one supported sign;
- no hand is visible.

Do not invent a sign because you feel obliged to choose. Do not pick the
nearest supported sign for a gesture that is not one of them. Do not let a
single convincing frame override what the rest of the sequence shows.

CONFIDENCE
Report confidence as your genuine probability that the sign is correct, from
0.0 to 1.0. It should be low when evidence is thin. Do not inflate it, and do
not report high confidence merely because the handshape looked familiar.

In the description field, state briefly what movement you actually observed
across the frames, and say which parameter decided your answer or which one was
missing. Describe only what is in the frames.
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
            "description": "The movement observed across the sequence.",
        },
    },
    "required": ["recognized", "sign", "confidence", "description"],
}


class GeminiRecognizerError(RuntimeError):
    """Gemini could not be configured or did not return a usable answer."""


@dataclass(frozen=True)
class GeminiConfig:
    """Tunables for the Gemini call."""

    model: str = DEFAULT_MODEL
    temperature: float = 0.0  # recognition should be repeatable, not creative
    jpeg_quality: int = DEFAULT_JPEG_QUALITY
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    max_output_tokens: int | None = None
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS


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

        self._client = genai.Client(api_key=self._api_key or api_key_from_env())
        return self._client

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
        contents = self._build_contents(sequence)

        try:
            from google.genai import types

            response = client.models.generate_content(
                model=self.config.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_output_tokens,
                    # Bounded so a stalled request cannot hang the demo.
                    http_options=types.HttpOptions(
                        timeout=int(self.config.request_timeout_seconds * 1000)
                    ),
                ),
            )
        except GeminiRecognizerError:
            raise
        except Exception as exc:
            # Deliberately reports the exception type only: provider messages
            # can echo request details.
            raise GeminiRecognizerError(
                f"The Gemini request failed ({type(exc).__name__}). Check the "
                "network connection and that the API key is valid."
            ) from exc

        return self._parse_response(response, len(sequence))

    def _build_contents(self, sequence: FrameSequence) -> list[Any]:
        """Interleave labels and images so the ordering is explicit to Gemini."""
        from google.genai import types

        jpegs = encode_frames_to_jpeg(sequence, self.config.jpeg_quality)
        total = len(jpegs)

        parts: list[Any] = [
            types.Part.from_text(
                text=(
                    f"The following {total} frames are one continuous clip of "
                    f"{sequence.duration:.2f} seconds, in chronological order. "
                    "Identify the single ASL sign being produced across the "
                    "whole clip, or answer unknown."
                )
            )
        ]
        for position, (jpeg, timestamp) in enumerate(zip(jpegs, sequence.timestamps)):
            parts.append(
                types.Part.from_text(
                    text=f"Frame {position + 1} of {total}, t = {timestamp:.2f}s:"
                )
            )
            parts.append(types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"))
        return parts

    def _parse_response(self, response: Any, frame_count: int) -> SignInterpretation:
        """Validate Gemini's JSON and refuse answers outside the vocabulary."""
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            return SignInterpretation.unknown(
                "Gemini returned an empty response.", frame_count
            )

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise GeminiRecognizerError(
                f"Gemini did not return valid JSON: {exc}"
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
