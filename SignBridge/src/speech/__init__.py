"""Speech output and input.

Text-to-speech lives in `text_to_speech`. Speech-to-text is not implemented yet.
"""

from src.speech.text_to_speech import TextToSpeech, speak

__all__ = ["TextToSpeech", "speak"]
