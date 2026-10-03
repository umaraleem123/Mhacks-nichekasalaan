"""Pipeline spelling helpers (no webcam / no model required for buffer ops)."""

from asl_recognition.pipeline import ASLRecognizer


def test_spelling_buffer_helpers(monkeypatch) -> None:
    # Avoid loading MediaPipe by stubbing HandTracker inside ASLRecognizer.__init__
    class _StubTracker:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

        def detect(self, frame):
            return []

    monkeypatch.setattr("asl_recognition.pipeline.HandTracker", _StubTracker)
    rec = ASLRecognizer(history=1, min_votes=1)
    assert rec.commit_letter("H") == "H"
    assert rec.commit_letter("I") == "I"
    assert rec.spelled == "HI"
    rec.backspace()
    assert rec.spelled == "H"
    rec.clear_text()
    assert rec.spelled == ""
    rec.close()
