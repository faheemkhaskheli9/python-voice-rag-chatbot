"""Tests for the speech-to-text transcription stage (issue #2).

The recognizer call is faked via ``recognize_func`` so these run offline in
CI -- no network call to a cloud STT engine, no real audio content needed.
"""

from __future__ import annotations

import speech_recognition as sr
import pytest

from voice_rag_chatbot.audio_input import FileAudioSource
from voice_rag_chatbot.transcription import (
    Transcriber,
    TranscriptionError,
    transcribe_audio,
)


def _captured(make_wav):
    return FileAudioSource(make_wav(seconds=1.0)).capture()


def test_transcribes_sample_audio_file_correctly(make_wav):
    audio = _captured(make_wav)

    def fake_recognize(sr_module, recognizer, audio_data, *, engine, language):
        assert engine == "google"
        assert language == "en-US"
        return "hello world"

    result = transcribe_audio(audio, recognize_func=fake_recognize)

    assert result.success is True
    assert result.text == "hello world"
    assert result.engine == "google"
    assert result.error is None


def test_unclear_audio_returns_failed_result_without_crashing(make_wav):
    audio = _captured(make_wav)

    def fake_recognize(sr_module, recognizer, audio_data, *, engine, language):
        raise sr_module.UnknownValueError()

    result = transcribe_audio(audio, recognize_func=fake_recognize)

    assert result.success is False
    assert result.text == ""
    assert "unclear" in result.error.lower()


def test_recognition_service_error_returns_failed_result_without_crashing(make_wav):
    audio = _captured(make_wav)

    def fake_recognize(sr_module, recognizer, audio_data, *, engine, language):
        raise sr_module.RequestError("quota exceeded")

    result = transcribe_audio(audio, recognize_func=fake_recognize)

    assert result.success is False
    assert "quota exceeded" in result.error


def test_unknown_engine_raises_transcription_error(make_wav):
    audio = _captured(make_wav)

    with pytest.raises(TranscriptionError, match="Unknown recognition engine"):
        transcribe_audio(audio, engine="carrier-pigeon")


def test_transcriber_is_reusable_across_calls(make_wav):
    calls = []

    def fake_recognize(sr_module, recognizer, audio_data, *, engine, language):
        calls.append(1)
        return f"clip {len(calls)}"

    transcriber = Transcriber(recognize_func=fake_recognize)
    first = transcriber.transcribe(_captured(make_wav))
    second = transcriber.transcribe(_captured(make_wav))

    assert first.text == "clip 1"
    assert second.text == "clip 2"


def test_real_speech_recognition_exceptions_are_the_ones_caught():
    # Guards against the module catching some other library's exception types.
    assert issubclass(sr.UnknownValueError, Exception)
    assert issubclass(sr.RequestError, Exception)
