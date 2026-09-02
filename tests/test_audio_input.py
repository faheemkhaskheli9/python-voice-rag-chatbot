"""Tests for the microphone-input capture stage (issue #1)."""

from __future__ import annotations

import wave

import pytest

from voice_rag_chatbot.audio_input import (
    CapturedAudio,
    FileAudioSource,
    MicrophoneSource,
    MicrophoneUnavailableError,
    capture_audio,
)

try:  # microphone tests are only meaningful when there is no real backend
    import pyaudio  # noqa: F401

    HAS_PYAUDIO = True
except Exception:  # pragma: no cover - environment dependent
    HAS_PYAUDIO = False


def test_file_source_reads_mono_wav(make_wav):
    path = make_wav(seconds=1.0, sample_rate=16000, channels=1)

    audio = FileAudioSource(path).capture()

    assert audio.sample_rate == 16000
    assert audio.sample_width == 2
    assert len(audio.raw_data) == 16000 * 2
    assert audio.duration_seconds == pytest.approx(1.0, abs=0.01)
    assert audio.source == f"file:{path.name}"


def test_file_source_respects_max_duration(make_wav):
    path = make_wav(seconds=2.0, sample_rate=16000, channels=1)

    audio = FileAudioSource(path, max_duration_seconds=0.5).capture()

    assert audio.duration_seconds == pytest.approx(0.5, abs=0.01)
    assert len(audio.raw_data) == int(0.5 * 16000) * 2


def test_file_source_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        FileAudioSource(tmp_path / "nope.wav").capture()


def test_file_source_rejects_stereo_with_clear_message(make_wav):
    path = make_wav(seconds=0.5, channels=2)

    with pytest.raises(ValueError, match="mono"):
        FileAudioSource(path).capture()


def test_file_source_rejects_non_wav(tmp_path):
    junk = tmp_path / "not_audio.wav"
    junk.write_bytes(b"this is definitely not a RIFF header")

    with pytest.raises(ValueError, match="WAV"):
        FileAudioSource(junk).capture()


def test_captured_audio_write_wav_is_atomic_and_roundtrips(tmp_path, make_wav):
    original = FileAudioSource(make_wav(seconds=1.0)).capture()
    out = tmp_path / "nested" / "out.wav"

    written = original.write_wav(out)

    assert written == out
    assert not any(p.name.startswith(".out.wav") for p in out.parent.iterdir())
    with wave.open(str(out), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == original.sample_rate
        assert wav.readframes(wav.getnframes()) == original.raw_data


def test_duration_seconds_handles_degenerate_header():
    assert CapturedAudio(b"", 0, 0, "x").duration_seconds == 0.0


def test_capture_audio_unknown_source():
    with pytest.raises(ValueError, match="Unknown audio source"):
        capture_audio(source="carrier-pigeon")


def test_capture_audio_file_requires_path():
    with pytest.raises(ValueError, match="file_path is required"):
        capture_audio(source="file")


def test_capture_audio_file_source_end_to_end(make_wav):
    path = make_wav(seconds=1.0)
    audio = capture_audio(source="file", duration_seconds=None, file_path=path)
    assert audio.duration_seconds == pytest.approx(1.0, abs=0.01)


def test_microphone_source_rejects_nonpositive_duration():
    with pytest.raises(ValueError, match="duration_seconds"):
        MicrophoneSource(duration_seconds=0).capture()


@pytest.mark.skipif(HAS_PYAUDIO, reason="a real PyAudio backend is installed")
def test_microphone_unavailable_without_backend_raises_clear_error():
    with pytest.raises(MicrophoneUnavailableError) as excinfo:
        MicrophoneSource(duration_seconds=1.0).capture()

    assert "microphone" in str(excinfo.value).lower()


@pytest.mark.skipif(HAS_PYAUDIO, reason="a real PyAudio backend is installed")
def test_capture_audio_microphone_path_maps_to_unavailable_error():
    with pytest.raises(MicrophoneUnavailableError):
        capture_audio(source="microphone", duration_seconds=1.0)
