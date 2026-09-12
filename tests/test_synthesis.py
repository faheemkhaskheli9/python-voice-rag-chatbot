from __future__ import annotations

from pathlib import Path

import pytest

from voice_rag_chatbot.synthesis import (
    SynthesisError,
    Synthesizer,
    play_audio,
    synthesize_speech,
)


def _fake_synth_writes_file(engine, text, output_path, *, language):
    """Stand-in for a real TTS backend: writes a placeholder audio file."""
    Path(output_path).write_bytes(f"fake-audio:{engine}:{language}:{text}".encode())


def test_synthesize_produces_playable_audio_file(tmp_path):
    out = tmp_path / "out.mp3"
    result = synthesize_speech(
        "hello there", str(out), engine="pyttsx3", synth_func=_fake_synth_writes_file
    )
    assert result.success is True
    assert result.audio_path == str(out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_engine_is_configurable_pyttsx3_vs_gtts(tmp_path):
    for engine in ("pyttsx3", "gtts"):
        out = tmp_path / f"out_{engine}.mp3"
        result = synthesize_speech(
            "configurable engine", str(out), engine=engine, synth_func=_fake_synth_writes_file
        )
        assert result.success is True
        assert result.engine == engine


def test_unknown_engine_raises_without_injected_synth_func(tmp_path):
    with pytest.raises(SynthesisError):
        synthesize_speech("hi", str(tmp_path / "out.mp3"), engine="not-a-real-engine")


def test_empty_text_is_a_failed_result_not_a_crash(tmp_path):
    result = synthesize_speech(
        "   ", str(tmp_path / "out.mp3"), synth_func=_fake_synth_writes_file
    )
    assert result.success is False
    assert result.audio_path is None
    assert "empty" in result.error


def test_backend_exception_becomes_a_failed_result(tmp_path):
    def _boom(engine, text, output_path, *, language):
        raise RuntimeError("engine exploded")

    result = synthesize_speech("hi", str(tmp_path / "out.mp3"), synth_func=_boom)
    assert result.success is False
    assert "engine exploded" in result.error


def test_backend_that_reports_success_but_writes_nothing_is_a_failed_result(tmp_path):
    def _no_op(engine, text, output_path, *, language):
        pass  # pretends to succeed but never writes the file

    result = synthesize_speech("hi", str(tmp_path / "missing.mp3"), synth_func=_no_op)
    assert result.success is False
    assert "no audio file" in result.error


def test_synthesizer_class_matches_module_function(tmp_path):
    out = tmp_path / "out.mp3"
    synth = Synthesizer(engine="pyttsx3", synth_func=_fake_synth_writes_file)
    result = synth.synthesize("hello", str(out))
    assert result.success is True
    assert out.exists()


def test_play_audio_returns_false_for_missing_file(tmp_path):
    assert play_audio(str(tmp_path / "does-not-exist.mp3")) is False


def test_play_audio_uses_injected_player_and_reports_success(tmp_path):
    out = tmp_path / "out.mp3"
    out.write_bytes(b"fake-audio")
    calls = []
    assert play_audio(str(out), player_func=calls.append) is True
    assert calls == [str(out)]


def test_play_audio_failure_in_player_is_reported_not_raised(tmp_path):
    out = tmp_path / "out.mp3"
    out.write_bytes(b"fake-audio")

    def _boom(path):
        raise RuntimeError("no audio device")

    assert play_audio(str(out), player_func=_boom) is False
