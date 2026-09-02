"""Tests for the Phase 1 CLI entrypoint."""

from __future__ import annotations

import pytest

from voice_rag_chatbot import cli
from voice_rag_chatbot.audio_input import MicrophoneUnavailableError


def test_cli_file_mode_reports_capture(make_wav, capsys):
    path = make_wav(seconds=1.0, sample_rate=16000)

    rc = cli.main(["--source", "file", "--file", str(path)])

    out = capsys.readouterr().out
    assert rc == 0
    assert "captured 1.00s" in out
    assert "sample_rate=16000 Hz" in out
    assert "16-bit" in out


def test_cli_file_mode_can_save(make_wav, tmp_path, capsys):
    path = make_wav(seconds=0.5)
    out_path = tmp_path / "saved.wav"

    rc = cli.main(["--source", "file", "--file", str(path), "--save", str(out_path)])

    assert rc == 0
    assert out_path.exists()
    assert "saved ->" in capsys.readouterr().out


def test_cli_missing_file_returns_2(tmp_path, capsys):
    rc = cli.main(["--source", "file", "--file", str(tmp_path / "ghost.wav")])

    assert rc == 2
    assert "error:" in capsys.readouterr().err


def test_cli_until_silence_rejected_for_file_source(make_wav, capsys):
    rc = cli.main(["--source", "file", "--file", str(make_wav()), "--until-silence"])

    assert rc == 2
    assert "--until-silence only applies" in capsys.readouterr().err


def test_cli_microphone_unavailable_returns_2_with_hint(monkeypatch, capsys):
    def _boom(**_kwargs):
        raise MicrophoneUnavailableError("No usable microphone: no backend.")

    monkeypatch.setattr(cli, "capture_audio", _boom)

    rc = cli.main(["--source", "microphone", "--duration", "1"])

    err = capsys.readouterr().err
    assert rc == 2
    assert "error:" in err
    assert "--source file" in err
