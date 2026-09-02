"""Shared test helpers.

`src/` is a plain directory (no editable install in CI), so put it on the
path here and expose a small WAV-file factory used across the audio tests.
"""

from __future__ import annotations

import math
import struct
import sys
import wave
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _write_sine_wav(
    path: Path,
    *,
    seconds: float,
    sample_rate: int,
    channels: int,
    freq_hz: float = 220.0,
    amplitude: int = 8000,
) -> Path:
    n_frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for i in range(n_frames):
            sample = int(amplitude * math.sin(2 * math.pi * freq_hz * i / sample_rate))
            wav.writeframes(struct.pack("<h", sample) * channels)
    return path


@pytest.fixture
def make_wav(tmp_path: Path):
    """Return a factory that writes a sine-wave WAV and returns its path."""

    counter = {"n": 0}

    def _factory(
        *,
        seconds: float = 1.0,
        sample_rate: int = 16000,
        channels: int = 1,
        name: str | None = None,
    ) -> Path:
        counter["n"] += 1
        filename = name or f"clip_{counter['n']}.wav"
        return _write_sine_wav(
            tmp_path / filename,
            seconds=seconds,
            sample_rate=sample_rate,
            channels=channels,
        )

    return _factory
