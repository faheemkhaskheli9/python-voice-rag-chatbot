"""Generate a tiny synthetic WAV so the file-based capture path is runnable
without a microphone (used by the VS Code 'file' launch config and the docs).

    python scripts/make_example_audio.py
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "examples" / "hello.wav"
SAMPLE_RATE = 16000
SECONDS = 1.5
FREQ_HZ = 196.0  # low G, an arbitrary public tone -- not real speech


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUT), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        for i in range(int(SECONDS * SAMPLE_RATE)):
            # simple fade so it isn't a harsh square-ish click at the ends
            envelope = min(1.0, i / 2000, (SECONDS * SAMPLE_RATE - i) / 2000)
            sample = int(8000 * envelope * math.sin(2 * math.pi * FREQ_HZ * i / SAMPLE_RATE))
            wav.writeframes(struct.pack("<h", sample))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
