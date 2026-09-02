"""Command-line entrypoint for the voice RAG chatbot.

Phase 1 exposes only the microphone-input stage: capture audio and report
what was captured (optionally saving it to a WAV). Later phases extend this
same CLI with transcription, retrieval, and spoken responses.

    python -m voice_rag_chatbot --source microphone --duration 5
    python -m voice_rag_chatbot --source microphone --until-silence
    python -m voice_rag_chatbot --source file --file examples/hello.wav --save out.wav
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .audio_input import MicrophoneUnavailableError, capture_audio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice_rag_chatbot",
        description="Capture microphone (or file) audio for the voice RAG chatbot pipeline.",
    )
    parser.add_argument(
        "--source",
        choices=["microphone", "file"],
        default="microphone",
        help="Where to read audio from (default: microphone).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Seconds to capture (default: 5). Ignored when --until-silence is set.",
    )
    parser.add_argument(
        "--until-silence",
        action="store_true",
        help="Microphone only: stop when the speaker pauses instead of a fixed duration.",
    )
    parser.add_argument(
        "--file",
        dest="file_path",
        help="WAV file to read when --source file.",
    )
    parser.add_argument(
        "--save",
        dest="save_path",
        help="Write the captured audio to this WAV path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.until_silence and args.source == "file":
        print("error: --until-silence only applies to --source microphone", file=sys.stderr)
        return 2

    duration: float | None
    duration = None if args.until_silence else args.duration

    try:
        audio = capture_audio(
            source=args.source,
            duration_seconds=duration,
            file_path=args.file_path,
        )
    except MicrophoneUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "hint: run without a microphone via `--source file --file <path.wav>`.",
            file=sys.stderr,
        )
        return 2
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"captured {audio.duration_seconds:.2f}s from {audio.source}")
    print(
        f"  sample_rate={audio.sample_rate} Hz  "
        f"sample_width={audio.sample_width * 8}-bit  "
        f"bytes={len(audio.raw_data)}"
    )

    if args.save_path:
        written = audio.write_wav(args.save_path)
        print(f"  saved -> {written}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
