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
from .synthesis import SynthesisError, play_audio, synthesize_speech
from .transcription import TranscriptionError, transcribe_audio


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
    parser.add_argument(
        "--transcribe",
        action="store_true",
        help="Also run speech-to-text on the captured audio and print the text "
        "(calls a network STT engine by default; requires 'google' reachability).",
    )
    parser.add_argument(
        "--speak",
        dest="speak_text",
        help="Synthesize this text to speech and play it back "
        "(closes the mic-to-text/text-to-speech loop for Phase 1).",
    )
    parser.add_argument(
        "--tts-engine",
        choices=["pyttsx3", "gtts"],
        default="pyttsx3",
        help="TTS engine for --speak: offline pyttsx3 (default) or cloud gTTS.",
    )
    parser.add_argument(
        "--speak-out",
        dest="speak_out_path",
        default="response.mp3",
        help="Where to write the synthesized audio for --speak (default: response.mp3).",
    )
    parser.add_argument(
        "--no-playback",
        action="store_true",
        help="With --speak, only generate the audio file; skip playback "
        "(useful on machines with no speaker/audio driver).",
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

    if args.transcribe:
        try:
            result = transcribe_audio(audio)
        except TranscriptionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if result.success:
            print(f"  transcript: {result.text!r}")
        else:
            print(f"  transcript failed: {result.error}", file=sys.stderr)

    if args.speak_text:
        try:
            synth_result = synthesize_speech(
                args.speak_text, args.speak_out_path, engine=args.tts_engine
            )
        except SynthesisError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        if not synth_result.success:
            print(f"error: {synth_result.error}", file=sys.stderr)
            return 2

        print(f"  synthesized ({synth_result.engine}) -> {synth_result.audio_path}")
        if not args.no_playback:
            if not play_audio(synth_result.audio_path):
                print(
                    "  warning: could not play audio (no audio backend/device); "
                    "file was still written",
                    file=sys.stderr,
                )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
