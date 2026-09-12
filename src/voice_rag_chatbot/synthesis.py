"""Text-to-speech synthesis for the voice RAG chatbot.

Two interchangeable engines, selected by name so the CLI can switch between
them without code changes:

* ``pyttsx3`` -- fully offline, works without network access.
* ``gtts`` -- Google's cloud TTS; needs network access but broader/naturalier
  voices.

The actual synthesis call is injected through :attr:`Synthesizer.synth_func`
(same pattern as :mod:`voice_rag_chatbot.transcription`'s ``recognize_func``)
so tests can generate deterministic "audio" without a real TTS engine, an
audio driver, or a network call. Playback is a separate, optional step
(:func:`play_audio`) so text -> audio generation can be smoke-tested on
machines with no speaker.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

__all__ = [
    "SynthesisError",
    "SynthesisResult",
    "Synthesizer",
    "synthesize_speech",
    "play_audio",
]


class SynthesisError(RuntimeError):
    """Raised for setup problems: missing dependency or unknown engine name."""


@dataclass(frozen=True)
class SynthesisResult:
    """Outcome of one synthesis attempt."""

    audio_path: str | None
    success: bool
    engine: str
    error: str | None = None


def _import_pyttsx3():
    try:
        import pyttsx3  # noqa: PLC0415 (lazy: optional heavy dep)
    except ImportError as exc:  # pragma: no cover - depends on install
        raise SynthesisError(
            "The 'pyttsx3' package is not installed; run `pip install -r requirements.txt`."
        ) from exc
    return pyttsx3


def _import_gtts():
    try:
        from gtts import gTTS  # noqa: PLC0415 (lazy: optional heavy dep)
    except ImportError as exc:  # pragma: no cover - depends on install
        raise SynthesisError(
            "The 'gTTS' package is not installed; run `pip install -r requirements.txt`."
        ) from exc
    return gTTS


def _default_synth(engine: str, text: str, output_path: str, *, language: str) -> None:
    """Write ``text`` as speech audio to ``output_path`` using ``engine``."""
    if engine == "pyttsx3":
        pyttsx3 = _import_pyttsx3()
        tts_engine = pyttsx3.init()
        tts_engine.save_to_file(text, output_path)
        tts_engine.runAndWait()
    elif engine == "gtts":
        gTTS = _import_gtts()
        gTTS(text=text, lang=language).save(output_path)
    else:
        raise SynthesisError(
            f"Unknown TTS engine: {engine!r} (expected 'pyttsx3' or 'gtts')"
        )


@dataclass
class Synthesizer:
    """Converts text to a playable audio file using a configurable TTS engine.

    ``synth_func``, when set, replaces the call to the real TTS backend --
    used by tests. It is called as
    ``synth_func(engine, text, output_path, language=...)`` and must write
    the audio file itself (or raise on failure); its return value is ignored.
    """

    engine: str = "pyttsx3"
    language: str = "en"
    synth_func: Callable[..., None] | None = None

    def synthesize(self, text: str, output_path: str) -> SynthesisResult:
        if not text.strip():
            return SynthesisResult(
                audio_path=None, success=False, engine=self.engine,
                error="cannot synthesize empty text",
            )

        synth = self.synth_func or _default_synth
        try:
            synth(self.engine, text, output_path, language=self.language)
        except SynthesisError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as a failed result, not a crash
            return SynthesisResult(
                audio_path=None, success=False, engine=self.engine,
                error=f"synthesis failed: {exc}",
            )

        if not Path(output_path).exists() or Path(output_path).stat().st_size == 0:
            return SynthesisResult(
                audio_path=None, success=False, engine=self.engine,
                error="TTS engine reported success but produced no audio file",
            )
        return SynthesisResult(audio_path=output_path, success=True, engine=self.engine)


def synthesize_speech(
    text: str,
    output_path: str,
    *,
    engine: str = "pyttsx3",
    language: str = "en",
    synth_func: Callable[..., None] | None = None,
) -> SynthesisResult:
    """Convenience wrapper around :class:`Synthesizer` for one-off calls."""
    return Synthesizer(engine=engine, language=language, synth_func=synth_func).synthesize(
        text, output_path
    )


def play_audio(path: str, *, player_func: Callable[[str], None] | None = None) -> bool:
    """Play back the audio file at ``path``. Returns ``True`` on success.

    ``player_func``, when set, replaces the real playback call (used by
    tests and by headless environments with no audio device). The default
    playback uses ``playsound`` lazily, so importing this module never
    requires an audio backend to be installed.
    """
    if not Path(path).exists():
        return False

    play = player_func
    if play is None:
        try:
            from playsound import playsound as play  # noqa: PLC0415
        except ImportError:
            return False

    try:
        play(path)
    except Exception:  # noqa: BLE001 - playback failure should not crash the caller
        return False
    return True
