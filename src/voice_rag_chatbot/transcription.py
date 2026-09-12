"""Speech-to-text transcription for captured audio via ``SpeechRecognition``.

The recognizer call is injected through :attr:`Transcriber.recognize_func` so
tests can transcribe a sample audio file deterministically without a live
network call to a cloud STT API (the default ``google`` engine needs one).
Recognition failure (unclear/silent audio) and service errors both come back
as a failed :class:`TranscriptionResult` rather than an uncaught exception --
only a missing dependency or an unknown engine name raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .audio_input import CapturedAudio

__all__ = [
    "TranscriptionError",
    "TranscriptionResult",
    "Transcriber",
    "transcribe_audio",
]


class TranscriptionError(RuntimeError):
    """Raised for setup problems: missing dependency or unknown engine name."""


@dataclass(frozen=True)
class TranscriptionResult:
    """Outcome of one transcription attempt."""

    text: str
    success: bool
    engine: str
    error: str | None = None


def _import_speech_recognition():
    try:
        import speech_recognition as sr  # noqa: PLC0415 (lazy: optional heavy dep)
    except ImportError as exc:  # pragma: no cover - depends on install
        raise TranscriptionError(
            "The 'SpeechRecognition' package is not installed; run "
            "`pip install -r requirements.txt`."
        ) from exc
    return sr


def _default_recognize(sr_module, recognizer, audio_data, *, engine: str, language: str) -> str:
    recognize = getattr(recognizer, f"recognize_{engine}", None)
    if recognize is None:
        raise TranscriptionError(
            f"Unknown recognition engine: {engine!r} (no recognizer.recognize_{engine})"
        )
    return recognize(audio_data, language=language)


@dataclass
class Transcriber:
    """Transcribes :class:`CapturedAudio` using a ``SpeechRecognition`` engine.

    ``recognize_func``, when set, replaces the call to the real recognizer
    engine -- used by tests and by fully offline engines. It is called as
    ``recognize_func(sr_module, recognizer, audio_data, engine=..., language=...)``
    and must either return the transcribed text or raise
    ``speech_recognition.UnknownValueError`` / ``RequestError``, exactly like
    the real ``recognizer.recognize_*`` methods.
    """

    engine: str = "google"
    language: str = "en-US"
    recognize_func: Callable[..., str] | None = None

    def transcribe(self, audio: CapturedAudio) -> TranscriptionResult:
        sr = _import_speech_recognition()
        audio_data = sr.AudioData(audio.raw_data, audio.sample_rate, audio.sample_width)
        recognizer = sr.Recognizer()
        recognize = self.recognize_func or _default_recognize

        try:
            text = recognize(
                sr, recognizer, audio_data, engine=self.engine, language=self.language
            )
        except sr.UnknownValueError:
            return TranscriptionResult(
                text="", success=False, engine=self.engine, error="unclear audio: could not transcribe"
            )
        except sr.RequestError as exc:
            return TranscriptionResult(
                text="", success=False, engine=self.engine, error=f"recognition service error: {exc}"
            )

        return TranscriptionResult(text=text, success=True, engine=self.engine)


def transcribe_audio(
    audio: CapturedAudio,
    *,
    engine: str = "google",
    language: str = "en-US",
    recognize_func: Callable[..., str] | None = None,
) -> TranscriptionResult:
    """Convenience wrapper around :class:`Transcriber` for one-off calls."""
    return Transcriber(
        engine=engine, language=language, recognize_func=recognize_func
    ).transcribe(audio)
