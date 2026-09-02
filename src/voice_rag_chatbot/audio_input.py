"""Microphone input capture for the voice RAG chatbot.

The STT stage needs a real audio source. This module provides two
interchangeable sources behind a small :class:`AudioSource` protocol:

* :class:`MicrophoneSource` -- live capture through the ``SpeechRecognition``
  library's microphone interface, either for a fixed duration or until the
  speaker pauses (silence detection).
* :class:`FileAudioSource` -- reads a WAV file with the standard library only.
  Used by tests, CI, and headless containers where no input device exists.

Environments without a usable microphone (no PyAudio backend, no input
device -- e.g. CI) raise :class:`MicrophoneUnavailableError` with an
actionable message instead of a cryptic backend error.
"""

from __future__ import annotations

import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

__all__ = [
    "AudioSource",
    "CapturedAudio",
    "FileAudioSource",
    "MicrophoneSource",
    "MicrophoneUnavailableError",
    "capture_audio",
]


class MicrophoneUnavailableError(RuntimeError):
    """Raised when live microphone capture is requested but impossible.

    Covers a missing ``SpeechRecognition``/PyAudio backend, no input device
    (CI, headless containers), and "asked to listen but heard nothing".
    """


@dataclass(frozen=True)
class CapturedAudio:
    """A finished, in-memory recording.

    ``raw_data`` is little-endian signed PCM, ``sample_width`` bytes per
    sample, single channel.
    """

    raw_data: bytes
    sample_rate: int
    sample_width: int
    source: str

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0 or self.sample_width <= 0:
            return 0.0
        return (len(self.raw_data) / self.sample_width) / self.sample_rate

    def write_wav(self, path: str | os.PathLike[str]) -> Path:
        """Write the recording to ``path`` as a mono WAV file.

        The file is written to a sibling temp path and atomically renamed
        onto ``path`` so an interrupted write never leaves a truncated WAV.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            with wave.open(str(tmp), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(self.sample_width)
                wav.setframerate(self.sample_rate)
                wav.writeframes(self.raw_data)
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return target


@runtime_checkable
class AudioSource(Protocol):
    """Anything that can produce one :class:`CapturedAudio` recording."""

    def capture(self) -> CapturedAudio: ...


def _import_speech_recognition():
    try:
        import speech_recognition as sr  # noqa: PLC0415 (lazy: optional heavy dep)
    except ImportError as exc:  # pragma: no cover - depends on install
        raise MicrophoneUnavailableError(
            "The 'SpeechRecognition' package is not installed; run "
            "`pip install -r requirements.txt` or use a file source instead."
        ) from exc
    return sr


@dataclass
class MicrophoneSource:
    """Live capture via ``SpeechRecognition``'s microphone interface.

    With ``duration_seconds`` set, records exactly that many seconds. With
    ``duration_seconds=None`` it listens until the speaker pauses for
    ``pause_threshold`` seconds (silence detection), giving up after
    ``timeout_seconds`` of initial silence.
    """

    duration_seconds: float | None = 5.0
    device_index: int | None = None
    timeout_seconds: float = 10.0
    pause_threshold: float = 0.8
    calibrate_ambient_noise: bool = True

    def capture(self) -> CapturedAudio:
        if self.duration_seconds is not None and self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be > 0 (or None to listen until silence)")

        sr = _import_speech_recognition()
        try:
            microphone = sr.Microphone(device_index=self.device_index)
        except (AttributeError, OSError) as exc:
            # AttributeError: PyAudio backend missing. OSError: no input device.
            raise MicrophoneUnavailableError(
                "No usable microphone: " + str(exc).strip() + ". "
                "Install PyAudio and connect an input device, or use a file source."
            ) from exc

        recognizer = sr.Recognizer()
        recognizer.pause_threshold = self.pause_threshold
        try:
            with microphone as source:
                if self.calibrate_ambient_noise:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                if self.duration_seconds is None:
                    audio = recognizer.listen(
                        source,
                        timeout=self.timeout_seconds,
                        phrase_time_limit=None,
                    )
                else:
                    audio = recognizer.record(source, duration=self.duration_seconds)
        except sr.WaitTimeoutError as exc:
            raise MicrophoneUnavailableError(
                f"No speech detected within {self.timeout_seconds:.0f}s of starting to listen."
            ) from exc
        except OSError as exc:
            raise MicrophoneUnavailableError(
                f"Microphone read failed: {str(exc).strip()}."
            ) from exc

        return CapturedAudio(
            raw_data=audio.get_raw_data(),
            sample_rate=audio.sample_rate,
            sample_width=audio.sample_width,
            source="microphone",
        )


@dataclass
class FileAudioSource:
    """Read a mono WAV file as if it were captured from a microphone.

    Standard-library only -- no PyAudio, no ``SpeechRecognition``.
    """

    path: str | os.PathLike[str]
    max_duration_seconds: float | None = None

    def capture(self) -> CapturedAudio:
        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        try:
            with wave.open(str(path), "rb") as wav:
                n_channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                frame_rate = wav.getframerate()
                n_frames = wav.getnframes()
                if self.max_duration_seconds is not None:
                    if self.max_duration_seconds <= 0:
                        raise ValueError("max_duration_seconds must be > 0 when set")
                    n_frames = min(n_frames, int(self.max_duration_seconds * frame_rate))
                raw = wav.readframes(n_frames)
        except wave.Error as exc:
            raise ValueError(f"Not a readable WAV file ({path}): {exc}") from exc

        if n_channels != 1:
            raise ValueError(
                f"Expected a mono WAV file, got {n_channels} channels ({path}). "
                "Down-mix to mono first (e.g. `ffmpeg -i in.wav -ac 1 out.wav`)."
            )

        return CapturedAudio(
            raw_data=raw,
            sample_rate=frame_rate,
            sample_width=sample_width,
            source=f"file:{path.name}",
        )


def capture_audio(
    source: str = "microphone",
    *,
    duration_seconds: float | None = 5.0,
    file_path: str | os.PathLike[str] | None = None,
    **microphone_kwargs: object,
) -> CapturedAudio:
    """Capture one recording from ``"microphone"`` or ``"file"``.

    ``duration_seconds=None`` with the microphone source listens until the
    speaker pauses; with the file source it means "read the whole file".
    """
    if source == "microphone":
        return MicrophoneSource(
            duration_seconds=duration_seconds,
            **microphone_kwargs,  # type: ignore[arg-type]
        ).capture()
    if source == "file":
        if file_path is None:
            raise ValueError("file_path is required when source='file'")
        return FileAudioSource(file_path, max_duration_seconds=duration_seconds).capture()
    raise ValueError(f"Unknown audio source: {source!r} (expected 'microphone' or 'file')")
