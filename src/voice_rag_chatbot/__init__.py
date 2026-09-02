"""Python Voice RAG Chatbot.

Phase 1 focuses on the plain STT -> RAG -> LLM -> TTS loop. This package
currently exposes the microphone-input stage; later phases add recognition,
retrieval, generation, and speech synthesis behind the same style of
swappable interfaces.
"""

from .audio_input import (
    AudioSource,
    CapturedAudio,
    FileAudioSource,
    MicrophoneSource,
    MicrophoneUnavailableError,
    capture_audio,
)

__all__ = [
    "AudioSource",
    "CapturedAudio",
    "FileAudioSource",
    "MicrophoneSource",
    "MicrophoneUnavailableError",
    "capture_audio",
]
