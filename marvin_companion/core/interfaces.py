"""Provider and pipeline interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator, Protocol

from marvin_companion.core.events import BaseEvent


class LLMProvider(Protocol):
    async def generate(self, messages: list[dict], stream: bool = False) -> str:
        """Generate a model completion."""

    async def generate_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Stream model deltas."""


class STTProvider(Protocol):
    async def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file."""


class TTSProvider(Protocol):
    async def synthesize(self, text: str) -> bytes:
        """Synthesize text into WAV audio bytes."""


class RobotController(Protocol):
    async def send_command(self, command: str, face: str | None = None) -> dict:
        """Send a robot command."""

    async def get_status(self) -> dict:
        """Read robot status."""


class EventPublisher(Protocol):
    async def publish(self, event: BaseEvent) -> None:
        """Publish an event."""


class PlaybackSink(Protocol):
    async def play_audio(self, wav_audio: bytes) -> None:
        """Play a WAV audio buffer."""


class AudioInput(Protocol):
    async def record_phrase(self, timeout_seconds: float | None = None) -> Path:
        """Capture a phrase and return a path to a normalized WAV file."""

    async def cleanup(self, audio_path: Path) -> None:
        """Clean up a captured audio artifact."""

