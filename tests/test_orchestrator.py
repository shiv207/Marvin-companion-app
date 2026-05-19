from __future__ import annotations

import asyncio
import io
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from marvin_companion.config.settings import AppSettings
from marvin_companion.core.orchestrator import MarvinOrchestrator
from marvin_companion.services.websocket_manager import WebSocketManager


class StubLLM:
    async def generate(self, messages: list[dict], stream: bool = False) -> str:
        del messages, stream
        return json.dumps(
            {"command": "wave", "face": "happy", "response": "Hi friend!", "reasoning": "Greeting"}
        )

    async def generate_stream(self, messages: list[dict]):
        del messages
        yield "unused"


class SlowLLM(StubLLM):
    async def generate(self, messages: list[dict], stream: bool = False) -> str:
        del messages, stream
        await asyncio.sleep(10)
        return "{}"


class StubSTT:
    async def transcribe(self, audio_path: str) -> str:
        del audio_path
        return "hello marvin"


class StubTTS:
    async def synthesize(self, text: str) -> bytes:
        del text
        buffer = io.BytesIO()
        samples = np.zeros(240, dtype=np.float32)
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24_000)
            wav_file.writeframes((samples * 32767).astype(np.int16).tobytes())
        return buffer.getvalue()


class StubRobot:
    def __init__(self) -> None:
        self.commands: list[tuple[str, str | None]] = []

    async def send_command(self, command: str, face: str | None = None) -> dict:
        self.commands.append((command, face))
        return {"status": "ok"}

    async def get_status(self) -> dict:
        return {"currentCommand": "idle"}


class StubAudio:
    async def record_phrase(self, timeout_seconds: float | None = None) -> Path:
        del timeout_seconds
        return Path("unused.wav")

    async def cleanup(self, audio_path: Path) -> None:
        del audio_path

    async def play_audio(self, wav_audio: bytes) -> None:
        assert wav_audio


def build_settings() -> AppSettings:
    return AppSettings(
        groq={"api_key": "test-key"},
        features={"voice_enabled": False},
        robot={"mock_mode": True},
    )


@pytest.mark.asyncio
async def test_orchestrator_happy_path() -> None:
    robot = StubRobot()
    orchestrator = MarvinOrchestrator(
        settings=build_settings(),
        llm=StubLLM(),
        stt=StubSTT(),
        tts=StubTTS(),
        robot=robot,
        audio_input=StubAudio(),
        playback=StubAudio(),
        events=WebSocketManager(),
    )

    result = await orchestrator.handle_text("hello")

    assert result.assistant_text == "Hi friend!"
    assert robot.commands == [("wave", "happy")]


@pytest.mark.asyncio
async def test_orchestrator_cancellation() -> None:
    orchestrator = MarvinOrchestrator(
        settings=build_settings(),
        llm=SlowLLM(),
        stt=StubSTT(),
        tts=StubTTS(),
        robot=StubRobot(),
        audio_input=StubAudio(),
        playback=StubAudio(),
        events=WebSocketManager(),
    )

    task = asyncio.create_task(orchestrator.handle_text("hello"))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
