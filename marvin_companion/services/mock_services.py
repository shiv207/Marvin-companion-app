"""Demo-mode mock services."""

from __future__ import annotations

import asyncio
import io
import json
import wave
from dataclasses import dataclass

import numpy as np

from marvin_companion.core.interfaces import LLMProvider, RobotController, STTProvider, TTSProvider


class MockLLMService(LLMProvider):
    """Deterministic mock LLM for demo mode."""

    async def generate(self, messages: list[dict], stream: bool = False) -> str:
        user_text = messages[-1]["content"].lower()
        if "dance" in user_text:
            payload = {"command": "dance", "face": None, "response": "okay, I will dance.", "reasoning": "User asked for a dance."}
        elif "hello" in user_text or "hi" in user_text:
            payload = {"command": "wave", "face": "happy", "response": "Hi friend!", "reasoning": "Greeting detected."}
        else:
            payload = {"command": None, "face": "happy", "response": "I am here to help.", "reasoning": "Conversational reply."}
        if stream:
            return "".join([chunk async for chunk in self.generate_stream(messages)])
        return json.dumps(payload)

    async def generate_stream(self, messages: list[dict]):
        response = await self.generate(messages, stream=False)
        for chunk in [response[i : i + 24] for i in range(0, len(response), 24)]:
            await asyncio.sleep(0)
            yield chunk


class MockSTTService(STTProvider):
    """Mock STT that returns a canned transcript."""

    async def transcribe(self, audio_path: str) -> str:
        del audio_path
        return "Hello Marvin"


class MockTTSService(TTSProvider):
    """Mock TTS that returns a short sine-wave WAV."""

    async def synthesize(self, text: str) -> bytes:
        sample_rate = 24_000
        duration = max(0.25, min(2.0, len(text) * 0.02))
        times = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
        samples = (0.2 * np.sin(2 * np.pi * 440 * times)).astype(np.float32)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes((samples * 32767).astype(np.int16).tobytes())
        return buffer.getvalue()


class MockRobotController(RobotController):
    """Mock robot controller."""

    async def send_command(self, command: str, face: str | None = None) -> dict:
        return {"status": "success", "command": command, "face": face, "mock": True}

    async def get_status(self) -> dict:
        return {"currentCommand": "idle", "currentFace": "happy", "networkConnected": True, "mock": True}
