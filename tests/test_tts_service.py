from __future__ import annotations

import io
import json
import wave

import httpx
import numpy as np
import pytest

from marvin_companion.config.settings import AppSettings
from marvin_companion.services.http_client import GroqAPIClient
from marvin_companion.services.tts_service import TTSService
from marvin_companion.utils.audio import load_wav_bytes


def build_settings(temp_dir: str) -> AppSettings:
    return AppSettings(
        groq={"api_key": "test-key"},
        features={"voice_enabled": False},
        robot={"mock_mode": True},
        audio={"temp_dir": temp_dir},
    )


def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    samples = (0.1 * np.sin(np.linspace(0.0, 200.0, 480))).astype(np.float32)
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes((samples * 32767).astype(np.int16).tobytes())
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_tts_chunking_and_cache(tmp_path) -> None:
    calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        calls.append(payload["input"])
        return httpx.Response(200, content=wav_bytes(), headers={"content-type": "audio/wav"})

    transport = httpx.MockTransport(handler)
    settings = build_settings(str(tmp_path))
    client = httpx.AsyncClient(transport=transport, base_url="https://api.groq.com/openai/v1")
    service = TTSService(GroqAPIClient(settings, client=client), settings)
    long_text = "Hello there. " * 30

    first = await service.synthesize(long_text)
    second = await service.synthesize(long_text)

    assert len(calls) >= 2
    assert first == second
    samples, sample_rate = load_wav_bytes(first)
    assert sample_rate == 24_000
    assert samples.size > 0
    await client.aclose()

