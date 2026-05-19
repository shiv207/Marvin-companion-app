from __future__ import annotations

import io
from pathlib import Path

import pytest

from marvin_companion.config.settings import AppSettings
from marvin_companion.core.exceptions import UpstreamAPIError
from marvin_companion.services.stt_service import STTService


class StubTransport:
    def __init__(self) -> None:
        self.calls = 0

    async def post_multipart(self, path: str, *, data: dict, files: dict):
        self.calls += 1
        assert path == "/audio/transcriptions"
        assert "file" in files
        if self.calls == 1:
            raise UpstreamAPIError("primary failed", status_code=500)

        class Response:
            data = {"text": "hello marvin", "language": "en"}

        return Response()


def build_settings() -> AppSettings:
    return AppSettings(
        groq={"api_key": "test-key"},
        features={"voice_enabled": False},
        robot={"mock_mode": True},
    )


@pytest.mark.asyncio
async def test_stt_fallback_model(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF0000WAVE")
    service = STTService(StubTransport(), build_settings())

    text = await service.transcribe(str(audio_path))

    assert text == "hello marvin"
