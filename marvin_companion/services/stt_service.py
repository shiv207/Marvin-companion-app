"""Groq speech-to-text service."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from marvin_companion.config.settings import AppSettings
from marvin_companion.core.exceptions import AudioValidationError, InvalidResponseError, UpstreamAPIError
from marvin_companion.core.interfaces import STTProvider
from marvin_companion.services.http_client import GroqAPIClient, JsonAPIResponse


class STTService(STTProvider):
    """Groq Whisper transcription service."""

    def __init__(
        self,
        transport: GroqAPIClient,
        settings: AppSettings,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self.transport = transport
        self.settings = settings
        self.logger = logger or logging.getLogger(__name__)

    async def transcribe(self, audio_path: str) -> str:
        response = await self.transcribe_verbose(audio_path)
        try:
            return response["text"]
        except KeyError as exc:
            raise InvalidResponseError("Verbose STT response did not include text.") from exc

    async def transcribe_verbose(self, audio_path: str) -> dict[str, Any]:
        path = Path(audio_path)
        if not path.exists():
            raise AudioValidationError(f"Audio file does not exist: {audio_path}")
        try:
            return (await self._transcribe_with_model(path, self.settings.groq.stt_model_primary)).data
        except UpstreamAPIError:
            self.logger.warning(
                "primary stt model failed, retrying fallback",
                extra={"primary_model": self.settings.groq.stt_model_primary},
            )
            return (await self._transcribe_with_model(path, self.settings.groq.stt_model_fallback)).data

    async def _transcribe_with_model(self, path: Path, model: str) -> JsonAPIResponse:
        with path.open("rb") as audio_file:
            files = {"file": (path.name, audio_file, "audio/wav")}
            data = {
                "model": model,
                "response_format": "verbose_json",
                "temperature": "0",
            }
            return await self.transport.post_multipart("/audio/transcriptions", data=data, files=files)

