"""Groq text-to-speech service."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from marvin_companion.config.settings import AppSettings
from marvin_companion.core.interfaces import TTSProvider
from marvin_companion.services.http_client import GroqAPIClient
from marvin_companion.utils.audio import chunk_text, concat_wav_bytes


class TTSService(TTSProvider):
    """Groq Orpheus text-to-speech service."""

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
        self.settings.audio_cache_dir.mkdir(parents=True, exist_ok=True)

    async def synthesize(self, text: str) -> bytes:
        cache_key = self._cache_key(text)
        cache_path = self.settings.audio_cache_dir / f"{cache_key}.wav"
        if cache_path.exists():
            return cache_path.read_bytes()

        chunks = chunk_text(text, self.settings.audio.tts_chunk_max_chars)
        wav_chunks: list[bytes] = []
        for chunk in chunks:
            wav_chunks.append(await self._synthesize_chunk(chunk))
        merged = concat_wav_bytes(wav_chunks)
        cache_path.write_bytes(merged)
        return merged

    async def _synthesize_chunk(self, chunk: str) -> bytes:
        payload = {
            "model": self.settings.groq.resolved_tts_model,
            "voice": self.settings.groq.tts_voice,
            "input": chunk,
            "response_format": "wav",
        }
        response = await self.transport.post_bytes("/audio/speech", payload)
        self.logger.info(
            "tts synthesized",
            extra={
                "model": self.settings.groq.resolved_tts_model,
                "voice": self.settings.groq.tts_voice,
                "byte_count": len(response.content),
            },
        )
        return response.content

    def _cache_key(self, text: str) -> str:
        digest = hashlib.sha256()
        digest.update(self.settings.groq.resolved_tts_model.encode("utf-8"))
        digest.update(self.settings.groq.tts_voice.encode("utf-8"))
        digest.update(text.encode("utf-8"))
        return digest.hexdigest()

