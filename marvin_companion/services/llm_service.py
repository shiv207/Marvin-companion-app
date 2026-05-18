"""Groq chat completions service."""

from __future__ import annotations

import logging
from typing import AsyncIterator

from marvin_companion.config.settings import AppSettings
from marvin_companion.core.events import TokenUsageEvent
from marvin_companion.core.exceptions import InvalidResponseError
from marvin_companion.core.interfaces import EventPublisher, LLMProvider
from marvin_companion.services.http_client import GroqAPIClient
from marvin_companion.utils.metrics import get_correlation_id


class LLMService(LLMProvider):
    """Groq-backed LLM service."""

    def __init__(
        self,
        transport: GroqAPIClient,
        settings: AppSettings,
        *,
        events: EventPublisher | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.transport = transport
        self.settings = settings
        self.events = events
        self.logger = logger or logging.getLogger(__name__)

    async def generate(self, messages: list[dict], stream: bool = False) -> str:
        if stream:
            chunks = [chunk async for chunk in self.generate_stream(messages)]
            return "".join(chunks)
        payload = {
            "model": self.settings.groq.resolved_llm_model,
            "messages": messages,
            "temperature": self.settings.groq.llm_temperature,
            "max_tokens": self.settings.groq.llm_max_tokens,
            "stream": False,
        }
        response = await self.transport.post_json("/chat/completions", payload)
        try:
            content = response.data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InvalidResponseError("Malformed LLM completion response.") from exc

        usage = response.data.get("usage") or {}
        await self._publish_usage(usage)
        self.logger.info(
            "llm completion",
            extra={
                "model": self.settings.groq.resolved_llm_model,
                "request_id": response.headers.get("x-request-id", get_correlation_id()),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
        )
        return content

    async def generate_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        payload = {
            "model": self.settings.groq.resolved_llm_model,
            "messages": messages,
            "temperature": self.settings.groq.llm_temperature,
            "max_tokens": self.settings.groq.llm_max_tokens,
            "stream": True,
        }
        async for data in self.transport.stream_json_lines("/chat/completions", payload):
            choices = data.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                yield content

    async def _publish_usage(self, usage: dict) -> None:
        if not self.events:
            return
        await self.events.publish(
            TokenUsageEvent(
                correlation_id=get_correlation_id(),
                type="token_usage",
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                total_tokens=int(usage.get("total_tokens", 0)),
                model=self.settings.groq.resolved_llm_model,
            )
        )

