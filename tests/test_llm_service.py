from __future__ import annotations

import json

import httpx
import pytest

from marvin_companion.config.settings import AppSettings
from marvin_companion.services.http_client import GroqAPIClient
from marvin_companion.services.llm_service import LLMService
from marvin_companion.services.websocket_manager import WebSocketManager


def build_settings() -> AppSettings:
    return AppSettings(
        groq={"api_key": "test-key"},
        features={"voice_enabled": False},
        robot={"mock_mode": True},
    )


@pytest.mark.asyncio
async def test_llm_generate_parses_content_and_usage() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/openai/v1/chat/completions"
        data = json.loads(request.content.decode("utf-8"))
        assert data["model"] == "openai/gpt-oss-20b"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{\"response\": \"hello\"}"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
            headers={"x-request-id": "req-1"},
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="https://api.groq.com/openai/v1")
    events = WebSocketManager()
    service = LLMService(GroqAPIClient(build_settings(), client=client), build_settings(), events=events)
    subscriber = events.subscribe()

    content = await service.generate([{"role": "user", "content": "hello"}])

    assert content == "{\"response\": \"hello\"}"
    usage_event = await subscriber.get()
    assert usage_event.type == "token_usage"
    await client.aclose()


@pytest.mark.asyncio
async def test_llm_generate_stream_yields_deltas() -> None:
    body = (
        'data: {"choices":[{"delta":{"content":"{\\"response\\":"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":" \\"hi\\"}"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/openai/v1/chat/completions"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="https://api.groq.com/openai/v1")
    service = LLMService(GroqAPIClient(build_settings(), client=client), build_settings())

    chunks = [chunk async for chunk in service.generate_stream([{"role": "user", "content": "hello"}])]

    assert "".join(chunks) == '{"response": "hi"}'
    await client.aclose()
