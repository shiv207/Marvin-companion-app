"""Shared async HTTP transport for Groq APIs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

from marvin_companion.config.settings import AppSettings
from marvin_companion.core.exceptions import UpstreamAPIError
from marvin_companion.utils.metrics import get_correlation_id
from marvin_companion.utils.retry import run_with_retry


@dataclass(slots=True)
class JsonAPIResponse:
    """JSON API response envelope."""

    data: dict[str, Any]
    headers: httpx.Headers
    status_code: int


@dataclass(slots=True)
class BinaryAPIResponse:
    """Binary API response envelope."""

    content: bytes
    headers: httpx.Headers
    status_code: int


class GroqAPIClient:
    """Reusable Groq API client built on top of httpx.AsyncClient."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        client: httpx.AsyncClient | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.settings = settings
        self.logger = logger or logging.getLogger(__name__)
        self.client = client or httpx.AsyncClient(
            base_url=self.settings.groq.base_url,
            timeout=httpx.Timeout(self.settings.groq.request_timeout_seconds),
        )

    async def post_json(self, path: str, payload: dict[str, Any]) -> JsonAPIResponse:
        """POST a JSON body and parse a JSON response."""

        async def operation() -> JsonAPIResponse:
            response = await self.client.post(
                path,
                json=payload,
                headers=self._headers(),
            )
            return self._json_response(response)

        return await run_with_retry(
            operation,
            settings=self.settings.retry,
            retryable=(httpx.HTTPError, UpstreamAPIError),
        )

    async def post_bytes(self, path: str, payload: dict[str, Any]) -> BinaryAPIResponse:
        """POST a JSON body and return raw bytes."""

        async def operation() -> BinaryAPIResponse:
            response = await self.client.post(
                path,
                json=payload,
                headers=self._headers(),
            )
            if response.status_code >= 400:
                raise self._upstream_error(response)
            return BinaryAPIResponse(response.content, response.headers, response.status_code)

        return await run_with_retry(
            operation,
            settings=self.settings.retry,
            retryable=(httpx.HTTPError, UpstreamAPIError),
        )

    async def post_multipart(
        self,
        path: str,
        *,
        data: dict[str, Any],
        files: dict[str, Any],
    ) -> JsonAPIResponse:
        """POST multipart form data."""

        async def operation() -> JsonAPIResponse:
            response = await self.client.post(
                path,
                data=data,
                files=files,
                headers=self._headers(include_json=False),
            )
            return self._json_response(response)

        return await run_with_retry(
            operation,
            settings=self.settings.retry,
            retryable=(httpx.HTTPError, UpstreamAPIError),
        )

    async def stream_json_lines(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream server-sent event lines as JSON payloads."""

        async with self.client.stream(
            "POST",
            path,
            json=payload,
            headers=self._headers(),
        ) as response:
            if response.status_code >= 400:
                raise self._upstream_error(response)
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                yield json.loads(data)

    async def aclose(self) -> None:
        """Close the underlying client."""

        await self.client.aclose()

    def _headers(self, *, include_json: bool = True) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.settings.groq.api_key}",
            "X-Request-ID": get_correlation_id(),
        }
        if include_json:
            headers["Content-Type"] = "application/json"
        return headers

    def _json_response(self, response: httpx.Response) -> JsonAPIResponse:
        if response.status_code >= 400:
            raise self._upstream_error(response)
        return JsonAPIResponse(response.json(), response.headers, response.status_code)

    def _upstream_error(self, response: httpx.Response) -> UpstreamAPIError:
        request_id = response.headers.get("x-request-id", get_correlation_id())
        return UpstreamAPIError(
            f"Upstream request failed with status {response.status_code}.",
            status_code=response.status_code,
            request_id=request_id,
            response_body=response.text,
        )

