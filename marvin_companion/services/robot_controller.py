"""Async Marvin robot controller."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from marvin_companion.config.settings import RobotSettings
from marvin_companion.core.interfaces import RobotController


class AsyncRobotController(RobotController):
    """HTTP client for the Marvin robot integration."""

    def __init__(
        self,
        settings: RobotSettings,
        *,
        client: httpx.AsyncClient | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.settings = settings
        self.logger = logger or logging.getLogger(__name__)
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(settings.timeout_seconds))

    async def send_command(self, command: str, face: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any]
        if command == "idle" and face:
            payload = {"face": face}
        else:
            payload = {"command": command}
            if face:
                payload["face"] = face

        if self.settings.mock_mode:
            return {"status": "success", "mock": True, "payload": payload}
        if not self.settings.base_url:
            return {"error": "Robot base URL is not configured."}

        try:
            response = await self.client.post(
                f"{self.settings.base_url.rstrip('/')}/api/command",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            self.logger.warning("robot command failed", extra={"command": command, "error": str(exc)})
            return {"error": str(exc)}

    async def get_status(self) -> dict[str, Any]:
        if self.settings.mock_mode:
            return {
                "currentCommand": "idle",
                "currentFace": "happy",
                "networkConnected": True,
                "mock": True,
            }
        if not self.settings.base_url:
            return {"error": "Robot base URL is not configured."}
        try:
            response = await self.client.get(f"{self.settings.base_url.rstrip('/')}/api/status")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            return {"error": str(exc)}

    async def aclose(self) -> None:
        await self.client.aclose()
