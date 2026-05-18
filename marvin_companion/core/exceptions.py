"""Typed application exceptions."""

from __future__ import annotations


class MarvinCompanionError(Exception):
    """Base class for application errors."""


class ConfigurationError(MarvinCompanionError):
    """Raised when configuration is invalid."""


class UpstreamAPIError(MarvinCompanionError):
    """Raised when an upstream API returns an error."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id
        self.response_body = response_body


class RetryExhaustedError(MarvinCompanionError):
    """Raised when retry attempts are exhausted."""


class InvalidResponseError(MarvinCompanionError):
    """Raised when a provider response cannot be parsed or validated."""


class AudioValidationError(MarvinCompanionError):
    """Raised when audio data is malformed or unsupported."""


class PlaybackError(MarvinCompanionError):
    """Raised when audio playback fails."""


class HealthCheckError(MarvinCompanionError):
    """Raised when startup diagnostics fail."""

