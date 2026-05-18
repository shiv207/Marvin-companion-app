"""Retry helpers built on top of Tenacity."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from tenacity import AsyncRetrying, RetryError, retry_if_exception_type, stop_after_attempt, wait_exponential

from marvin_companion.config.settings import RetrySettings
from marvin_companion.core.exceptions import RetryExhaustedError


def build_retrying(
    settings: RetrySettings,
    retryable: tuple[type[BaseException], ...],
) -> AsyncRetrying:
    """Create a reusable retry policy."""

    return AsyncRetrying(
        reraise=True,
        stop=stop_after_attempt(settings.attempts),
        wait=wait_exponential(
            multiplier=settings.base_delay_seconds,
            max=settings.max_delay_seconds,
        ),
        retry=retry_if_exception_type(retryable),
    )


async def run_with_retry(
    func: Callable[[], Awaitable[object]],
    *,
    settings: RetrySettings,
    retryable: tuple[type[BaseException], ...],
) -> object:
    """Run an async operation with retry handling."""

    try:
        async for attempt in build_retrying(settings, retryable):
            with attempt:
                return await func()
    except RetryError as exc:
        raise RetryExhaustedError("Retry attempts exhausted.") from exc
    raise RetryExhaustedError("Retry attempts exhausted without execution.")

