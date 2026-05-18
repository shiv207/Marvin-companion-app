"""Metrics and correlation helpers."""

from __future__ import annotations

import contextlib
import contextvars
import time
from dataclasses import dataclass


_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id",
    default="system",
)


def get_correlation_id() -> str:
    """Return the current correlation identifier."""

    return _correlation_id.get()


@contextlib.contextmanager
def correlation_context(correlation_id: str):
    """Set a correlation id for the current logical request."""

    token = _correlation_id.set(correlation_id)
    try:
        yield
    finally:
        _correlation_id.reset(token)


@dataclass(slots=True)
class Timer:
    """Simple timing context manager."""

    start: float = 0.0
    elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000.0

