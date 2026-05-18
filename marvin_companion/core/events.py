"""Typed event payloads emitted by the orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True)
class BaseEvent:
    """Base event envelope."""

    correlation_id: str
    type: str


@dataclass(slots=True)
class TraceEvent(BaseEvent):
    """Stage timing trace."""

    stage: str
    latency_ms: float
    detail: str = ""


@dataclass(slots=True)
class TranscriptEvent(BaseEvent):
    """Transcript state update."""

    text: str
    source: Literal["microphone", "llm", "user"]


@dataclass(slots=True)
class TokenUsageEvent(BaseEvent):
    """LLM token usage event."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str


@dataclass(slots=True)
class PlaybackEvent(BaseEvent):
    """Audio playback lifecycle event."""

    status: Literal["started", "finished", "failed"]
    detail: str = ""


@dataclass(slots=True)
class HealthEvent(BaseEvent):
    """Health-check event."""

    status: Literal["ok", "warning", "error"]
    detail: str

