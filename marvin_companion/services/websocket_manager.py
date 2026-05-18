"""In-memory async event broadcaster."""

from __future__ import annotations

import asyncio
from typing import Any

from marvin_companion.core.events import BaseEvent
from marvin_companion.core.interfaces import EventPublisher


class WebSocketManager(EventPublisher):
    """Broadcast events to multiple async subscribers."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[BaseEvent]] = set()

    def subscribe(self, *, maxsize: int = 100) -> asyncio.Queue[BaseEvent]:
        queue: asyncio.Queue[BaseEvent] = asyncio.Queue(maxsize=maxsize)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[BaseEvent]) -> None:
        self._subscribers.discard(queue)

    async def publish(self, event: BaseEvent) -> None:
        stale: list[asyncio.Queue[BaseEvent]] = []
        for queue in self._subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                    queue.put_nowait(event)
                except asyncio.QueueEmpty:
                    stale.append(queue)
        for queue in stale:
            self._subscribers.discard(queue)

