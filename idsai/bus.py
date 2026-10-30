"""In-process pub/sub between the stream pipeline and its consumers (API, WebSocket).

`publish` never blocks and never waits for a slow consumer: each subscriber has its own bounded
queue and loses its oldest message when that queue is full; `dropped[topic]` counts every message
lost, over all subscribers. Use from one event loop (`asyncio.Queue` is not thread-safe).
"""

import asyncio
from typing import Any

TOPICS = ("bars", "scores", "alerts", "health")


class Bus:
    def __init__(self, maxsize: int = 1000):
        if maxsize <= 0:
            raise ValueError(f"maxsize must be > 0, got {maxsize}")
        self.maxsize = maxsize
        self.dropped: dict[str, int] = dict.fromkeys(TOPICS, 0)
        self._subs: dict[str, list[asyncio.Queue]] = {t: [] for t in TOPICS}

    def _queues(self, topic: str) -> list[asyncio.Queue]:
        if topic not in self._subs:
            raise ValueError(f"unknown topic {topic!r}; valid topics: {TOPICS}")
        return self._subs[topic]

    def subscribe(self, topic: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self.maxsize)
        self._queues(topic).append(q)
        return q

    def unsubscribe(self, topic: str, queue: asyncio.Queue) -> None:
        """Stop delivering to `queue`; removing a queue that is not subscribed is a no-op."""
        queues = self._queues(topic)
        if queue in queues:
            queues.remove(queue)

    def publish(self, topic: str, msg: Any) -> None:
        for q in self._queues(topic):
            if q.full():
                q.get_nowait()
                self.dropped[topic] += 1
            q.put_nowait(msg)
