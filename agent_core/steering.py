"""Steering queue — a dependency-free, in-memory inbox for mid-run user messages.

Design
------
"Steering" lets a user inject guidance into an agent *while* it is working:
messages are pushed into a queue at any time and the running loop drains them
at safe checkpoints (e.g. between tool calls / turns).

Design choices:

* **Dependency-free** — only ``collections.deque`` and the standard library;
  no imports from this package, so it can be imported from anywhere without
  circular-import risk.
* **FIFO order** — messages are drained in arrival order via a ``deque``, so
  older steering guidance is never starved by newer messages.
* **Per-message length cap** (``max_message_len``): over-long messages are
  truncated to the cap at push time (with trailing whitespace stripped) rather
  than rejected, keeping queue memory bounded and predictable.
* **Queue cap** (``max_queue_size``): when full, the *newest* message is
  dropped (drop-newest back-pressure) — already-delivered guidance is never
  silently evicted, and losing one surplus message is preferable to unbounded
  growth.
* **Drain semantics** — ``drain()`` atomically returns all pending messages
  and empties the queue, so each message is delivered exactly once.
* **Not thread-safe** by design: callers are expected to push from the same
  event loop; a lock can be added later without changing the API.

API summary
-----------
``push(message)``  enqueue a steering message (truncated to per-message cap;
                   dropped if the queue is full).
``drain()``        pop all pending messages FIFO and return them as a list.
``pending()``      number of undelivered messages currently queued.
``clear()``        discard all pending messages.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Final, List

#: Default per-message length cap (characters).  Steering messages are short
#: guidance snippets; anything longer is truncated to keep memory bounded.
DEFAULT_MAX_MESSAGE_LEN: Final[int] = 2000

#: Default maximum number of undelivered messages held in the queue.
DEFAULT_MAX_QUEUE_SIZE: Final[int] = 32


class SteeringQueue:
    """FIFO steering-message inbox with length and depth caps.

    Parameters
    ----------
    max_message_len:
        Per-message character cap; longer messages are truncated on push.
    max_queue_size:
        Maximum number of undelivered messages; pushes beyond this cap are
        dropped (drop-newest) rather than evicting older queued messages.
    """

    __slots__ = ("_queue", "max_message_len", "max_queue_size", "dropped_count")

    def __init__(
        self,
        max_message_len: int = DEFAULT_MAX_MESSAGE_LEN,
        max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
    ) -> None:
        if max_message_len <= 0:
            raise ValueError("max_message_len must be positive")
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        self.max_message_len = int(max_message_len)
        self.max_queue_size = int(max_queue_size)
        self._queue: Deque[str] = deque()
        #: Number of pushes dropped because the queue was full.
        self.dropped_count: int = 0

    def push(self, message: str) -> bool:
        """Enqueue a steering message.

        Returns ``True`` if the message was queued, ``False`` if it was
        dropped (empty/whitespace-only message, or queue at capacity).
        Whitespace is stripped and over-long messages are truncated to
        ``max_message_len``.
        """
        text = str(message).strip()[: self.max_message_len]
        if not text:
            return False
        if len(self._queue) >= self.max_queue_size:
            self.dropped_count += 1
            return False
        self._queue.append(text)
        return True

    def drain(self) -> List[str]:
        """Return all pending messages (FIFO order) and empty the queue."""
        if not self._queue:
            return []
        messages = list(self._queue)
        self._queue.clear()
        return messages

    def pending(self) -> int:
        """Number of undelivered messages currently queued."""
        return len(self._queue)

    def clear(self) -> None:
        """Discard all pending messages."""
        self._queue.clear()


__all__: Final[list[str]] = [
    "SteeringQueue",
    "DEFAULT_MAX_MESSAGE_LEN",
    "DEFAULT_MAX_QUEUE_SIZE",
]
