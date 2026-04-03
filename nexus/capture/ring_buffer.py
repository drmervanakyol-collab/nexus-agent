"""
nexus/capture/ring_buffer.py
Thread-safe fixed-capacity ring buffer.

Uses a ``collections.deque(maxlen=capacity)`` as the backing store so
that overflow (push when full) automatically drops the oldest item — no
manual bookkeeping required.

All public methods acquire a single ``threading.Lock`` to guarantee
safe concurrent access from producer and consumer threads.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Generic, TypeVar

T = TypeVar("T")


class RingBuffer(Generic[T]):
    """
    Thread-safe ring buffer of fixed capacity.

    When the buffer is full, ``push()`` silently discards the oldest
    item (overflow: en eski silinir).

    Parameters
    ----------
    capacity:
        Maximum number of items the buffer holds.  Must be >= 1.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._buf: deque[T] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def push(self, item: T) -> None:
        """
        Add *item* to the buffer.

        If the buffer is already at capacity the oldest item is dropped
        automatically (``deque`` maxlen semantics).
        """
        with self._lock:
            self._buf.append(item)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def latest(self) -> T | None:
        """Return the most recently pushed item, or ``None`` if empty."""
        with self._lock:
            return self._buf[-1] if self._buf else None

    def last_n(self, n: int) -> list[T]:
        """
        Return up to *n* most recent items in chronological order
        (oldest first).

        Never raises even when *n* exceeds the current buffer size;
        returns fewer items in that case.
        """
        with self._lock:
            items = list(self._buf)
        return items[-n:] if n < len(items) else items

    def clear(self) -> None:
        """Remove all items from the buffer."""
        with self._lock:
            self._buf.clear()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    @property
    def capacity(self) -> int:
        """Maximum number of items the buffer can hold."""
        return self._capacity

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return len(self._buf) == 0

    @property
    def is_full(self) -> bool:
        with self._lock:
            return len(self._buf) == self._capacity
