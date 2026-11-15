"""Shared locking primitives for mutable in-memory buffers."""

from __future__ import annotations

from functools import wraps
from threading import RLock
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def synchronized(method: F) -> F:
    """Run a buffer operation while holding its re-entrant lock."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapped  # type: ignore[return-value]


class ThreadSafeBuffer:
    """Mixin providing a re-entrant lock for compound buffer operations."""

    def __init__(self) -> None:
        self._lock = RLock()
