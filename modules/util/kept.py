"""Values a menu reads off disk, answered from memory and read again on a worker thread.

A :class:`Kept` first reads once :func:`warm_all` runs, and again once expired, timed by
:func:`modules.util.file_listing.reuse_until`. A request waits at most :data:`WAIT` seconds.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Generic, Hashable, TypeVar

from .. import log
from .file_listing import WALK_WAIT, reuse_until, walks_deferred

__all__ = ["Kept", "KeptMap", "WAIT", "warm_all"]

logger = log.get_logger("util.kept")

T = TypeVar("T")

#: Seconds, counted from the start of a read, a request waits for it before being answered
#: from the last read that finished.
WAIT = WALK_WAIT

#: Values made before :func:`warm_all` ran, waiting for their first read.
_waiting: list["Kept"] = []

#: Whether :func:`warm_all` has run, after which a value starts reading as it is made.
_warmed = False

#: Serializes :data:`_waiting` and :data:`_warmed`.
_registry = threading.Lock()


def warm_all() -> None:
    """Start the first read of every value made so far, and of every value made from now on."""
    global _warmed
    with _registry:
        _warmed = True
        held = list(_waiting)
        _waiting.clear()
    for kept in held:
        kept.warm()


class Kept(Generic[T]):
    """One value a worker thread keeps current.

    Args:
        name: What the value is, named in the worker thread's name and in a debug line when a
            read fails.
        read: Reads the value off disk, on a worker thread.
        empty: Answered before the first read finishes, and kept where a read raises.
    """

    def __init__(self, name: str, read: Callable[[], T], empty: T):
        self.name = name
        self._read = read
        self._value = empty
        self._until = 0.0
        self._running = False
        self._began = 0.0
        self._done = threading.Condition()
        with _registry:
            start = _warmed
            if not start:
                _waiting.append(self)
        if start:
            self.warm()

    def warm(self) -> None:
        """Start a read on a worker thread unless one is running."""
        with self._done:
            self._launch()

    def expire(self) -> None:
        """Have the next request start a read and wait for it up to :data:`WAIT`."""
        with self._done:
            self._until = 0.0

    def get(self) -> T:
        """The value, with a read started once it has expired and waited for up to :data:`WAIT`."""
        with self._done:
            # While registration holds requests off, the value held is answered at once.
            if walks_deferred():
                return self._value
            if not self._running and time.monotonic() >= self._until:
                self._launch()
            deadline = self._began + WAIT
            while self._running:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self._done.wait(left)
            return self._value

    def read_now(self) -> T:
        """Read the value on the calling thread, keep it and answer it.

        Raises:
            Exception: Whatever the read raises; the value held is left as it was.
        """
        began = time.monotonic()
        value = self._read()
        with self._done:
            self._value = value
            self._until = reuse_until(began, time.monotonic())
        return value

    def _launch(self) -> None:
        """Start the worker unless it is running. Called holding the condition."""
        if self._running:
            return
        self._running = True
        self._began = time.monotonic()
        threading.Thread(target=self._run, name=f"was-kept-{self.name}", daemon=True).start()

    def _run(self) -> None:
        """Read the value and publish it, on the worker thread."""
        began = time.monotonic()
        try:
            value, failed = self._read(), False
        except Exception as error:
            logger.debug("%s could not be read: %s", self.name, error)
            value, failed = None, True
        with self._done:
            if not failed:
                self._value = value
            self._until = reuse_until(began, time.monotonic())
            self._running = False
            self._done.notify_all()


class KeptMap(Generic[T]):
    """One :class:`Kept` per key, each made on its key's first request.

    Args:
        name: What the values are.
        read: Reads one key's value off disk, on a worker thread.
        empty: Answered for a key before its first read finishes.
        most: Keys held at once; the oldest is dropped past it.
    """

    def __init__(self, name: str, read: Callable[[Hashable], T], empty: T, most: int = 64):
        self.name = name
        self._read = read
        self._empty = empty
        self._most = max(1, int(most))
        self._held: dict[Hashable, Kept[T]] = {}
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> T:
        """The value for ``key``, as :meth:`Kept.get` answers it."""
        with self._lock:
            held = self._held.get(key)
            if held is None:
                while len(self._held) >= self._most:
                    self._held.pop(next(iter(self._held)))
                held = self._held[key] = Kept(self.name, lambda: self._read(key), self._empty)
        return held.get()
