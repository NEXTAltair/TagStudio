# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only

"""Request control for viewer image loads: at most one running and one (latest) pending.

The scheduler is a pure state machine. It doesn't start threads or decode anything; each method
returns the actions its owner must carry out (start a request, request a cancellation). This
keeps the rules testable without Qt, threads or a database.
"""

import itertools
from collections.abc import Hashable
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class LoadRequest:
    """One attempt to load an image.

    Attributes:
        request_id: Unique for the scheduler's whole lifetime.
        generation: The scheduler generation the request was made in.
        key: Identifies the result (e.g. a cache key). Requests with equal keys produce
            interchangeable results.
    """

    request_id: int
    generation: int
    key: Hashable


@dataclass(frozen=True, slots=True)
class SubmitActions:
    """What to do after `LoadScheduler.submit()`.

    Attributes:
        request: The request now wanted by the viewer.
        start: A request to start running now, if any.
        cancel: A running request to cooperatively cancel, if any.
    """

    request: LoadRequest
    start: LoadRequest | None = None
    cancel: LoadRequest | None = None


@dataclass(frozen=True, slots=True)
class FinishActions:
    """What to do after `LoadScheduler.finish()`.

    Attributes:
        apply: Whether the finished request's result should be shown.
        start: A pending request to start running now, if any.
    """

    apply: bool
    start: LoadRequest | None = None


@dataclass(slots=True)
class _Stats:
    submitted: int = 0
    started: int = 0
    finished: int = 0
    applied: int = 0
    superseded: int = 0
    reused: int = 0


@dataclass(slots=True)
class LoadScheduler:
    """Coordinates loads so that only one decode runs at a time across all viewer sessions.

    Rules:
        - At most one request runs. A running request keeps its slot until `finish()` is called
          for it, even after it was asked to cancel; cancellation is cooperative.
        - At most one request waits. A newer submission replaces the waiting one.
        - Only the most recently submitted request (the "wanted" one) may be applied, and only
          if it was made in the current generation. Results of older requests are never shown,
          even when they're for the same image (A -> B -> A).
        - `invalidate()` drops the waiting request and starts a new generation, e.g. when the
          viewer closes or the library changes. A request that is still running keeps the slot
          until it finishes, so reopening the viewer doesn't stack up decodes.
        - Submitting the key that is already running (and not being canceled) reuses it.
    """

    _ids: itertools.count[int] = field(default_factory=lambda: itertools.count(1))
    _generation: int = 0
    _running: LoadRequest | None = None
    _running_cancel_requested: bool = False
    _pending: LoadRequest | None = None
    _wanted: LoadRequest | None = None
    stats: _Stats = field(default_factory=_Stats)

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def running(self) -> LoadRequest | None:
        return self._running

    @property
    def pending(self) -> LoadRequest | None:
        return self._pending

    @property
    def wanted(self) -> LoadRequest | None:
        return self._wanted

    @property
    def is_idle(self) -> bool:
        return self._running is None and self._pending is None

    def _start(self, request: LoadRequest) -> LoadRequest:
        self._running = request
        self._running_cancel_requested = False
        self.stats.started += 1
        return request

    def _cancel_running(self) -> LoadRequest | None:
        if self._running is None or self._running_cancel_requested:
            return None
        self._running_cancel_requested = True
        return self._running

    def submit(self, key: Hashable) -> SubmitActions:
        """Ask for the result for `key`, replacing whatever was wanted before.

        If the running request is for the same key, wasn't asked to cancel, and belongs to the
        current generation, it is reused instead of decoding the same image twice.
        """
        self.stats.submitted += 1
        if self._pending is not None:
            self.stats.superseded += 1
        self._pending = None

        running = self._running
        if (
            running is not None
            and not self._running_cancel_requested
            and running.generation == self._generation
            and running.key == key
        ):
            self.stats.reused += 1
            self._wanted = running
            return SubmitActions(running)

        request = LoadRequest(next(self._ids), self._generation, key)
        self._wanted = request

        if running is None:
            return SubmitActions(request, start=self._start(request))

        self._pending = request
        return SubmitActions(request, cancel=self._cancel_running())

    def satisfy(self) -> LoadRequest | None:
        """Mark the wanted result as available without loading (e.g. a cache hit).

        Returns:
            A running request to cooperatively cancel, if any.
        """
        if self._pending is not None:
            self.stats.superseded += 1
        self._pending = None
        self._wanted = None
        return self._cancel_running()

    def finish(self, request_id: int) -> FinishActions:
        """Record that a request ended (successfully, with an error, or canceled).

        Calls for unknown or already finished requests are ignored, so the slot is released
        exactly once per started request.
        """
        if self._running is None or self._running.request_id != request_id:
            return FinishActions(apply=False)

        finished = self._running
        self._running = None
        self._running_cancel_requested = False
        self.stats.finished += 1

        apply = (
            self._wanted is not None
            and self._wanted.request_id == finished.request_id
            and finished.generation == self._generation
        )
        if apply:
            self._wanted = None
            self.stats.applied += 1

        start: LoadRequest | None = None
        if self._pending is not None:
            start = self._start(self._pending)
            self._pending = None
        return FinishActions(apply=apply, start=start)

    def invalidate(self) -> LoadRequest | None:
        """Drop the pending request and start a new generation.

        Returns:
            A running request to cooperatively cancel, if any.
        """
        self._generation += 1
        self._pending = None
        self._wanted = None
        return self._cancel_running()
