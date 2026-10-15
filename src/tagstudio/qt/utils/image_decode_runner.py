# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only

"""Runs image viewer decodes off the GUI thread and reports results on it."""

import functools
import queue
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass

import structlog
from PySide6.QtCore import QObject, Qt, Signal

from tagstudio.core.viewer.decoding import (
    CancelCheck,
    DecodeErrorKind,
    DecodeFailure,
    DecodeOutcome,
    DecodeRequest,
    decode_image,
)

logger = structlog.get_logger(__name__)

DecodeFunction = Callable[[DecodeRequest, CancelCheck | None], DecodeOutcome]


@dataclass(frozen=True, slots=True)
class _Job:
    request_id: int
    request: DecodeRequest
    cancel: threading.Event


class _Dispatcher(QObject):
    """Moves results from worker threads to the GUI thread.

    One instance is created on the GUI thread and is never deleted, so a worker thread never
    holds the last reference to a QObject (which would destroy it on the wrong thread).
    """

    delivered = Signal(object, int, object)

    def __init__(self) -> None:
        super().__init__()
        self.delivered.connect(self._deliver, Qt.ConnectionType.QueuedConnection)

    def _deliver(
        self,
        runner_ref: weakref.ref[ImageDecodeRunner],
        request_id: int,
        outcome: DecodeOutcome,
    ) -> None:
        if (runner := runner_ref()) is not None:
            runner._on_delivered(request_id, outcome)  # pyright: ignore[reportPrivateUsage]


@functools.cache
def _dispatcher() -> _Dispatcher:
    return _Dispatcher()


def _worker_loop(
    jobs: queue.SimpleQueue[_Job | None],
    decode: DecodeFunction,
    runner_ref: weakref.ref[ImageDecodeRunner],
    dispatcher: _Dispatcher,
) -> None:
    # NOTE: Only holds a weak reference to the runner; see `_Dispatcher`.
    while (job := jobs.get()) is not None:
        try:
            outcome = decode(job.request, job.cancel.is_set)
        except Exception as e:
            logger.exception("[ImageDecodeRunner] Decoder raised", path=job.request.path)
            outcome = DecodeFailure(
                DecodeErrorKind.CORRUPT, job.request.path, f"{type(e).__name__}: {e}"
            )
        try:
            dispatcher.delivered.emit(runner_ref, job.request_id, outcome)
        except RuntimeError:
            # Qt objects are being torn down because the application is exiting.
            return


def _stop_worker(jobs: queue.SimpleQueue[_Job | None], active: dict[int, threading.Event]) -> None:
    for event in active.values():
        event.set()
    jobs.put(None)


class ImageDecodeRunner(QObject):
    """Decodes images on a single background thread.

    Contract:
        - `start()` never blocks the GUI thread. Callers (the load scheduler) keep at most one
          request outstanding; jobs are otherwise processed one at a time in order.
        - `finished(request_id, outcome)` is emitted on the GUI thread exactly once for every
          started request, whether it succeeded, failed, raised, or was canceled.
        - `cancel()` is cooperative. A decode inside a native codec call can't be interrupted;
          it still finishes and reports its result, which the receiver may still cache.
        - Results only contain immutable, self-owned buffers (`DecodedImage.data` is `bytes`).
          `QPixmap`s are created by the receiver on the GUI thread.
        - After `shutdown()`, or once the runner is garbage collected, nothing is emitted and
          the worker thread ends after its current decode. It is a daemon thread, so a decode
          that never returns can't keep the application from exiting.
        - The worker thread never holds a strong reference to the runner, so the runner is
          always destroyed on the thread that owns it.
    """

    finished = Signal(int, object)

    def __init__(self, decode: DecodeFunction = decode_image, parent: QObject | None = None):
        super().__init__(parent)
        self._decode = decode
        self._jobs: queue.SimpleQueue[_Job | None] = queue.SimpleQueue()
        self._active: dict[int, threading.Event] = {}
        self._thread: threading.Thread | None = None
        self._shut_down = False
        self._dispatcher = _dispatcher()
        self._finalizer = weakref.finalize(self, _stop_worker, self._jobs, self._active)
        self.max_active = 0
        """The most requests that were outstanding at once (for tests and measurements)."""
        self.started_count = 0

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def is_shut_down(self) -> bool:
        return self._shut_down

    def start(self, request_id: int, request: DecodeRequest) -> None:
        if self._shut_down:
            logger.warning("[ImageDecodeRunner] Ignoring request after shutdown", id=request_id)
            return
        if request_id in self._active:
            raise ValueError(f"Request {request_id} is already running")
        if self._thread is None:
            self._thread = threading.Thread(
                target=_worker_loop,
                args=(self._jobs, self._decode, weakref.ref(self), self._dispatcher),
                name="ImageViewerDecoder",
                daemon=True,
            )
            self._thread.start()
        cancel = threading.Event()
        self._active[request_id] = cancel
        self.max_active = max(self.max_active, len(self._active))
        self.started_count += 1
        self._jobs.put(_Job(request_id, request, cancel))

    def cancel(self, request_id: int) -> None:
        if (event := self._active.get(request_id)) is not None:
            event.set()

    def shutdown(self, timeout: float = 0.0) -> None:
        """Stop delivering results and let the worker thread end.

        Args:
            timeout: Seconds to wait for a running decode to return. Zero doesn't wait.
        """
        if self._shut_down:
            return
        self._shut_down = True
        self._finalizer()
        if self._thread is not None and timeout > 0:
            self._thread.join(timeout)

    def _on_delivered(self, request_id: int, outcome: DecodeOutcome) -> None:
        if self._active.pop(request_id, None) is None:
            return
        if not self._shut_down:
            self.finished.emit(request_id, outcome)
