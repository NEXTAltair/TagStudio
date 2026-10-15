# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only

"""Runs image viewer decodes off the GUI thread and reports results on it."""

import queue
import threading
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
        - After `shutdown()`, nothing is emitted. The worker is a daemon thread, so a decode
          that never returns can't keep the application from exiting.
    """

    finished = Signal(int, object)
    _delivered = Signal(int, object)

    def __init__(self, decode: DecodeFunction = decode_image, parent: QObject | None = None):
        super().__init__(parent)
        self._decode = decode
        self._jobs: queue.SimpleQueue[_Job | None] = queue.SimpleQueue()
        self._active: dict[int, threading.Event] = {}
        self._thread: threading.Thread | None = None
        self._shut_down = False
        self.max_active = 0
        """The most requests that were outstanding at once (for tests and measurements)."""
        self.started_count = 0
        # Emitted from the worker thread; queued so the slot always runs on this object's thread.
        self._delivered.connect(self._on_delivered, Qt.ConnectionType.QueuedConnection)

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
                target=self._run, name="ImageViewerDecoder", daemon=True
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
        for event in self._active.values():
            event.set()
        self._jobs.put(None)
        if self._thread is not None and timeout > 0:
            self._thread.join(timeout)

    def _run(self) -> None:
        while (job := self._jobs.get()) is not None:
            try:
                outcome = self._decode(job.request, job.cancel.is_set)
            except Exception as e:
                logger.exception("[ImageDecodeRunner] Decoder raised", path=job.request.path)
                outcome = DecodeFailure(
                    DecodeErrorKind.CORRUPT, job.request.path, f"{type(e).__name__}: {e}"
                )
            try:
                self._delivered.emit(job.request_id, outcome)
            except RuntimeError:
                # The runner was deleted while the application is shutting down.
                return

    def _on_delivered(self, request_id: int, outcome: DecodeOutcome) -> None:
        if self._active.pop(request_id, None) is None:
            return
        if not self._shut_down:
            self.finished.emit(request_id, outcome)
