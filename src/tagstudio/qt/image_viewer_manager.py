# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only

"""Connects the built-in image viewer to an opened library.

The manager owns one viewer window and the long-lived pieces shared by every viewer session:
the decode runner, the load scheduler, and the image cache. Sharing them means that closing and
reopening the viewer can never run more than one decode at a time.

The manager only reads from the library. It never changes the main window's selection, page,
search, or browsing history, and closing the viewer doesn't restore any of them.
"""

from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING

import structlog
from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QWidget

from tagstudio.core.viewer.cache import ImageCache, ImageKey, LoadKey
from tagstudio.core.viewer.decoding import (
    DEFAULT_POLICY,
    DecodedImage,
    DecodeErrorKind,
    DecodeFailure,
    DecodeOutcome,
    DecodePolicy,
    DecodeRequest,
    FileVersion,
)
from tagstudio.core.viewer.formats import viewable_name_predicate
from tagstudio.core.viewer.library_reader import (
    EntryLocation,
    LibraryReader,
    LibraryUnavailableError,
)
from tagstudio.core.viewer.scheduler import LoadRequest, LoadScheduler
from tagstudio.core.viewer.session import ViewerSession
from tagstudio.i18n.translations import Translations
from tagstudio.qt.controllers.image_viewer import ImageViewer
from tagstudio.qt.utils.image_decode_runner import ImageDecodeRunner

if TYPE_CHECKING:
    from tagstudio.core.library.alchemy.library import Library

logger = structlog.get_logger(__name__)

RESIZE_RELOAD_DELAY_MS = 150

_ERROR_KEYS: dict[DecodeErrorKind, str] = {
    DecodeErrorKind.MISSING: "image_viewer.error.missing_file",
    DecodeErrorKind.UNSUPPORTED: "image_viewer.error.unsupported",
    DecodeErrorKind.CODEC_UNAVAILABLE: "image_viewer.error.codec_unavailable",
    DecodeErrorKind.CORRUPT: "image_viewer.error.corrupt",
    DecodeErrorKind.OVER_BUDGET: "image_viewer.error.too_large",
    DecodeErrorKind.CHANGED: "image_viewer.error.changed",
    DecodeErrorKind.CANCELED: "image_viewer.loading",
}


class OpenResult(Enum):
    OPENED = "opened"
    NO_LIBRARY = "no_library"
    NO_IMAGES = "no_images"


class ImageViewerManager(QObject):
    """Opens the image viewer for a snapshot of entries and keeps it consistent.

    Args:
        parent_window: The window the viewer belongs to. Focus returns to it when the viewer
            closes, and the viewer doesn't keep the application running after it closes.
        runner: The decode runner. A new one is created if not given.
        cache: The decoded image cache. A new one is created if not given.
    """

    def __init__(
        self,
        parent_window: QWidget | None = None,
        runner: ImageDecodeRunner | None = None,
        cache: ImageCache | None = None,
        policy: DecodePolicy = DEFAULT_POLICY,
    ) -> None:
        super().__init__(parent_window)
        self._parent_window = parent_window
        self._runner = runner if runner is not None else ImageDecodeRunner(parent=self)
        self._runner.finished.connect(self._on_decoded)
        self._scheduler = LoadScheduler()
        self._cache = cache if cache is not None else ImageCache()
        self._policy = policy

        self._viewer: ImageViewer | None = None
        self._placed = False
        self._reader: LibraryReader | None = None
        self._session: ViewerSession | None = None
        self._location: EntryLocation | None = None
        self._retried_change = False

        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(RESIZE_RELOAD_DELAY_MS)
        self._resize_timer.timeout.connect(self._on_target_size_settled)

    # Public API ===============================================================

    @property
    def viewer(self) -> ImageViewer | None:
        return self._viewer

    @property
    def session(self) -> ViewerSession | None:
        return self._session

    @property
    def cache(self) -> ImageCache:
        return self._cache

    @property
    def scheduler(self) -> LoadScheduler:
        return self._scheduler

    @property
    def runner(self) -> ImageDecodeRunner:
        return self._runner

    @property
    def is_open(self) -> bool:
        return self._session is not None

    def open(self, library: Library, entry_ids: Sequence[int], start_id: int | None) -> OpenResult:
        """Show the viewer for a copy of `entry_ids`, starting at `start_id`.

        Only entries whose file type the viewer supports are included. If `start_id` isn't
        viewable, the closest following (or else preceding) viewable entry is shown. Calling
        this while the viewer is open replaces its session.
        """
        self._end_session()
        try:
            reader = LibraryReader(library)
        except LibraryUnavailableError:
            return OpenResult.NO_LIBRARY

        snapshot = tuple(entry_ids)
        viewable = reader.scan(snapshot, viewable_name_predicate())
        session = ViewerSession.from_snapshot(snapshot, viewable, start_id)
        if session.count == 0:
            reader.close()
            return OpenResult.NO_IMAGES

        self._reader = reader
        self._session = session
        viewer = self._ensure_viewer()
        self._show_window(viewer)
        self._show_current()
        return OpenResult.OPENED

    def close(self) -> None:
        """Close the viewer window and end its session."""
        if self._viewer is not None and self._viewer.isVisible():
            self._viewer.close()  # Ends the session through `closed`
        else:
            self._end_session()

    def close_library(self) -> None:
        """Must be called before the library is closed or replaced."""
        self.close()
        self._cache.clear()

    def shutdown(self) -> None:
        """Release everything when the application exits. Doesn't wait for a running decode."""
        self.close_library()
        self._runner.shutdown()
        if self._viewer is not None:
            self._viewer.deleteLater()
            self._viewer = None

    # Window ===================================================================

    def _ensure_viewer(self) -> ImageViewer:
        if self._viewer is None:
            viewer = ImageViewer(self._parent_window)
            viewer.navigate.connect(self._on_navigate)
            viewer.navigate_first.connect(lambda: self._on_navigate_to_end(last=False))
            viewer.navigate_last.connect(lambda: self._on_navigate_to_end(last=True))
            viewer.closed.connect(self._on_viewer_closed)
            viewer.canvas.target_size_changed.connect(self._resize_timer.start)
            self._viewer = viewer
        return self._viewer

    def _show_window(self, viewer: ImageViewer) -> None:
        if not viewer.isVisible():
            if not self._placed:
                # Later openings keep the size and position the user left the window at.
                self._place_window(viewer)
                self._placed = True
            viewer.show()
        viewer.raise_()
        viewer.activateWindow()
        viewer.setFocus()

    def _place_window(self, viewer: ImageViewer) -> None:
        screen = (
            self._parent_window.screen() if self._parent_window is not None else viewer.screen()
        )
        if screen is None:
            return
        area = screen.availableGeometry()
        viewer.resize(int(area.width() * 0.8), int(area.height() * 0.8))
        frame = viewer.frameGeometry()
        frame.moveCenter(area.center())
        viewer.move(frame.topLeft())

    def _on_viewer_closed(self) -> None:
        self._end_session()
        parent = self._parent_window
        if parent is not None and parent.isVisible():
            parent.activateWindow()

    # Session ==================================================================

    def _end_session(self) -> None:
        self._resize_timer.stop()
        if self._session is not None:
            self._session.invalidate()
        self._session = None
        if self._reader is not None:
            self._reader.close()
        self._reader = None
        self._location = None
        self._cancel(self._scheduler.invalidate())
        if self._viewer is not None:
            self._viewer.clear()

    def _on_navigate(self, delta: int) -> None:
        session = self._session
        if session is None:
            return
        if session.move(delta):
            self._show_current()
        elif self._viewer is not None:
            self._viewer.show_boundary(at_end=delta > 0)

    def _on_navigate_to_end(self, last: bool) -> None:
        session = self._session
        if session is None:
            return
        moved = session.move_to_last() if last else session.move_to_first()
        if moved:
            self._show_current()

    def _on_target_size_settled(self) -> None:
        if self._session is not None and self._location is not None:
            self._show_current(same_entry=True)

    # Loading ==================================================================

    def _show_current(self, same_entry: bool = False) -> None:
        """Show the session's current entry, from the cache or by requesting a decode.

        Args:
            same_entry: Reload the entry that is already shown (e.g. after a resize). The
                current image stays visible until the new one is ready.
        """
        session, reader, viewer = self._session, self._reader, self._ensure_viewer()
        entry_id = session.current_id if session is not None else None
        if session is None or reader is None or entry_id is None:
            return
        if not same_entry:
            self._retried_change = False
        viewer.show_position(session.index or 0, session.count)

        try:
            location = reader.resolve_one(entry_id)
        except LibraryUnavailableError:
            logger.warning("[ImageViewer] Library is no longer available")
            self.close()
            return

        self._location = location
        if location is None:
            self._show_message(Translations["image_viewer.error.missing_entry"])
            return
        viewer.show_file_name(location.path.name)

        version = FileVersion.of(location.path)
        if version is None:
            self._show_message(Translations["image_viewer.error.missing_file"])
            return

        key = LoadKey(
            ImageKey(reader.token, entry_id, location.path, version, self._policy),
            viewer.target_size(),
        )
        if (cached := self._cache.get(key)) is not None:
            self._cancel(self._scheduler.satisfy())
            viewer.show_image(cached)
            return

        current = viewer.canvas.image
        if not (same_entry and current is not None):
            viewer.show_message(Translations["image_viewer.loading"])
        actions = self._scheduler.submit(key)
        self._cancel(actions.cancel)
        if actions.start is not None:
            self._start(actions.start)

    def _show_message(self, message: str) -> None:
        self._cancel(self._scheduler.satisfy())
        if self._viewer is not None:
            self._viewer.show_message(message)

    def _cancel(self, request: LoadRequest | None) -> None:
        if request is not None:
            self._runner.cancel(request.request_id)

    def _start(self, request: LoadRequest) -> None:
        key = request.key
        assert isinstance(key, LoadKey)
        if (cached := self._cache.get(key)) is not None:
            # e.g. an earlier, superseded request for the same image finished in the meantime.
            self._finish(request.request_id, cached)
            return
        decode_request = DecodeRequest(
            path=key.image.path,
            target_size=key.target_size,
            expected_version=key.image.version,
            policy=key.image.policy,
        )
        self._runner.start(request.request_id, decode_request)

    def _on_decoded(self, request_id: int, outcome: DecodeOutcome) -> None:
        running = self._scheduler.running
        if running is None or running.request_id != request_id:
            return
        key = running.key
        assert isinstance(key, LoadKey)
        reader = self._reader
        if (
            isinstance(outcome, DecodedImage)
            and reader is not None
            and reader.is_valid()
            and key.image.library_token == reader.token
        ):
            self._cache.put(key, outcome)
        self._finish(request_id, outcome)

    def _finish(self, request_id: int, outcome: DecodeOutcome) -> None:
        actions = self._scheduler.finish(request_id)
        if actions.apply and self._session is not None and self._viewer is not None:
            if isinstance(outcome, DecodedImage):
                self._viewer.show_image(outcome)
            elif outcome.kind == DecodeErrorKind.CHANGED and not self._retried_change:
                # The file was modified after it was checked; check it again, once.
                self._retried_change = True
                QTimer.singleShot(0, lambda: self._show_current(same_entry=True))
            else:
                self._show_failure(outcome)
        if actions.start is not None:
            self._start(actions.start)

    def _show_failure(self, failure: DecodeFailure) -> None:
        assert self._viewer is not None
        logger.info(
            "[ImageViewer] Could not show image",
            path=failure.path,
            kind=failure.kind.value,
            message=failure.message,
        )
        self._viewer.show_message(Translations[_ERROR_KEYS[failure.kind]])
