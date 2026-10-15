# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only


import html
from typing import override

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QKeyEvent, QWheelEvent
from PySide6.QtWidgets import QWidget

from tagstudio.core.viewer.decoding import DecodedImage
from tagstudio.i18n.translations import Translations
from tagstudio.qt.views.image_viewer_view import ImageViewerView

WHEEL_STEP = 120
"""Angle delta of one standard mouse wheel notch (15 degrees, in eighths of a degree)."""

PIXELS_PER_STEP = 120
"""Scroll distance that moves one entry, for events that only report a pixel delta."""

_IGNORED_WHEEL_MODIFIERS = (
    Qt.KeyboardModifier.ControlModifier
    | Qt.KeyboardModifier.ShiftModifier
    | Qt.KeyboardModifier.AltModifier
    | Qt.KeyboardModifier.MetaModifier
)

_NEXT_KEYS = {Qt.Key.Key_Right, Qt.Key.Key_Down, Qt.Key.Key_PageDown}
_PREVIOUS_KEYS = {Qt.Key.Key_Left, Qt.Key.Key_Up, Qt.Key.Key_PageUp}


class WheelStepper:
    """Turns vertical wheel deltas into whole navigation steps.

    High resolution wheels and touchpads send many small deltas; they're accumulated until they
    add up to a full notch. Changing direction discards the partial amount.
    """

    def __init__(self, step: int = WHEEL_STEP) -> None:
        self._step = step
        self._accumulated = 0

    def reset(self) -> None:
        self._accumulated = 0

    def feed(self, delta_y: int) -> int:
        """Add a vertical angle delta.

        Returns:
            The number of entries to move: positive for next (wheel down), negative for previous.
        """
        if delta_y == 0:
            return 0
        if self._accumulated and (delta_y > 0) != (self._accumulated > 0):
            self._accumulated = 0
        self._accumulated += delta_y
        notches = int(self._accumulated / self._step)  # Truncates toward zero
        self._accumulated -= notches * self._step
        return -notches


class ImageViewer(ImageViewerView):
    """A read-only window that shows one image at a time from a viewer session.

    The window only reports navigation intents; the image viewer manager owns the session and
    decides what to show. Input is handled locally:
        - Mouse wheel down/up: next/previous image. Touchpads that only report pixel deltas
          move one image per `PIXELS_PER_STEP` pixels. Horizontal scrolling and wheel input with
          modifier keys are ignored.
        - Right/Down/Page Down and Left/Up/Page Up: next/previous image.
        - Home/End: first/last image.
        - Esc: close the viewer.

    While the viewer has focus, application shortcuts are suppressed, so key presses can't
    reach editing actions (delete, paste, tagging, clearing the selection) in the main window.
    """

    navigate = Signal(int)
    """Move by the given number of entries."""
    navigate_first = Signal()
    navigate_last = Signal()
    closed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._wheel = WheelStepper()
        self._boundary_timer = QTimer(self)
        self._boundary_timer.setSingleShot(True)
        self._boundary_timer.setInterval(1500)
        self._boundary_timer.timeout.connect(self._restore_position_text)
        self._position_text = ""
        self.setWindowTitle(Translations["image_viewer.title"])
        # Handle the wheel over any part of the window, without relying on event propagation.
        for widget in (self.canvas, self.status_widget, *self.status_widget.findChildren(QWidget)):
            widget.installEventFilter(self)

    # Display ==================================================================

    def show_position(self, index: int, count: int) -> None:
        self._position_text = Translations.format(
            "image_viewer.position", index=index + 1, count=count
        )
        self._boundary_timer.stop()
        self.position_label.setText(self._position_text)

    def show_boundary(self, at_end: bool) -> None:
        """Briefly tell the user that there is no next (or previous) image."""
        key = "image_viewer.last_image" if at_end else "image_viewer.first_image"
        self.position_label.setText(f"{self._position_text}  ({Translations[key]})")
        self._boundary_timer.start()

    def _restore_position_text(self) -> None:
        self.position_label.setText(self._position_text)

    def show_file_name(self, file_name: str) -> None:
        # File names are shown literally, never interpreted as markup.
        self.name_label.setText(file_name)
        self.name_label.setToolTip(f"<p>{html.escape(file_name)}</p>" if file_name else "")
        self.setWindowTitle(
            Translations.format("image_viewer.window_title", file_name=file_name)
            if file_name
            else Translations["image_viewer.title"]
        )

    def show_image(self, image: DecodedImage) -> None:
        self.canvas.set_image(image)
        details = Translations.format(
            "image_viewer.dimensions", width=image.source_width, height=image.source_height
        )
        if image.is_animated:
            details = f"{details}  {Translations['image_viewer.animated']}"
        self.details_label.setText(f"{details}  {image.source_format}")

    def show_message(self, message: str) -> None:
        """Replace the image with a message (loading, missing file, errors)."""
        self.canvas.set_message(message)
        self.details_label.setText("")

    def clear(self) -> None:
        self.canvas.clear()
        self._wheel.reset()
        self._boundary_timer.stop()
        self._position_text = ""
        for label in (self.position_label, self.name_label, self.details_label):
            label.setText("")
        self.setWindowTitle(Translations["image_viewer.title"])

    def target_size(self) -> tuple[int, int]:
        return self.canvas.target_size()

    # Input ====================================================================

    @override
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Wheel and isinstance(event, QWheelEvent):
            self.wheelEvent(event)
            return True
        return super().eventFilter(watched, event)

    @override
    def wheelEvent(self, event: QWheelEvent) -> None:
        event.accept()
        if event.modifiers() & _IGNORED_WHEEL_MODIFIERS:
            self._wheel.reset()
            return
        delta_y = event.angleDelta().y()
        if delta_y == 0:
            # Some touchpads only report a pixel delta.
            delta_y = round(event.pixelDelta().y() * WHEEL_STEP / PIXELS_PER_STEP)
        steps = self._wheel.feed(delta_y)
        if steps:
            self.navigate.emit(steps)

    @override
    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        modifiers = event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier
        event.accept()
        if modifiers != Qt.KeyboardModifier.NoModifier:
            return
        if key == Qt.Key.Key_Escape:
            self.close()
        elif key in _NEXT_KEYS:
            self.navigate.emit(1)
        elif key in _PREVIOUS_KEYS:
            self.navigate.emit(-1)
        elif key == Qt.Key.Key_Home:
            self.navigate_first.emit()
        elif key == Qt.Key.Key_End:
            self.navigate_last.emit()

    @override
    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.ShortcutOverride:
            # Claim every key press, so no application shortcut (e.g. Delete, Ctrl+V) triggers
            # while the viewer is focused. Keys are then handled by `keyPressEvent`.
            event.accept()
            return True
        return super().event(event)

    @override
    def closeEvent(self, event: QCloseEvent) -> None:
        super().closeEvent(event)
        if event.isAccepted():
            self.closed.emit()
