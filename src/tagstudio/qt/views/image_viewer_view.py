# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only


import math
from typing import override

from PySide6.QtCore import QEvent, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QPainter,
    QPaintEvent,
    QPixmap,
    QResizeEvent,
)
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from tagstudio.core.viewer.decoding import DecodedImage, fit_size

BACKGROUND_COLOR = QColor("#1e1e1e")
MESSAGE_COLOR = QColor("#c8c8c8")
CHECKER_COLORS = (QColor("#3a3a3a"), QColor("#2c2c2c"))
CHECKER_SIZE = 8

_QIMAGE_FORMATS: dict[str, QImage.Format] = {
    "L": QImage.Format.Format_Grayscale8,
    "RGB": QImage.Format.Format_RGB888,
    "RGBA": QImage.Format.Format_RGBA8888,
}


def to_qimage(image: DecodedImage) -> QImage:
    """Return a read-only QImage that uses a decoded image's pixel buffer without copying it.

    The QImage doesn't own the buffer: keep `image` alive for as long as the QImage is used.
    """
    return QImage(
        image.data,
        image.width,
        image.height,
        image.bytes_per_line,
        _QIMAGE_FORMATS[image.mode],
    )


def _checker_brush() -> QBrush:
    tile = QPixmap(CHECKER_SIZE * 2, CHECKER_SIZE * 2)
    tile.fill(CHECKER_COLORS[0])
    painter = QPainter(tile)
    painter.fillRect(0, 0, CHECKER_SIZE, CHECKER_SIZE, CHECKER_COLORS[1])
    painter.fillRect(CHECKER_SIZE, CHECKER_SIZE, CHECKER_SIZE, CHECKER_SIZE, CHECKER_COLORS[1])
    painter.end()
    return QBrush(tile)


class ImageCanvas(QWidget):
    """Draws one image fitted to the widget, or a message when there is no image.

    The image is fitted using physical (device) pixels: on a High DPI screen an image is never
    shown larger than one image pixel per device pixel, and small images aren't enlarged.
    Transparent areas are drawn over a checkerboard; nothing is composited into the image.
    """

    target_size_changed = Signal()
    """Emitted when the physical size available for the image changes (resize or DPI)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(160, 120)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        # `_qimage` reads from `_image.data`, so the two are always replaced together.
        self._image: DecodedImage | None = None
        self._qimage: QImage | None = None
        self._scaled: QPixmap | None = None
        self._scaled_key: tuple[int, int, float] | None = None
        self._message = ""
        self._upscale = False
        self._checker = _checker_brush()
        self._last_target: tuple[int, int] | None = None

    @property
    def image(self) -> DecodedImage | None:
        return self._image

    @property
    def message(self) -> str:
        return self._message

    def set_image(self, image: DecodedImage | None) -> None:
        if image is self._image:
            return
        self._image = image
        self._qimage = to_qimage(image) if image is not None else None
        self._scaled = None
        self._scaled_key = None
        if image is not None:
            self._message = ""
        self.update()

    def set_message(self, message: str) -> None:
        """Show a message instead of the image."""
        self.set_image(None)
        self._message = message
        self.update()

    def clear(self) -> None:
        self.set_message("")

    def target_size(self) -> tuple[int, int]:
        """The area available for the image, in physical pixels."""
        dpr = self.devicePixelRatioF()
        return (
            max(1, math.ceil(self.width() * dpr)),
            max(1, math.ceil(self.height() * dpr)),
        )

    def display_size(self) -> tuple[int, int] | None:
        """The size the current image is drawn at, in physical pixels."""
        if self._image is None:
            return None
        source = (self._image.source_width, self._image.source_height)
        return fit_size(source, self.target_size(), upscale=self._upscale)

    def image_rect(self) -> QRectF | None:
        """Where the current image is drawn, in logical (widget) coordinates."""
        size = self.display_size()
        if size is None:
            return None
        # Align the image to whole device pixels so it isn't resampled again when drawn.
        dpr = self.devicePixelRatioF()
        target = self.target_size()
        left, top = (target[0] - size[0]) // 2, (target[1] - size[1]) // 2
        return QRectF(left / dpr, top / dpr, size[0] / dpr, size[1] / dpr)

    def _scaled_pixmap(self) -> QPixmap | None:
        if self._qimage is None:
            return None
        size = self.display_size()
        assert size is not None
        dpr = self.devicePixelRatioF()
        key = (size[0], size[1], dpr)
        if self._scaled is None or self._scaled_key != key:
            if (self._qimage.width(), self._qimage.height()) == size:
                pixmap = QPixmap.fromImage(self._qimage)
            else:
                pixmap = QPixmap.fromImage(
                    self._qimage.scaled(
                        size[0],
                        size[1],
                        Qt.AspectRatioMode.IgnoreAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            pixmap.setDevicePixelRatio(dpr)
            self._scaled = pixmap
            self._scaled_key = key
        return self._scaled

    @override
    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), BACKGROUND_COLOR)
        pixmap = self._scaled_pixmap()
        rect = self.image_rect()
        if pixmap is not None and rect is not None and self._image is not None:
            if self._image.has_alpha:
                painter.fillRect(rect, self._checker)
            painter.drawPixmap(rect.topLeft(), pixmap)
        elif self._message:
            painter.setPen(MESSAGE_COLOR)
            painter.drawText(
                self.rect().adjusted(16, 16, -16, -16),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self._message,
            )
        painter.end()

    def _check_target_size(self) -> None:
        target = self.target_size()
        if target != self._last_target:
            self._last_target = target
            self.target_size_changed.emit()

    @override
    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._check_target_size()

    @override
    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.DevicePixelRatioChange:
            self._scaled = None
            self._check_target_size()
            self.update()
        return super().event(event)


class ImageViewerView(QWidget):
    """Layout for the image viewer window: the image, with a status line below it."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.Window, on=True)
        self.setMinimumSize(320, 240)

        self.root_layout = QVBoxLayout(self)
        self.root_layout.setContentsMargins(0, 0, 0, 0)
        self.root_layout.setSpacing(0)

        self.canvas = ImageCanvas(self)
        self.root_layout.addWidget(self.canvas, stretch=1)

        self.status_widget = QWidget(self)
        self.status_layout = QHBoxLayout(self.status_widget)
        self.status_layout.setContentsMargins(8, 4, 8, 4)
        self.position_label = QLabel(self.status_widget)
        self.name_label = QLabel(self.status_widget)
        self.name_label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.name_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.details_label = QLabel(self.status_widget)
        self.details_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.status_layout.addWidget(self.position_label)
        self.status_layout.addWidget(self.name_label, stretch=1)
        self.status_layout.addWidget(self.details_label)
        self.root_layout.addWidget(self.status_widget)

        # Keyboard input is handled by the window itself.
        for widget in (self.canvas, self.status_widget, self.position_label, self.name_label):
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.details_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
