# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only


import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QAction, QKeySequence, QWheelEvent
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
from pytestqt.qtbot import QtBot

from tagstudio.core.viewer.decoding import DecodedImage
from tagstudio.qt.controllers.image_viewer import ImageViewer, WheelStepper
from tagstudio.qt.views.image_viewer_view import ImageCanvas


def _image(width: int, height: int, mode: str = "RGB") -> DecodedImage:
    channels = {"L": 1, "RGB": 3, "RGBA": 4}[mode]
    return DecodedImage(
        data=bytes([128]) * (width * height * channels),
        width=width,
        height=height,
        mode=mode,  # pyright: ignore[reportArgumentType]
        source_width=width,
        source_height=height,
        source_format="PNG",
    )


def _wheel(
    widget: QWidget,
    delta_y: int,
    delta_x: int = 0,
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    pixel_delta_y: int = 0,
) -> None:
    pos = QPointF(widget.width() / 2, widget.height() / 2)
    event = QWheelEvent(
        pos,
        widget.mapToGlobal(pos),
        QPoint(0, pixel_delta_y),
        QPoint(delta_x, delta_y),
        Qt.MouseButton.NoButton,
        modifiers,
        Qt.ScrollPhase.NoScrollPhase,
        False,  # noqa: FBT003
    )
    QApplication.sendEvent(widget, event)


@pytest.fixture
def viewer(qtbot: QtBot) -> ImageViewer:
    v = ImageViewer()
    qtbot.addWidget(v)
    v.resize(400, 300)
    v.show()
    qtbot.waitExposed(v)
    return v


# WheelStepper =================================================================


def test_wheel_stepper_full_notches():
    stepper = WheelStepper()
    assert stepper.feed(-120) == 1  # Wheel down: next
    assert stepper.feed(120) == -1  # Wheel up: previous
    assert stepper.feed(-360) == 3


def test_wheel_stepper_accumulates_small_deltas():
    stepper = WheelStepper()
    assert [stepper.feed(-40) for _ in range(6)] == [0, 0, 1, 0, 0, 1]


def test_wheel_stepper_direction_change_discards_partial_delta():
    stepper = WheelStepper()
    assert stepper.feed(-100) == 0
    assert stepper.feed(100) == 0
    assert stepper.feed(20) == -1


def test_wheel_stepper_zero():
    assert WheelStepper().feed(0) == 0


# Wheel input ==================================================================


def test_wheel_navigates(qtbot: QtBot, viewer: ImageViewer):
    steps: list[int] = []
    viewer.navigate.connect(steps.append)
    _wheel(viewer.canvas, -120)
    _wheel(viewer.canvas, 120)
    _wheel(viewer.canvas, -240)
    assert steps == [1, -1, 2]


def test_wheel_on_status_bar_navigates(viewer: ImageViewer):
    steps: list[int] = []
    viewer.navigate.connect(steps.append)
    _wheel(viewer.name_label, -120)
    assert steps == [1]


def test_wheel_high_resolution_deltas(viewer: ImageViewer):
    steps: list[int] = []
    viewer.navigate.connect(steps.append)
    for _ in range(5):
        _wheel(viewer.canvas, -30)
    assert steps == [1]


def test_pixel_only_touchpad_deltas(viewer: ImageViewer):
    """Touchpads that report pixel deltas without angle deltas still navigate."""
    steps: list[int] = []
    viewer.navigate.connect(steps.append)
    for _ in range(4):
        _wheel(viewer.canvas, 0, pixel_delta_y=-40)
    assert steps == [1]
    _wheel(viewer.canvas, 0, pixel_delta_y=130)
    assert steps == [1, -1]


def test_angle_delta_takes_precedence_over_pixel_delta(viewer: ImageViewer):
    steps: list[int] = []
    viewer.navigate.connect(steps.append)
    _wheel(viewer.canvas, -120, pixel_delta_y=-1000)
    assert steps == [1]


@pytest.mark.parametrize(
    "modifier",
    [
        Qt.KeyboardModifier.ControlModifier,
        Qt.KeyboardModifier.ShiftModifier,
        Qt.KeyboardModifier.AltModifier,
    ],
)
def test_wheel_with_modifiers_is_ignored(viewer: ImageViewer, modifier: Qt.KeyboardModifier):
    steps: list[int] = []
    viewer.navigate.connect(steps.append)
    _wheel(viewer.canvas, -120, modifiers=modifier)
    assert steps == []


def test_horizontal_and_zero_wheel_is_ignored(viewer: ImageViewer):
    steps: list[int] = []
    viewer.navigate.connect(steps.append)
    _wheel(viewer.canvas, 0, delta_x=-120)
    _wheel(viewer.canvas, 0)
    assert steps == []


# Keys =========================================================================


@pytest.mark.parametrize(
    ["key", "expected"],
    [
        (Qt.Key.Key_Right, 1),
        (Qt.Key.Key_Down, 1),
        (Qt.Key.Key_PageDown, 1),
        (Qt.Key.Key_Left, -1),
        (Qt.Key.Key_Up, -1),
        (Qt.Key.Key_PageUp, -1),
    ],
)
def test_navigation_keys(qtbot: QtBot, viewer: ImageViewer, key: Qt.Key, expected: int):
    steps: list[int] = []
    viewer.navigate.connect(steps.append)
    qtbot.keyClick(viewer, key)
    assert steps == [expected]


def test_home_and_end(qtbot: QtBot, viewer: ImageViewer):
    with qtbot.waitSignal(viewer.navigate_first, timeout=1000):
        qtbot.keyClick(viewer, Qt.Key.Key_Home)
    with qtbot.waitSignal(viewer.navigate_last, timeout=1000):
        qtbot.keyClick(viewer, Qt.Key.Key_End)


def test_escape_closes_viewer(qtbot: QtBot, viewer: ImageViewer):
    with qtbot.waitSignal(viewer.closed, timeout=1000):
        qtbot.keyClick(viewer, Qt.Key.Key_Escape)
    assert not viewer.isVisible()


def test_keys_with_modifiers_do_nothing(qtbot: QtBot, viewer: ImageViewer):
    steps: list[int] = []
    viewer.navigate.connect(steps.append)
    qtbot.keyClick(viewer, Qt.Key.Key_Right, Qt.KeyboardModifier.ControlModifier)
    assert steps == []


APP_PASTE = "Ctrl+Alt+F11"
APP_TAG = "Ctrl+Alt+F12"


class ParentWindow(QMainWindow):
    """Stands in for the main window: clears its "selection" on Esc, like MainWindow."""

    def __init__(self):
        super().__init__()
        self.escape_presses = 0
        self.triggered: list[str] = []
        # NOTE: The application-wide shortcuts use sequences no other test registers, so leftover
        # widgets from other tests can't make them ambiguous.
        for name, shortcut, context in (
            ("delete", QKeySequence(Qt.Key.Key_Delete), Qt.ShortcutContext.WindowShortcut),
            ("paste", QKeySequence(APP_PASTE), Qt.ShortcutContext.ApplicationShortcut),
            ("tag", QKeySequence(APP_TAG), Qt.ShortcutContext.ApplicationShortcut),
        ):
            action = QAction(name, self)
            action.setShortcut(shortcut)
            action.setShortcutContext(context)
            action.triggered.connect(lambda checked=False, n=name: self.triggered.append(n))
            self.addAction(action)

    def keyPressEvent(self, event):  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.escape_presses += 1
        super().keyPressEvent(event)


def _press_editing_keys(qtbot: QtBot, widget: QWidget) -> None:
    ctrl_alt = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier
    qtbot.keyClick(widget, Qt.Key.Key_Delete)
    qtbot.keyClick(widget, Qt.Key.Key_F11, ctrl_alt)
    qtbot.keyClick(widget, Qt.Key.Key_F12, ctrl_alt)


def _activate(qtbot: QtBot, widget: QWidget) -> None:
    widget.show()
    widget.activateWindow()
    widget.setFocus()
    qtbot.waitUntil(lambda: QApplication.activeWindow() is widget, timeout=2000)


def test_editing_shortcuts_do_not_reach_main_window(qtbot: QtBot):
    parent = ParentWindow()
    qtbot.addWidget(parent)
    parent.show()

    # Control: an ordinary child window lets application shortcuts through.
    other = QWidget(parent)
    other.setWindowFlag(Qt.WindowType.Window)
    other.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    _activate(qtbot, other)
    _press_editing_keys(qtbot, other)
    assert parent.triggered == ["paste", "tag"]
    other.close()
    parent.triggered.clear()

    viewer = ImageViewer(parent)
    _activate(qtbot, viewer)
    _press_editing_keys(qtbot, viewer)
    qtbot.keyClick(viewer, Qt.Key.Key_Escape)

    assert parent.triggered == []
    assert parent.escape_presses == 0
    assert not viewer.isVisible()


def test_viewer_is_not_a_primary_window():
    """Closing the main window must quit the app even if the viewer is still open."""
    parent = QWidget()
    viewer = ImageViewer(parent)
    assert viewer.isWindow()
    assert viewer.parentWidget() is parent
    viewer.deleteLater()
    parent.deleteLater()


# Display ======================================================================


def test_show_image_and_details(viewer: ImageViewer):
    viewer.show_position(2, 10)
    viewer.show_file_name("cat.png")
    viewer.show_image(_image(40, 20))
    assert viewer.position_label.text() == "3 / 10"
    assert viewer.name_label.text() == "cat.png"
    assert "cat.png" in viewer.windowTitle()
    assert "40" in viewer.details_label.text() and "20" in viewer.details_label.text()
    assert viewer.canvas.image is not None


def test_file_name_is_plain_text(viewer: ImageViewer):
    viewer.show_file_name("<b>photo</b>.png")
    assert viewer.name_label.textFormat() == Qt.TextFormat.PlainText
    assert viewer.name_label.text() == "<b>photo</b>.png"
    assert "&lt;b&gt;photo&lt;/b&gt;.png" in viewer.name_label.toolTip()


def test_message_replaces_previous_image(viewer: ImageViewer):
    viewer.show_image(_image(40, 20))
    viewer.show_message("Loading")
    assert viewer.canvas.image is None
    assert viewer.canvas.message == "Loading"
    assert viewer.details_label.text() == ""


def test_boundary_hint(viewer: ImageViewer):
    viewer.show_position(9, 10)
    viewer.show_boundary(at_end=True)
    assert viewer.position_label.text().startswith("10 / 10")
    assert viewer.position_label.text() != "10 / 10"


def test_clear(viewer: ImageViewer):
    viewer.show_position(0, 1)
    viewer.show_file_name("a.png")
    viewer.show_image(_image(4, 4))
    viewer.clear()
    assert viewer.canvas.image is None
    assert viewer.position_label.text() == ""
    assert viewer.name_label.text() == ""


# Canvas =======================================================================


@pytest.fixture
def canvas(qtbot: QtBot) -> ImageCanvas:
    c = ImageCanvas()
    qtbot.addWidget(c)
    c.resize(400, 300)
    c.show()
    qtbot.waitExposed(c)
    return c


@pytest.mark.parametrize(
    ["size", "expected"],
    [
        ((800, 400), (400, 200)),  # Landscape, shrunk to the width
        ((300, 900), (100, 300)),  # Portrait, shrunk to the height
        ((40, 20), (40, 20)),  # Small images aren't enlarged
    ],
)
def test_canvas_fits_and_centers(
    canvas: ImageCanvas, size: tuple[int, int], expected: tuple[int, int]
):
    canvas.set_image(_image(*size))
    assert canvas.display_size() == expected
    rect = canvas.image_rect()
    assert rect is not None
    assert (rect.width(), rect.height()) == expected
    assert rect.center().x() == pytest.approx(200, abs=1)
    assert rect.center().y() == pytest.approx(150, abs=1)


def test_canvas_uses_source_size_for_reduced_images(canvas: ImageCanvas):
    """A reduced decode is shown at the size of the full image, not the reduced buffer."""
    reduced = DecodedImage(
        data=bytes(100 * 50 * 3),
        width=100,
        height=50,
        mode="RGB",
        source_width=4000,
        source_height=2000,
        source_format="JPEG",
    )
    canvas.set_image(reduced)
    assert canvas.display_size() == (400, 200)


def test_canvas_high_dpi(canvas: ImageCanvas, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(canvas, "devicePixelRatioF", lambda: 2.0)
    assert canvas.target_size() == (800, 600)
    canvas.set_image(_image(1000, 500))
    assert canvas.display_size() == (800, 400)  # Physical pixels
    rect = canvas.image_rect()
    assert rect is not None
    assert (rect.width(), rect.height()) == (400, 200)  # Logical pixels
    canvas.grab()


def test_canvas_upscale(canvas: ImageCanvas):
    canvas.set_image(_image(40, 20))
    assert canvas.display_size() == (40, 20)
    canvas.set_upscale(True)
    assert canvas.display_size() == (400, 200)


def test_canvas_resize_emits_target_size_changed(qtbot: QtBot, canvas: ImageCanvas):
    with qtbot.waitSignal(canvas.target_size_changed, timeout=1000):
        canvas.resize(500, 400)
    assert canvas.target_size() == (500, 400)


@pytest.mark.parametrize(["mode", "expected"], [("L", 128), ("RGB", 128), ("RGBA", None)])
def test_canvas_paints_all_modes(canvas: ImageCanvas, mode: str, expected: int | None):
    canvas.set_image(_image(30, 20, mode))
    pixmap = canvas.grab()
    assert not pixmap.isNull()
    center = pixmap.toImage().pixelColor(200, 150)
    if expected is None:
        # Half transparent: blended with the checkerboard, not with a baked-in background.
        assert center.red() != 128
    else:
        assert center.red() == pytest.approx(expected, abs=2)


def test_canvas_message(canvas: ImageCanvas):
    canvas.set_image(_image(30, 20))
    canvas.set_message("File not found.")
    assert canvas.image is None
    assert canvas.display_size() is None
    assert canvas.image_rect() is None
    assert not canvas.grab().isNull()
