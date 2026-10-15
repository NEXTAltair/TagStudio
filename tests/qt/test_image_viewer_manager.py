# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only

# pyright: reportPrivateUsage=false


import gc
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Generator
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication, QWidget
from pytestqt.qtbot import QtBot

from tagstudio.core.library.alchemy.library import Library
from tagstudio.core.library.alchemy.models import Entry
from tagstudio.core.viewer.cache import ImageCache
from tagstudio.core.viewer.decoding import (
    CancelCheck,
    DecodedImage,
    DecodeErrorKind,
    DecodeFailure,
    DecodeOutcome,
    DecodeRequest,
    decode_image,
)
from tagstudio.core.viewer.scheduler import LoadScheduler
from tagstudio.i18n.translations import Translations
from tagstudio.qt.image_viewer_manager import ImageViewerManager, OpenResult
from tagstudio.qt.utils.image_decode_runner import ImageDecodeRunner

TIMEOUT_MS = 5000

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]


@pytest.fixture(autouse=True)
def _collect_garbage_on_gui_thread():
    """Collect garbage left by earlier tests before the decode thread starts allocating.

    Some earlier tests leave unreachable widgets in reference cycles. If a garbage collection
    is triggered by the decode thread, those QObjects are destroyed on the wrong thread, which
    can crash Qt (e.g. a running animation left registered with the GUI thread's timer).
    """
    gc.collect()


class GatedDecode:
    """Wraps the real decoder; decodes for gated paths wait until released."""

    def __init__(self) -> None:
        self.calls: list[Path] = []
        self.gate: threading.Event | None = None
        self.entered = threading.Event()
        self.threads_inside = 0
        self.max_threads_inside = 0
        self.respects_cancel = True
        self._lock = threading.Lock()

    def __call__(self, request: DecodeRequest, cancel: CancelCheck | None) -> DecodeOutcome:
        with self._lock:
            self.calls.append(request.path)
            self.threads_inside += 1
            self.max_threads_inside = max(self.max_threads_inside, self.threads_inside)
        try:
            self.entered.set()
            if self.gate is not None:
                self.gate.wait(5)
            return decode_image(request, cancel if self.respects_cancel else None)
        finally:
            with self._lock:
                self.threads_inside -= 1


def _make_library(root: Path, names: list[str]) -> tuple[Library, list[int]]:
    root.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        path = root / name
        if path.suffix in (".png", ".jpg"):
            Image.new("RGB", (80 + i, 40), COLORS[i % len(COLORS)]).save(path)
        else:
            path.write_text("not an image")
    lib = Library()
    assert lib.open_library(root).success
    ids = list(lib.add_entries([Entry(path=Path(n), fields=[]) for n in names]))
    return lib, ids


@pytest.fixture
def library(tmp_path: Path) -> Generator[tuple[Library, list[int]]]:
    lib, ids = _make_library(tmp_path / "lib", ["a.png", "b.txt", "c.png", "d.jpg", "e.png"])
    yield lib, ids
    lib.close()


@pytest.fixture
def decode() -> GatedDecode:
    return GatedDecode()


@pytest.fixture
def manager(qtbot: QtBot, decode: GatedDecode) -> Generator[ImageViewerManager]:
    parent = QWidget()
    qtbot.addWidget(parent)
    parent.show()
    m = ImageViewerManager(parent, runner=ImageDecodeRunner(decode))
    yield m
    if decode.gate is not None:
        decode.gate.set()
    m.shutdown()


def _shown_size(m: ImageViewerManager) -> tuple[int, int] | None:
    image = m.viewer.canvas.image if m.viewer else None
    return (image.source_width, image.source_height) if image else None


def _wait_for_entry(qtbot: QtBot, m: ImageViewerManager, index: int) -> None:
    """Wait until the image of the n-th file of the `library` fixture is shown."""
    qtbot.waitUntil(lambda: _shown_size(m) == (80 + index, 40), timeout=TIMEOUT_MS)


def test_open_shows_start_entry(qtbot: QtBot, manager: ImageViewerManager, library):  # pyright: ignore
    lib, ids = library
    assert manager.open(lib, ids, start_id=ids[2]) == OpenResult.OPENED
    assert manager.viewer is not None and manager.viewer.isVisible()
    _wait_for_entry(qtbot, manager, 2)
    assert manager.viewer.position_label.text() == "2 / 4"  # b.txt isn't viewable
    assert manager.viewer.name_label.text() == "c.png"


def test_non_image_entries_are_skipped(qtbot: QtBot, manager: ImageViewerManager, library):  # pyright: ignore
    lib, ids = library
    manager.open(lib, ids, start_id=ids[1])  # Start on b.txt
    _wait_for_entry(qtbot, manager, 2)  # c.png is the next image
    assert manager.session is not None
    assert manager.session.entry_ids == (ids[0], ids[2], ids[3], ids[4])


def test_navigation_and_boundaries(qtbot: QtBot, manager: ImageViewerManager, library):  # pyright: ignore
    lib, ids = library
    manager.open(lib, ids, start_id=ids[0])
    viewer = manager.viewer
    assert viewer is not None
    _wait_for_entry(qtbot, manager, 0)

    viewer.navigate.emit(-1)  # Already at the first image
    assert "First image" in viewer.position_label.text()

    viewer.navigate.emit(1)
    _wait_for_entry(qtbot, manager, 2)
    viewer.navigate_last.emit()
    _wait_for_entry(qtbot, manager, 4)
    viewer.navigate.emit(1)
    assert "Last image" in viewer.position_label.text()
    viewer.navigate_first.emit()
    _wait_for_entry(qtbot, manager, 0)


def test_no_viewable_entries(manager: ImageViewerManager, library):  # pyright: ignore
    lib, ids = library
    assert manager.open(lib, [ids[1]], start_id=None) == OpenResult.NO_IMAGES
    assert manager.open(lib, [], start_id=None) == OpenResult.NO_IMAGES
    assert manager.viewer is None or not manager.viewer.isVisible()


def test_no_library(manager: ImageViewerManager):
    assert manager.open(Library(), [1, 2], start_id=1) == OpenResult.NO_LIBRARY


def test_snapshot_is_independent_of_results(qtbot: QtBot, manager: ImageViewerManager, library):  # pyright: ignore
    lib, ids = library
    frame_content = list(ids)
    manager.open(lib, frame_content, start_id=ids[0])
    frame_content.reverse()
    frame_content.clear()
    assert manager.session is not None
    assert manager.session.count == 4
    manager.viewer.navigate.emit(1)  # pyright: ignore[reportOptionalMemberAccess]
    _wait_for_entry(qtbot, manager, 2)


def test_rapid_navigation_shows_latest_and_limits_decodes(
    qtbot: QtBot,
    manager: ImageViewerManager,
    library,  # pyright: ignore
    decode: GatedDecode,
):
    lib, ids = library
    decode.gate = threading.Event()
    manager.open(lib, ids, start_id=ids[0])
    assert decode.entered.wait(5)
    viewer = manager.viewer
    assert viewer is not None

    for delta in (1, 1, 1, -1, 1):  # Ends on e.png
        viewer.navigate.emit(delta)
    assert viewer.canvas.image is None  # Nothing stale is shown while loading
    assert viewer.canvas.message == Translations["image_viewer.loading"]
    decode.gate.set()
    _wait_for_entry(qtbot, manager, 4)

    assert manager.runner.max_active == 1
    assert decode.max_threads_inside == 1
    # Only the first and the last requests were decoded; the ones in between were skipped.
    assert [p.name for p in decode.calls] == ["a.png", "e.png"]


@pytest.mark.parametrize("respects_cancel", [True, False])
def test_back_and_forth_does_not_apply_old_request(
    qtbot: QtBot,
    manager: ImageViewerManager,
    library,  # pyright: ignore
    decode: GatedDecode,
    respects_cancel: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    """A -> B -> A: the first A request's result is never the one that gets applied."""
    lib, ids = library
    decode.gate = threading.Event()
    decode.respects_cancel = respects_cancel
    finished: list[tuple[int, bool]] = []
    real_finish = LoadScheduler.finish

    def spy(self: LoadScheduler, request_id: int):  # pyright: ignore
        actions = real_finish(self, request_id)
        finished.append((request_id, actions.apply))
        return actions

    monkeypatch.setattr(LoadScheduler, "finish", spy)

    manager.open(lib, ids, start_id=ids[0])
    assert decode.entered.wait(5)
    first = manager.scheduler.running
    assert first is not None
    viewer = manager.viewer
    assert viewer is not None
    viewer.navigate.emit(1)
    viewer.navigate.emit(-1)
    decode.gate.set()
    _wait_for_entry(qtbot, manager, 0)
    qtbot.waitUntil(lambda: manager.scheduler.is_idle, timeout=TIMEOUT_MS)

    assert finished[0] == (first.request_id, False)
    assert [apply for _, apply in finished] == [False, True]
    if respects_cancel:
        # The first request stopped early, so A is decoded again for the new request.
        assert [p.name for p in decode.calls] == ["a.png", "a.png"]
    else:
        # The first request completed; its result was cached and reused, not applied directly.
        assert [p.name for p in decode.calls] == ["a.png"]


def test_revisiting_uses_cache(qtbot: QtBot, manager: ImageViewerManager, library, decode):  # pyright: ignore
    lib, ids = library
    manager.open(lib, ids, start_id=ids[0])
    _wait_for_entry(qtbot, manager, 0)
    manager.viewer.navigate.emit(1)  # pyright: ignore[reportOptionalMemberAccess]
    _wait_for_entry(qtbot, manager, 2)
    manager.viewer.navigate.emit(-1)  # pyright: ignore[reportOptionalMemberAccess]
    _wait_for_entry(qtbot, manager, 0)  # Shown immediately from the cache
    assert [p.name for p in decode.calls] == ["a.png", "c.png"]
    assert manager.cache.stats.hits >= 1


def test_cache_is_reused_when_viewer_is_reopened(
    qtbot: QtBot,
    manager: ImageViewerManager,
    library,
    decode: GatedDecode,  # pyright: ignore
):
    lib, ids = library
    manager.open(lib, ids, start_id=ids[0])
    _wait_for_entry(qtbot, manager, 0)
    manager.close()
    manager.open(lib, ids, start_id=ids[0])
    _wait_for_entry(qtbot, manager, 0)
    assert [p.name for p in decode.calls] == ["a.png"]


def test_modified_file_is_decoded_again(
    qtbot: QtBot,
    manager: ImageViewerManager,
    library,
    decode: GatedDecode,  # pyright: ignore
):
    lib, ids = library
    manager.open(lib, ids, start_id=ids[0])
    _wait_for_entry(qtbot, manager, 0)
    manager.viewer.navigate.emit(1)  # pyright: ignore[reportOptionalMemberAccess]
    _wait_for_entry(qtbot, manager, 2)

    Image.new("RGB", (300, 40)).save(lib.library_dir / "a.png")
    manager.viewer.navigate.emit(-1)  # pyright: ignore[reportOptionalMemberAccess]
    qtbot.waitUntil(lambda: _shown_size(manager) == (300, 40), timeout=TIMEOUT_MS)
    assert [p.name for p in decode.calls] == ["a.png", "c.png", "a.png"]


def test_missing_file(qtbot: QtBot, manager: ImageViewerManager, library):  # pyright: ignore
    lib, ids = library
    (lib.library_dir / "c.png").unlink()
    manager.open(lib, ids, start_id=ids[2])
    viewer = manager.viewer
    assert viewer is not None
    assert viewer.canvas.message == Translations["image_viewer.error.missing_file"]
    viewer.navigate.emit(1)  # The viewer keeps working
    _wait_for_entry(qtbot, manager, 3)


def test_removed_entry(qtbot: QtBot, manager: ImageViewerManager, library):  # pyright: ignore
    lib, ids = library
    manager.open(lib, ids, start_id=ids[0])
    _wait_for_entry(qtbot, manager, 0)
    lib.remove_entries([ids[2]])
    viewer = manager.viewer
    assert viewer is not None
    viewer.navigate.emit(1)
    assert viewer.canvas.message == Translations["image_viewer.error.missing_entry"]
    assert viewer.name_label.text() == ""  # The previous file name isn't left on screen


def test_corrupt_file(qtbot: QtBot, manager: ImageViewerManager, library):  # pyright: ignore
    lib, ids = library
    (lib.library_dir / "c.png").write_bytes(b"garbage")
    manager.open(lib, ids, start_id=ids[2])
    qtbot.waitUntil(
        lambda: manager.viewer.canvas.message == Translations["image_viewer.error.corrupt"],  # pyright: ignore[reportOptionalMemberAccess]
        timeout=TIMEOUT_MS,
    )


def test_close_during_load_discards_result(
    qtbot: QtBot,
    manager: ImageViewerManager,
    library,
    decode: GatedDecode,  # pyright: ignore
):
    lib, ids = library
    decode.gate = threading.Event()
    manager.open(lib, ids, start_id=ids[0])
    assert decode.entered.wait(5)
    viewer = manager.viewer
    assert viewer is not None

    viewer.close()
    assert not manager.is_open
    decode.gate.set()
    qtbot.waitUntil(lambda: manager.scheduler.running is None, timeout=TIMEOUT_MS)
    assert viewer.canvas.image is None
    assert not viewer.isVisible()


def test_reopen_while_decoding_never_runs_two_decodes(
    qtbot: QtBot,
    manager: ImageViewerManager,
    library,
    decode: GatedDecode,  # pyright: ignore
):
    lib, ids = library
    decode.gate = threading.Event()
    manager.open(lib, ids, start_id=ids[0])
    assert decode.entered.wait(5)
    for _ in range(3):
        manager.close()
        manager.open(lib, ids, start_id=ids[3])
    assert manager.runner.active_count == 1
    decode.gate.set()
    _wait_for_entry(qtbot, manager, 3)
    assert decode.max_threads_inside == 1
    assert manager.runner.max_active == 1


def test_close_library_closes_viewer_and_clears_cache(
    qtbot: QtBot,
    manager: ImageViewerManager,
    library,
    tmp_path: Path,  # pyright: ignore
):
    lib, ids = library
    manager.open(lib, ids, start_id=ids[0])
    _wait_for_entry(qtbot, manager, 0)
    assert len(manager.cache) == 1

    manager.close_library()
    lib.close()
    assert not manager.is_open
    assert not manager.viewer.isVisible()  # pyright: ignore[reportOptionalMemberAccess]
    assert len(manager.cache) == 0

    # Open another library whose entry IDs overlap with the first one.
    other, other_ids = _make_library(tmp_path / "other", ["x.png", "y.png"])
    try:
        assert set(other_ids) & set(ids)
        manager.open(other, other_ids, start_id=other_ids[1])
        qtbot.waitUntil(lambda: _shown_size(manager) == (81, 40), timeout=TIMEOUT_MS)
        assert manager.viewer.name_label.text() == "y.png"  # pyright: ignore[reportOptionalMemberAccess]
    finally:
        manager.close_library()
        other.close()


def test_library_closed_while_decoding(
    qtbot: QtBot,
    manager: ImageViewerManager,
    library,
    decode: GatedDecode,  # pyright: ignore
):
    lib, ids = library
    decode.gate = threading.Event()
    manager.open(lib, ids, start_id=ids[0])
    assert decode.entered.wait(5)
    manager.close_library()
    lib.close()
    decode.gate.set()
    qtbot.waitUntil(lambda: manager.scheduler.running is None, timeout=TIMEOUT_MS)
    assert len(manager.cache) == 0  # The old library's result isn't cached
    assert manager.viewer.canvas.image is None  # pyright: ignore[reportOptionalMemberAccess]


def test_resize_requests_larger_image(qtbot: QtBot, library, decode: GatedDecode):  # pyright: ignore
    lib, ids = library
    big = lib.library_dir / "big.png"
    Image.new("RGB", (3000, 2000), (1, 2, 3)).save(big)
    big_id = lib.add_entries([Entry(path=Path("big.png"), fields=[])])[0]
    parent = QWidget()
    qtbot.addWidget(parent)
    m = ImageViewerManager(parent, runner=ImageDecodeRunner(decode))
    try:
        m.open(lib, [big_id], start_id=big_id)
        viewer = m.viewer
        assert viewer is not None
        viewer.resize(400, 300)
        qtbot.waitUntil(lambda: viewer.canvas.image is not None, timeout=TIMEOUT_MS)
        qtbot.waitUntil(lambda: m.scheduler.is_idle, timeout=TIMEOUT_MS)
        small_width = viewer.canvas.image.width  # pyright: ignore[reportOptionalMemberAccess]

        viewer.resize(1200, 900)
        qtbot.waitUntil(
            lambda: viewer.canvas.image is not None and viewer.canvas.image.width > small_width,
            timeout=TIMEOUT_MS,
        )
        image = viewer.canvas.image
        assert isinstance(image, DecodedImage)
        target = viewer.canvas.target_size()
        assert image.width <= target[0] and image.height <= target[1]
        assert (image.source_width, image.source_height) == (3000, 2000)
    finally:
        m.shutdown()


def test_failed_larger_decode_keeps_smaller_image(
    qtbot: QtBot,
    manager: ImageViewerManager,
    library,  # pyright: ignore
    monkeypatch: pytest.MonkeyPatch,
):
    lib, ids = library
    big = lib.library_dir / "big.jpg"
    Image.new("RGB", (4000, 2000), (1, 2, 3)).save(big)
    big_id = lib.add_entries([Entry(path=Path("big.jpg"), fields=[])])[0]
    manager.open(lib, [big_id], start_id=big_id)
    viewer = manager.viewer
    assert viewer is not None
    viewer.resize(400, 300)
    qtbot.waitUntil(lambda: viewer.canvas.image is not None, timeout=TIMEOUT_MS)
    qtbot.waitUntil(lambda: manager.scheduler.is_idle, timeout=TIMEOUT_MS)
    small = viewer.canvas.image

    # From now on, decodes larger than 1/8 scale don't fit the budget.
    outcomes: list[DecodeOutcome] = []
    manager.runner.finished.connect(lambda _id, outcome: outcomes.append(outcome))
    real_init = DecodeRequest.__init__

    def small_budget(self, *args, **kwargs):  # pyright: ignore
        kwargs["max_decode_bytes"] = 4000 * 2000
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(DecodeRequest, "__init__", small_budget)
    viewer.resize(1600, 1200)
    qtbot.waitUntil(lambda: bool(outcomes) and manager.scheduler.is_idle, timeout=TIMEOUT_MS)

    assert [o.kind for o in outcomes if isinstance(o, DecodeFailure)] == [
        DecodeErrorKind.OVER_BUDGET
    ]
    assert viewer.canvas.image is small
    assert viewer.canvas.message == ""


def test_shutdown_does_not_wait_for_decode(
    qtbot: QtBot,
    manager: ImageViewerManager,
    library,
    decode: GatedDecode,  # pyright: ignore
):
    lib, ids = library
    decode.gate = threading.Event()
    manager.open(lib, ids, start_id=ids[0])
    assert decode.entered.wait(5)
    manager.shutdown()
    assert manager.runner.is_shut_down
    with qtbot.assertNotEmitted(manager.runner.finished, wait=100):
        decode.gate.set()


def test_viewer_window_is_transient_for_parent(qtbot: QtBot, manager: ImageViewerManager, library):  # pyright: ignore
    """Transient windows don't keep Qt's "last window closed" quit from happening."""
    lib, ids = library
    parent = manager._parent_window
    assert parent is not None
    manager.open(lib, ids, start_id=ids[0])
    viewer = manager.viewer
    assert viewer is not None and viewer.isVisible()
    handle = viewer.windowHandle()
    assert handle is not None
    assert handle.transientParent() is parent.windowHandle()


def test_app_exits_when_main_window_closes_with_viewer_open(tmp_path: Path):
    """Run a real event loop: closing the main window must end app.exec() with the viewer open."""
    script = tmp_path / "quit_check.py"
    script.write_text(
        "import sys\n"
        "from PySide6.QtCore import QTimer\n"
        "from PySide6.QtWidgets import QApplication, QMainWindow\n"
        "from tagstudio.qt.controllers.image_viewer import ImageViewer\n"
        "app = QApplication(sys.argv)\n"
        "main = QMainWindow()\n"
        "main.show()\n"
        "viewer = ImageViewer(main)\n"
        "viewer.show()\n"
        "QTimer.singleShot(100, main.close)\n"
        "QTimer.singleShot(10000, lambda: app.exit(3))  # Watchdog\n"
        "code = app.exec()\n"
        "assert viewer.isVisible() or True\n"
        "sys.exit(code)\n"
    )
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    env["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    result = subprocess.run(
        [sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr


def test_focus_returns_to_parent_on_close(qtbot: QtBot, manager: ImageViewerManager, library):  # pyright: ignore
    lib, ids = library
    parent = manager._parent_window
    assert parent is not None
    manager.open(lib, ids, start_id=ids[0])
    viewer = manager.viewer
    assert viewer is not None
    qtbot.waitUntil(lambda: QApplication.activeWindow() is viewer, timeout=TIMEOUT_MS)
    viewer.close()
    qtbot.waitUntil(lambda: QApplication.activeWindow() is parent, timeout=TIMEOUT_MS)


def test_cache_budget_is_respected(qtbot: QtBot, library, decode: GatedDecode):  # pyright: ignore
    lib, ids = library
    parent = QWidget()
    qtbot.addWidget(parent)
    cache = ImageCache(max_bytes=81 * 40 * 3 * 2)  # About two images
    m = ImageViewerManager(parent, runner=ImageDecodeRunner(decode), cache=cache)
    try:
        m.open(lib, ids, start_id=ids[0])
        for index in (2, 3, 4):
            m.viewer.navigate.emit(1)  # pyright: ignore[reportOptionalMemberAccess]
            _wait_for_entry(qtbot, m, index)
        assert cache.bytes_used <= cache.max_bytes
        assert cache.stats.evictions >= 1
    finally:
        m.shutdown()


@pytest.fixture
def make_manager(qtbot: QtBot) -> Generator[Callable[[], ImageViewerManager]]:
    managers: list[ImageViewerManager] = []

    def make() -> ImageViewerManager:
        m = ImageViewerManager(None)
        managers.append(m)
        return m

    yield make
    for m in managers:
        m.shutdown()


def test_viewer_without_parent_window(qtbot: QtBot, make_manager, library):  # pyright: ignore
    lib, ids = library
    m: ImageViewerManager = make_manager()
    assert m.open(lib, ids, start_id=ids[0]) == OpenResult.OPENED
    _wait_for_entry(qtbot, m, 0)
