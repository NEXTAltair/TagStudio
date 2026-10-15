# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only

# pyright: reportPrivateUsage=false


from collections.abc import Generator
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image
from pytestqt.qtbot import QtBot

from tagstudio.core.library.alchemy.library import Library
from tagstudio.core.library.alchemy.models import Entry
from tagstudio.i18n.translations import Translations
from tagstudio.qt.image_viewer_manager import ImageViewerManager
from tagstudio.qt.qt_driver import QtDriver


@pytest.fixture
def image_library(tmp_path: Path) -> Generator[tuple[Library, list[int]]]:
    names = ["a.png", "b.png", "c.txt", "d.png"]
    for i, name in enumerate(names):
        if name.endswith(".png"):
            Image.new("RGB", (10 + i, 10), (i * 40, 0, 0)).save(tmp_path / name)
        else:
            (tmp_path / name).write_text("text")
    lib = Library()
    assert lib.open_library(tmp_path).success
    ids = list(lib.add_entries([Entry(path=Path(n), fields=[]) for n in names]))
    yield lib, ids
    lib.close()


@pytest.fixture
def driver(qt_driver: QtDriver, image_library) -> Generator[QtDriver]:  # pyright: ignore
    lib, ids = image_library
    qt_driver.lib = lib
    qt_driver.frame_content = list(ids)
    # The fixture's main window is a mock, so the viewer gets no parent window.
    qt_driver.image_viewer = ImageViewerManager(None)
    yield qt_driver
    qt_driver.image_viewer.shutdown()


def test_open_uses_results_and_last_selected(qtbot: QtBot, driver: QtDriver, image_library):  # pyright: ignore
    _, ids = image_library
    driver._selected = dict.fromkeys([ids[0], ids[1]])  # pyright: ignore[reportAttributeAccessIssue]
    driver.open_image_viewer()
    viewer = driver.image_viewer
    assert viewer is not None and viewer.session is not None
    assert viewer.session.current_id == ids[1]
    assert viewer.session.entry_ids == (ids[0], ids[1], ids[3])


def test_viewer_does_not_change_main_window_state(qtbot: QtBot, driver: QtDriver, image_library):  # pyright: ignore
    _, ids = image_library
    driver._selected = dict.fromkeys([ids[0]])  # pyright: ignore[reportAttributeAccessIssue]
    selected = list(driver.selected)
    frame_content = list(driver.frame_content)
    history = deepcopy(driver.browsing_history.current)

    driver.open_image_viewer()
    viewer = driver.image_viewer
    assert viewer is not None and viewer.viewer is not None
    viewer.viewer.navigate.emit(1)
    viewer.viewer.navigate.emit(1)
    viewer.viewer.close()

    assert driver.selected == selected
    assert driver.frame_content == frame_content
    assert driver.browsing_history.current == history
    driver.main_window.preview_panel.set_selection.assert_not_called()  # pyright: ignore[reportAttributeAccessIssue]


def test_background_result_changes_are_not_reverted(qtbot: QtBot, driver: QtDriver, image_library):  # pyright: ignore
    """Changing the results behind the viewer isn't undone when it closes."""
    _, ids = image_library
    driver.open_image_viewer()
    driver.frame_content = [ids[3]]
    driver._selected = dict.fromkeys([ids[3]])  # pyright: ignore[reportAttributeAccessIssue]
    driver.image_viewer.close()  # pyright: ignore[reportOptionalMemberAccess]
    assert driver.frame_content == [ids[3]]
    assert driver.selected == [ids[3]]


def test_no_images_shows_status_message(driver: QtDriver, image_library):  # pyright: ignore
    _, ids = image_library
    driver.frame_content = [ids[2]]
    driver.open_image_viewer()
    driver.main_window.status_bar.showMessage.assert_called_with(  # pyright: ignore[reportAttributeAccessIssue]
        Translations["status.image_viewer.no_images"]
    )


def test_close_library_closes_viewer_first(driver: QtDriver, monkeypatch: pytest.MonkeyPatch):
    driver.open_image_viewer()
    viewer = driver.image_viewer
    assert viewer is not None and viewer.is_open

    calls: list[str] = []
    real_close_library = viewer.close_library
    monkeypatch.setattr(
        viewer, "close_library", lambda: (calls.append("viewer"), real_close_library())
    )
    real_lib_close = driver.lib.close
    monkeypatch.setattr(driver.lib, "close", lambda: (calls.append("library"), real_lib_close()))

    driver.close_library()
    assert calls == ["viewer", "library"]
    assert not viewer.is_open


def test_shutdown_shuts_down_viewer(driver: QtDriver, monkeypatch: pytest.MonkeyPatch):
    driver.open_image_viewer()
    viewer = driver.image_viewer
    assert viewer is not None
    monkeypatch.setattr("tagstudio.qt.qt_driver.QApplication", Mock())
    driver.shutdown()
    assert viewer.runner.is_shut_down
    assert not viewer.is_open
