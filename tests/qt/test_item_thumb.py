# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only


from unittest.mock import Mock

import pytest
from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from tagstudio.core.library.alchemy.enums import ItemType
from tagstudio.qt.mixed.item_thumb import BadgeType, ItemThumb
from tagstudio.qt.qt_driver import QtDriver


@pytest.mark.parametrize("new_value", (True, False))
def test_badge_visual_state(qt_driver: QtDriver, entry_min: int, new_value: bool):
    thumb = ItemThumb(
        ItemType.ENTRY, qt_driver.lib, qt_driver, (100, 100), show_filename_label=False
    )

    qt_driver.frame_content = [entry_min]
    qt_driver.toggle_item_selection(0, append=False, bridge=False)

    thumb.badges[BadgeType.FAVORITE].setChecked(new_value)
    assert thumb.badges[BadgeType.FAVORITE].isChecked() == new_value
    # TODO
    # assert thumb.favorite_badge.isHidden() == initial_state
    assert thumb.is_favorite == new_value


def test_double_click_opens_viewer_without_toggling_again(
    qtbot: QtBot, qt_driver: QtDriver, entry_min: int, monkeypatch: pytest.MonkeyPatch
):
    thumb = ItemThumb(
        ItemType.ENTRY, qt_driver.lib, qt_driver, (100, 100), show_filename_label=False
    )
    qtbot.addWidget(thumb)
    thumb.item_id = entry_min
    opened = Mock()
    toggled = Mock()
    monkeypatch.setattr(qt_driver, "open_image_viewer_at", opened)
    monkeypatch.setattr(thumb, "toggle_item_selection", toggled)

    qtbot.mouseDClick(thumb.thumb_button, Qt.MouseButton.LeftButton)

    opened.assert_called_once_with(entry_min)
    toggled.assert_not_called()
