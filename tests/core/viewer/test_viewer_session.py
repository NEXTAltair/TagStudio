# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only

# NOTE: These tests don't use Qt, a database, or an event loop.


import pytest

from tagstudio.core.viewer.session import ViewerSession


def test_empty_session():
    session = ViewerSession([])
    assert session.count == 0
    assert session.index is None
    assert session.current_id is None
    assert session.is_first and session.is_last
    assert not session.move(1)
    assert not session.move_to_first()
    assert not session.move_to_last()


def test_single_entry_session():
    session = ViewerSession([7], start_id=7)
    assert session.current_id == 7
    assert not session.move(1)
    assert not session.move(-1)
    assert session.current_id == 7


@pytest.mark.parametrize(["start_id", "expected"], [(3, 3), (None, 1), (99, 1)])
def test_start_entry(start_id: int | None, expected: int):
    session = ViewerSession([1, 2, 3, 4], start_id=start_id)
    assert session.current_id == expected


def test_navigation_stops_at_both_ends():
    session = ViewerSession([10, 20, 30], start_id=20)
    assert session.move(1)
    assert session.current_id == 30
    assert session.is_last
    assert not session.move(1)
    assert session.current_id == 30

    assert session.move(-5)  # Clamped to the first entry
    assert session.current_id == 10
    assert session.is_first
    assert not session.move(-1)


def test_back_and_forth_returns_to_same_entry():
    session = ViewerSession([1, 2, 3], start_id=1)
    session.move(1)
    session.move(-1)
    assert session.current_id == 1


def test_first_and_last():
    session = ViewerSession([1, 2, 3, 4], start_id=2)
    assert session.move_to_last()
    assert session.current_id == 4
    assert not session.move_to_last()
    assert session.move_to_first()
    assert session.current_id == 1


def test_duplicates_keep_first_occurrence():
    session = ViewerSession([5, 6, 5, 7, 6])
    assert session.entry_ids == (5, 6, 7)


def test_snapshot_is_copied():
    """Changing the source list (e.g. the search results) doesn't affect the session."""
    frame_content = [1, 2, 3]
    session = ViewerSession(frame_content, start_id=2)

    frame_content.reverse()
    frame_content.append(4)
    assert session.entry_ids == (1, 2, 3)
    assert session.current_id == 2

    frame_content.clear()
    assert session.move(1)
    assert session.current_id == 3


def test_invalidated_session_does_not_move():
    session = ViewerSession([1, 2, 3])
    session.invalidate()
    assert not session.is_active
    assert not session.move(1)
    assert session.current_id == 1


def test_from_snapshot_filters_and_keeps_order():
    session = ViewerSession.from_snapshot(
        [5, 4, 3, 2, 1], [True, False, True, False, True], start_id=3
    )
    assert session.entry_ids == (5, 3, 1)
    assert session.current_id == 3


def test_from_snapshot_start_on_unviewable_uses_next():
    session = ViewerSession.from_snapshot([1, 2, 3, 4], [True, False, False, True], start_id=2)
    assert session.current_id == 4


def test_from_snapshot_start_on_unviewable_at_end_uses_previous():
    session = ViewerSession.from_snapshot([1, 2, 3, 4], [True, True, False, False], start_id=4)
    assert session.current_id == 2


def test_from_snapshot_without_viewable_entries():
    session = ViewerSession.from_snapshot([1, 2], [False, False], start_id=1)
    assert session.count == 0
    assert session.current_id is None


def test_from_snapshot_length_mismatch():
    with pytest.raises(ValueError):
        ViewerSession.from_snapshot([1, 2], [True], start_id=None)
