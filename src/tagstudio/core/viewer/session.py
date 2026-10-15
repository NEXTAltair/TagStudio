# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only

"""The image viewer's browsing position, kept separate from the library view's selection."""

from collections.abc import Iterable, Sequence


class ViewerSession:
    """An ordered snapshot of entry IDs and the viewer's current position within it.

    The IDs are copied when the session is created, so later changes to the search results, the
    selection, the page, or the browsing history don't move or reorder the viewer. The session
    never writes to the library or to any UI state.

    Navigation stops at the first and last entries (it doesn't wrap around).
    """

    def __init__(self, entry_ids: Iterable[int], start_id: int | None = None) -> None:
        """Create a session.

        Args:
            entry_ids: The entry IDs to browse, in display order. Duplicates are dropped,
                keeping the first occurrence.
            start_id: The entry to show first. If it's not one of the entry IDs, the first
                entry is shown instead.
        """
        self._ids: tuple[int, ...] = tuple(dict.fromkeys(entry_ids))
        self._index: int | None = None
        self._active = True
        if self._ids:
            self._index = self._ids.index(start_id) if start_id in self._ids else 0

    @classmethod
    def from_snapshot(
        cls, snapshot: Sequence[int], viewable: Sequence[bool], start_id: int | None
    ) -> ViewerSession:
        """Create a session with only the viewable entries of a snapshot.

        If the start entry isn't viewable, the session starts at the next viewable entry after
        it, or else at the closest viewable entry before it.

        Args:
            snapshot: All entry IDs, in display order.
            viewable: Whether each entry of the snapshot can be shown by the viewer.
            start_id: The entry that was requested to be shown first.
        """
        if len(snapshot) != len(viewable):
            raise ValueError("snapshot and viewable must have the same length")

        kept = [entry_id for entry_id, ok in zip(snapshot, viewable, strict=True) if ok]
        start: int | None = None
        if start_id is not None and start_id in snapshot:
            pos = snapshot.index(start_id)
            after = (e for e, ok in zip(snapshot[pos:], viewable[pos:], strict=True) if ok)
            before = (
                e
                for e, ok in zip(reversed(snapshot[:pos]), reversed(viewable[:pos]), strict=True)
                if ok
            )
            start = next(after, None)
            if start is None:
                start = next(before, None)
        return cls(kept, start)

    @property
    def entry_ids(self) -> tuple[int, ...]:
        return self._ids

    @property
    def count(self) -> int:
        return len(self._ids)

    @property
    def index(self) -> int | None:
        """The zero-based position of the current entry, or None if there are no entries."""
        return self._index

    @property
    def current_id(self) -> int | None:
        if self._index is None:
            return None
        return self._ids[self._index]

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def is_first(self) -> bool:
        return self._index is None or self._index == 0

    @property
    def is_last(self) -> bool:
        return self._index is None or self._index == len(self._ids) - 1

    def move(self, delta: int) -> bool:
        """Move by `delta` entries, stopping at either end.

        Returns:
            Whether the current entry changed.
        """
        if not self._active or self._index is None or delta == 0:
            return False
        new_index = max(0, min(len(self._ids) - 1, self._index + delta))
        if new_index == self._index:
            return False
        self._index = new_index
        return True

    def move_to_first(self) -> bool:
        if self._index is None:
            return False
        return self.move(-self._index)

    def move_to_last(self) -> bool:
        if self._index is None:
            return False
        return self.move(len(self._ids) - 1 - self._index)

    def invalidate(self) -> None:
        """End the session. Navigation is disabled afterwards."""
        self._active = False
