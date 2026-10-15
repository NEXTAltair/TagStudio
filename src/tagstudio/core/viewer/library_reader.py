# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only

"""A small read-only adapter that maps entry IDs to file paths for the image viewer.

Only the `id` and `path` columns are read, in bounded chunks, with short-lived connections.
Decoders receive plain `Path` values and never see the library, sessions, or ORM objects.
"""

import itertools
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import Connection, select

from tagstudio.core.library.alchemy.models import Entry

if TYPE_CHECKING:
    from sqlalchemy import Engine

    from tagstudio.core.library.alchemy.library import Library

logger = structlog.get_logger(__name__)

# Kept well below SQLite's bound parameter limit (999 on older builds).
DEFAULT_CHUNK_SIZE: int = 500
DEFAULT_PATH_CACHE_SIZE: int = 2048

_token_counter = itertools.count(1)


class LibraryUnavailableError(RuntimeError):
    """Raised when reading from a library that was closed or replaced."""


@dataclass(frozen=True, slots=True)
class EntryLocation:
    """Where an entry's file is, as seen by a specific opened library.

    Attributes:
        library_token: Identifies the opened library the location was read from.
        entry_id: The entry's ID.
        path: The absolute path to the entry's file.
    """

    library_token: int
    entry_id: int
    path: Path


def _chunks(items: Sequence[int], size: int) -> Iterator[Sequence[int]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class LibraryReader:
    """Reads entry paths from one opened library.

    The reader captures the library's directory and database engine when created. If the library
    is closed, or another library (or the same one again) is opened, the reader becomes invalid
    and refuses to query. Entry IDs are only meaningful together with `token`.
    """

    def __init__(
        self,
        library: Library,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        cache_size: int = DEFAULT_PATH_CACHE_SIZE,
    ) -> None:
        if library.library_dir is None or library.engine is None:
            raise LibraryUnavailableError("No library is open")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self._library = library
        self._library_dir: Path = library.library_dir
        self._engine: Engine = library.engine
        self._token = next(_token_counter)
        self._chunk_size = chunk_size
        self._cache_size = max(0, cache_size)
        self._paths: OrderedDict[int, Path | None] = OrderedDict()
        self._closed = False
        self.query_count = 0
        """Number of SELECT statements executed (for tests and measurements)."""

    @property
    def token(self) -> int:
        """A unique value for the opened library this reader reads from."""
        return self._token

    @property
    def library_dir(self) -> Path:
        return self._library_dir

    def is_valid(self) -> bool:
        """Whether the library this reader was created for is still the open one."""
        return (
            not self._closed
            and self._library.library_dir == self._library_dir
            and self._library.engine is self._engine
        )

    def close(self) -> None:
        """Stop reading. Called before the library is closed."""
        self._closed = True
        self._paths.clear()

    def _ensure_valid(self) -> None:
        if not self.is_valid():
            raise LibraryUnavailableError("The library was closed or replaced")

    def _select_paths(self, conn: Connection, ids: Sequence[int]) -> dict[int, Path]:
        self.query_count += 1
        stmt = select(Entry.id, Entry.path).where(Entry.id.in_(ids))
        return {row[0]: row[1] for row in conn.execute(stmt)}

    def _remember(self, entry_id: int, path: Path | None) -> None:
        if self._cache_size == 0:
            return
        self._paths[entry_id] = path
        self._paths.move_to_end(entry_id)
        while len(self._paths) > self._cache_size:
            self._paths.popitem(last=False)

    def resolve(self, entry_ids: Sequence[int]) -> list[EntryLocation | None]:
        """Return the location of each entry, in the given order.

        Missing entries (e.g. removed from the library) are returned as None.

        Raises:
            LibraryUnavailableError: If the library was closed or replaced.
        """
        self._ensure_valid()
        rel_paths: dict[int, Path | None] = {}
        wanted: list[int] = []
        for entry_id in dict.fromkeys(entry_ids):
            if entry_id in self._paths:
                self._paths.move_to_end(entry_id)
                rel_paths[entry_id] = self._paths[entry_id]
            else:
                wanted.append(entry_id)

        if wanted:
            with self._engine.connect() as conn:
                for chunk in _chunks(wanted, self._chunk_size):
                    found = self._select_paths(conn, chunk)
                    for entry_id in chunk:
                        rel_paths[entry_id] = found.get(entry_id)
                        self._remember(entry_id, rel_paths[entry_id])

        return [
            None
            if (rel_path := rel_paths[entry_id]) is None
            else EntryLocation(self._token, entry_id, self._library_dir / rel_path)
            for entry_id in entry_ids
        ]

    def resolve_one(self, entry_id: int) -> EntryLocation | None:
        return self.resolve([entry_id])[0]

    def scan(self, entry_ids: Sequence[int], predicate: Callable[[Path], bool]) -> list[bool]:
        """Return `predicate(relative_path)` for each entry, in the given order.

        Paths are read in chunks over a single connection and aren't kept afterwards, so memory
        use stays bounded by the chunk size. Missing entries map to False.

        Raises:
            LibraryUnavailableError: If the library was closed or replaced.
        """
        self._ensure_valid()
        unique = list(dict.fromkeys(entry_ids))
        verdicts: dict[int, bool] = {}
        with self._engine.connect() as conn:
            for chunk in _chunks(unique, self._chunk_size):
                found = self._select_paths(conn, chunk)
                for entry_id in chunk:
                    path = found.get(entry_id)
                    verdicts[entry_id] = path is not None and predicate(path)
        return [verdicts[i] for i in entry_ids]
