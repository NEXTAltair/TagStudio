# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only


from pathlib import Path

import pytest
from sqlalchemy import select

from tagstudio.core.library.alchemy.library import Library
from tagstudio.core.library.alchemy.models import Entry
from tagstudio.core.viewer.library_reader import LibraryReader, LibraryUnavailableError


def _add_entries(lib: Library, count: int, prefix: str = "img") -> list[int]:
    ids = lib.add_entries([Entry(path=Path(f"{prefix}_{i}.png"), fields=[]) for i in range(count)])
    return list(ids)


def _all_rows(lib: Library) -> list[tuple[int, Path]]:
    with lib.engine.connect() as conn:  # pyright: ignore[reportOptionalMemberAccess]
        return [(r[0], r[1]) for r in conn.execute(select(Entry.id, Entry.path))]


@pytest.fixture
def file_library(tmp_path: Path):
    """A library backed by a real SQLite file (NullPool), unlike the in-memory fixtures."""
    lib = Library()
    assert lib.open_library(tmp_path).success
    yield lib
    lib.close()


def test_requires_open_library():
    with pytest.raises(LibraryUnavailableError):
        LibraryReader(Library())


def test_resolve_empty(library: Library):
    reader = LibraryReader(library)
    assert reader.resolve([]) == []
    assert reader.query_count == 0


def test_resolve_keeps_order_and_reports_missing(library: Library):
    reader = LibraryReader(library)
    locations = reader.resolve([2, 999, 1, 2])
    assert [loc.entry_id if loc else None for loc in locations] == [2, None, 1, 2]
    assert locations[0] is not None
    assert locations[0].path == reader.library_dir / "one/two/bar.md"
    assert locations[0].library_token == reader.token
    assert reader.query_count == 1


def test_resolve_uses_chunks(library: Library):
    ids = _add_entries(library, 5)
    reader = LibraryReader(library, chunk_size=2)
    locations = reader.resolve(ids)
    assert [loc.entry_id for loc in locations if loc] == ids
    assert reader.query_count == 3


def test_resolve_many_stays_below_parameter_limit(library: Library):
    ids = _add_entries(library, 1201)
    reader = LibraryReader(library)
    locations = reader.resolve(ids)
    assert all(loc is not None for loc in locations)
    assert reader.query_count == 3  # 500 + 500 + 201, never one query per entry


def test_resolve_caches_paths(library: Library):
    reader = LibraryReader(library)
    reader.resolve([1, 2])
    reader.resolve([2, 1])
    assert reader.query_count == 1


def test_resolve_without_cache_does_not_query_per_entry(library: Library):
    ids = _add_entries(library, 20)
    reader = LibraryReader(library, cache_size=0)
    assert all(reader.resolve(ids))
    assert reader.query_count == 1


def test_path_cache_is_bounded(library: Library):
    ids = _add_entries(library, 10)
    reader = LibraryReader(library, cache_size=3)
    reader.resolve(ids)
    assert len(reader._paths) == 3  # pyright: ignore[reportPrivateUsage]


def test_scan(library: Library):
    reader = LibraryReader(library, chunk_size=1)
    verdicts = reader.scan([1, 2, 404], lambda p: p.suffix == ".md")
    assert verdicts == [False, True, False]
    assert reader.query_count == 3


def test_reads_do_not_modify_library(library: Library):
    before = _all_rows(library)
    reader = LibraryReader(library)
    reader.resolve([1, 2, 3])
    reader.scan([1, 2], lambda p: True)
    assert _all_rows(library) == before


def test_closed_library_is_rejected(file_library: Library):
    ids = _add_entries(file_library, 2)
    reader = LibraryReader(file_library)
    assert reader.resolve(ids)[0] is not None

    file_library.close()
    assert not reader.is_valid()
    with pytest.raises(LibraryUnavailableError):
        reader.resolve(ids)
    with pytest.raises(LibraryUnavailableError):
        reader.scan(ids, lambda p: True)


def test_reopened_library_is_a_different_library(file_library: Library, tmp_path: Path):
    ids = _add_entries(file_library, 1)
    reader = LibraryReader(file_library)
    file_library.close()
    assert file_library.open_library(tmp_path).success

    # Same directory and same entry IDs, but a new opened library: IDs must not be trusted.
    assert not reader.is_valid()
    with pytest.raises(LibraryUnavailableError):
        reader.resolve(ids)
    new_reader = LibraryReader(file_library)
    assert new_reader.token != reader.token
    assert new_reader.resolve(ids)[0] is not None


def test_explicit_close(library: Library):
    reader = LibraryReader(library)
    reader.close()
    with pytest.raises(LibraryUnavailableError):
        reader.resolve([1])
