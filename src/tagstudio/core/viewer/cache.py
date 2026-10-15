# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only

"""A size-limited in-memory cache of decoded images, used only by the image viewer.

The cache is independent of the thumbnail cache: nothing is written to disk and the thumbnail
cache's files and size limits are never touched.
"""

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from tagstudio.core.viewer.decoding import DecodedImage, DecodePolicy, FileVersion, fit_size

MiB = 1024 * 1024
DEFAULT_CACHE_BYTES: int = 256 * MiB


@dataclass(frozen=True, slots=True)
class ImageKey:
    """Identifies the decoded content of a file, independent of the output size.

    Attributes:
        library_token: The opened library the entry belongs to.
        entry_id: The entry the file belongs to.
        path: The absolute file path.
        version: The file's version; a modified file never matches older results.
        policy: The decoding options; changing them never matches older results.
    """

    library_token: int
    entry_id: int
    path: Path
    version: FileVersion
    policy: DecodePolicy


@dataclass(frozen=True, slots=True)
class LoadKey:
    """An image at a specific maximum output size in physical pixels."""

    image: ImageKey
    target_size: tuple[int, int]


def covers(image: DecodedImage, target_size: tuple[int, int], policy: DecodePolicy) -> bool:
    """Whether a decoded image is detailed enough to be shown in a target box.

    A full resolution image covers every target. A reduced image covers targets that wouldn't
    be shown any larger than it already is.
    """
    if image.is_full_resolution and not policy.upscale:
        return True
    needed = fit_size((image.source_width, image.source_height), target_size, policy.upscale)
    return needed[0] <= image.width and needed[1] <= image.height


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    stores: int = 0
    rejected: int = 0
    evictions: int = 0
    peak_bytes: int = 0


class ImageCache:
    """An LRU cache of decoded images with a limit on the total size of their pixel buffers.

    The limit counts the decoded bytes held by the cache. It is not a limit on the process's
    memory: images being displayed or decoded, and the codecs' own temporary allocations, are
    not counted here.
    """

    def __init__(self, max_bytes: int = DEFAULT_CACHE_BYTES) -> None:
        self._max_bytes = max(0, max_bytes)
        self._items: OrderedDict[LoadKey, DecodedImage] = OrderedDict()
        self._bytes = 0
        self.stats = CacheStats()

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def bytes_used(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._items)

    def get(self, key: LoadKey) -> DecodedImage | None:
        """Return the smallest cached image that covers the requested target size, if any."""
        best_key: LoadKey | None = None
        best: DecodedImage | None = None
        for k, image in self._items.items():
            if k.image != key.image or not covers(image, key.target_size, key.image.policy):
                continue
            if best is None or image.nbytes < best.nbytes:
                best_key, best = k, image
        if best_key is None or best is None:
            self.stats.misses += 1
            return None
        self._items.move_to_end(best_key)
        self.stats.hits += 1
        return best

    def put(self, key: LoadKey, image: DecodedImage) -> bool:
        """Store an image, evicting the least recently used ones to stay within the limit.

        Returns:
            False if the image alone is larger than the limit (it isn't stored).
        """
        if image.nbytes > self._max_bytes:
            self.stats.rejected += 1
            return False
        if image.file_version is not None and image.file_version != key.image.version:
            # The file changed between the request and the decode; don't store it under the
            # older version.
            self.stats.rejected += 1
            return False
        if (old := self._items.pop(key, None)) is not None:
            self._bytes -= old.nbytes
        self._items[key] = image
        self._bytes += image.nbytes
        self.stats.stores += 1
        while self._bytes > self._max_bytes and self._items:
            _, evicted = self._items.popitem(last=False)
            self._bytes -= evicted.nbytes
            self.stats.evictions += 1
        self.stats.peak_bytes = max(self.stats.peak_bytes, self._bytes)
        return True

    def discard_library(self, library_token: int) -> None:
        """Remove every image read from a library."""
        for k in [k for k in self._items if k.image.library_token == library_token]:
            self._bytes -= self._items.pop(k).nbytes

    def clear(self) -> None:
        self._items.clear()
        self._bytes = 0
