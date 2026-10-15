# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only


from pathlib import Path

from tagstudio.core.viewer.cache import ImageCache, ImageKey, LoadKey, covers
from tagstudio.core.viewer.decoding import DEFAULT_POLICY, DecodedImage, DecodePolicy, FileVersion

V1 = FileVersion(1, 100)
V2 = FileVersion(2, 100)


def _key(
    entry_id: int = 1,
    target: tuple[int, int] = (100, 100),
    version: FileVersion = V1,
    token: int = 1,
    policy: DecodePolicy = DEFAULT_POLICY,
) -> LoadKey:
    return LoadKey(ImageKey(token, entry_id, Path(f"/lib/{entry_id}.png"), version, policy), target)


def _image(
    width: int, height: int, source: tuple[int, int] | None = None, version: FileVersion = V1
) -> DecodedImage:
    source = source or (width, height)
    return DecodedImage(
        data=bytes(width * height * 3),
        width=width,
        height=height,
        mode="RGB",
        source_width=source[0],
        source_height=source[1],
        source_format="PNG",
        file_version=version,
    )


def test_hit_and_miss():
    cache = ImageCache()
    assert cache.get(_key()) is None
    image = _image(100, 50)
    assert cache.put(_key(), image)
    assert cache.get(_key()) is image
    assert cache.get(_key(entry_id=2)) is None
    assert (cache.stats.hits, cache.stats.misses) == (1, 2)


def test_reduced_image_covers_smaller_targets_only():
    reduced = _image(100, 50, source=(1000, 500))
    assert covers(reduced, (100, 100), DEFAULT_POLICY)
    assert covers(reduced, (60, 60), DEFAULT_POLICY)
    assert not covers(reduced, (400, 400), DEFAULT_POLICY)

    cache = ImageCache()
    cache.put(_key(target=(100, 100)), reduced)
    assert cache.get(_key(target=(80, 80))) is reduced
    assert cache.get(_key(target=(400, 400))) is None


def test_full_resolution_image_covers_any_target():
    full = _image(100, 50)
    cache = ImageCache()
    cache.put(_key(target=(100, 100)), full)
    assert cache.get(_key(target=(4000, 4000))) is full


def test_smallest_covering_variant_is_returned():
    cache = ImageCache()
    small = _image(100, 50, source=(1000, 500))
    large = _image(400, 200, source=(1000, 500))
    cache.put(_key(target=(400, 400)), large)
    cache.put(_key(target=(100, 100)), small)
    assert cache.get(_key(target=(90, 90))) is small
    assert cache.get(_key(target=(300, 300))) is large


def test_eviction_respects_byte_limit():
    image_bytes = 10 * 10 * 3
    cache = ImageCache(max_bytes=image_bytes * 2)
    cache.put(_key(1), _image(10, 10))
    cache.put(_key(2), _image(10, 10))
    cache.get(_key(1))  # 1 is now the most recently used
    cache.put(_key(3), _image(10, 10))
    assert cache.get(_key(2)) is None
    assert cache.get(_key(1)) is not None
    assert cache.get(_key(3)) is not None
    assert cache.bytes_used == image_bytes * 2
    assert cache.stats.evictions == 1
    assert cache.stats.peak_bytes <= cache.max_bytes


def test_single_image_larger_than_limit_is_not_stored():
    cache = ImageCache(max_bytes=10)
    cache.put(_key(1), _image(1, 1))
    assert not cache.put(_key(2), _image(10, 10))
    assert cache.get(_key(1)) is not None
    assert cache.bytes_used == 3


def test_file_update_misses():
    cache = ImageCache()
    cache.put(_key(version=V1), _image(10, 10, version=V1))
    assert cache.get(_key(version=V2)) is None


def test_result_of_changed_file_is_not_stored_under_old_version():
    cache = ImageCache()
    assert not cache.put(_key(version=V1), _image(10, 10, version=V2))
    assert len(cache) == 0


def test_policy_change_misses():
    cache = ImageCache()
    cache.put(_key(), _image(10, 10))
    assert cache.get(_key(policy=DecodePolicy(upscale=True))) is None
    assert cache.get(_key(policy=DecodePolicy(version=2))) is None


def test_library_switch_misses_and_discard():
    cache = ImageCache()
    cache.put(_key(token=1), _image(10, 10))
    cache.put(_key(entry_id=2, token=2), _image(10, 10))
    assert cache.get(_key(token=2)) is None  # Same entry ID in another library

    cache.discard_library(1)
    assert cache.get(_key(token=1)) is None
    assert cache.get(_key(entry_id=2, token=2)) is not None
    assert cache.bytes_used == 300


def test_replacing_a_key_updates_size():
    cache = ImageCache()
    cache.put(_key(), _image(10, 10))
    cache.put(_key(), _image(20, 20))
    assert len(cache) == 1
    assert cache.bytes_used == 20 * 20 * 3


def test_clear():
    cache = ImageCache()
    cache.put(_key(), _image(10, 10))
    cache.clear()
    assert len(cache) == 0
    assert cache.bytes_used == 0
