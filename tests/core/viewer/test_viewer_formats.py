# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only


from pathlib import Path

import pytest

from tagstudio.core.media_types import MediaTypes
from tagstudio.core.viewer.formats import (
    FormatSupport,
    ViewerCodec,
    codec_for,
    is_viewable,
    support_for,
    viewable_name_predicate,
    viewer_ext,
)


@pytest.mark.parametrize(
    ["filename", "codec"],
    [
        ("a.png", ViewerCodec.PILLOW),
        ("A.JPG", ViewerCodec.PILLOW),
        ("a.jfif", ViewerCodec.PILLOW),
        ("a.webp", ViewerCodec.PILLOW),
        ("a.heic", ViewerCodec.HEIF),
        ("a.CR3", ViewerCodec.RAW),
        ("a.png.gz", ViewerCodec.PILLOW),
        ("a.nef.zst", ViewerCodec.RAW),
        ("a.exr", None),
        ("a.mp4", None),
        ("a.txt", None),
        ("a.tar.gz", None),
        ("noext", None),
    ],
)
def test_codec_for(filename: str, codec: ViewerCodec | None):
    assert codec_for(Path(filename)) == codec
    assert is_viewable(Path(filename)) is (codec is not None)


@pytest.mark.parametrize(
    ["filename", "support"],
    [
        ("a.png", FormatSupport.SUPPORTED),
        ("a.svg", FormatSupport.DEFERRED),
        ("a.exr", FormatSupport.DEFERRED),
        ("a.psd", FormatSupport.DEFERRED),
        ("a.mp4", FormatSupport.OUT_OF_SCOPE),
        ("a.pdf", FormatSupport.OUT_OF_SCOPE),
        ("a.zip", FormatSupport.OUT_OF_SCOPE),
    ],
)
def test_support_for(filename: str, support: FormatSupport):
    assert support_for(Path(filename)) == support


def test_viewer_ext_of_compressed_file():
    assert viewer_ext(Path("photo.PNG.xz")) == ".png"
    assert viewer_ext(Path("photo.png")) == ".png"


def test_format_lookup_does_not_modify_media_types():
    before = MediaTypes._snapshot()  # pyright: ignore[reportPrivateUsage]
    for name in ("a.png", "a.jpeg_large", "a.png.gz", "a.cr2", "a.mp4"):
        support_for(Path(name))
    assert MediaTypes._snapshot() == before  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    "name",
    [
        "a.png",
        "dir/A.JPG",
        "x.jpeg_large",
        "x.heic",
        "photo.CR3",
        "a.png.gz",
        "a.nef.zst",
        "a.txt.gz",
        "a.tar.gz",
        "a.mp4",
        "noext",
        "dir.png/noext",
        ".png",
        "dir/.png",
        "a.",
        "a..png",
        "a.exr",
        "archive.tar.xz",
    ],
)
def test_viewable_name_predicate_matches_is_viewable(name: str):
    predicate = viewable_name_predicate()
    assert predicate(name) == is_viewable(Path(name))
