# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only

"""The image viewer's file format support table.

This table describes what the viewer can actually decode. It is intentionally separate from the
`MediaTypes` SEARCH/RENDER registrations, which describe how files are classified or thumbnailed.
`MediaTypes` is only used here to read compound and equivalent extensions, never modified.
"""

from collections.abc import Callable
from enum import Enum
from pathlib import Path

from tagstudio.core.media_types import MediaTypeGroup, MediaTypes
from tagstudio.core.utils.decompression import get_inner_ext


class ViewerCodec(Enum):
    """The decoder adapter used for a supported format."""

    PILLOW = "pillow"
    HEIF = "heif"
    RAW = "raw"


class FormatSupport(Enum):
    """Whether a format is handled by the viewer."""

    SUPPORTED = "supported"
    DEFERRED = "deferred"
    """Image formats that may be supported later, but are not shown by the viewer yet."""
    OUT_OF_SCOPE = "out_of_scope"
    """Non-image formats (video, audio, documents, archives, etc.)."""


# NOTE: Animated formats (GIF, APNG, WebP, JXL) only show their first frame.
SUPPORTED_EXTS: dict[str, ViewerCodec] = {
    ".apng": ViewerCodec.PILLOW,
    ".avif": ViewerCodec.PILLOW,
    ".bmp": ViewerCodec.PILLOW,
    ".gif": ViewerCodec.PILLOW,
    ".heic": ViewerCodec.HEIF,
    ".heif": ViewerCodec.HEIF,
    ".icns": ViewerCodec.PILLOW,
    ".ico": ViewerCodec.PILLOW,
    ".j2k": ViewerCodec.PILLOW,
    ".jfif": ViewerCodec.PILLOW,
    ".jp2": ViewerCodec.PILLOW,
    ".jpeg": ViewerCodec.PILLOW,
    ".jpeg_large": ViewerCodec.PILLOW,
    ".jpg": ViewerCodec.PILLOW,
    ".jpg2": ViewerCodec.PILLOW,
    ".jpg_large": ViewerCodec.PILLOW,
    ".jxl": ViewerCodec.PILLOW,
    ".png": ViewerCodec.PILLOW,
    ".tif": ViewerCodec.PILLOW,
    ".tiff": ViewerCodec.PILLOW,
    ".webp": ViewerCodec.PILLOW,
    # RAW (decoded with rawpy/LibRaw)
    ".arw": ViewerCodec.RAW,
    ".cr2": ViewerCodec.RAW,
    ".cr3": ViewerCodec.RAW,
    ".crw": ViewerCodec.RAW,
    ".dng": ViewerCodec.RAW,
    ".nef": ViewerCodec.RAW,
    ".nrw": ViewerCodec.RAW,
    ".orf": ViewerCodec.RAW,
    ".raf": ViewerCodec.RAW,
    ".raw": ViewerCodec.RAW,
    ".rw2": ViewerCodec.RAW,
    ".srf": ViewerCodec.RAW,
    ".srf2": ViewerCodec.RAW,
}

DEFERRED_EXTS: frozenset[str] = frozenset(
    {
        ".eps",
        ".epsf",
        ".epsi",
        ".exr",
        ".icon",
        ".ora",
        ".psd",
        ".r3d",
        ".svg",
        ".svgz",
        ".vtf",
    }
)

# Pillow plugins the viewer allows when opening files, regardless of the file extension.
# Misnamed files (e.g. a PNG saved as ".jpg") are common, so the plugin isn't tied to the suffix.
PILLOW_FORMATS: tuple[str, ...] = (
    "AVIF",
    "BMP",
    "DIB",
    "GIF",
    "ICNS",
    "ICO",
    "JPEG",
    "JPEG2000",
    "JXL",
    "MPO",
    "PNG",
    "TIFF",
    "WEBP",
)


def _lookup(ext: str) -> ViewerCodec | None:
    for e in MediaTypes.get_equivalent_exts(ext) | {ext}:
        if (codec := SUPPORTED_EXTS.get(e)) is not None:
            return codec
    return None


def compressed_inner_ext(path: Path) -> str | None:
    """Return the inner extension of a compressed file (e.g. ".png" for "a.png.gz"), if any."""
    try:
        return get_inner_ext(path)
    except AttributeError:
        # The "compressed" media type group hasn't been registered (e.g. outside of the app).
        return None


def viewer_ext(path: Path) -> str:
    """Return the extension that determines how the viewer decodes a file.

    For compressed files (e.g. "image.png.gz") this is the inner extension (".png").
    """
    return compressed_inner_ext(path) or MediaTypes.get_ext(path)


def codec_for(path: Path) -> ViewerCodec | None:
    """Return the decoder adapter for a path, or None if the viewer doesn't support it."""
    return _lookup(viewer_ext(path))


def support_for(path: Path) -> FormatSupport:
    """Classify a path according to the viewer's format support table."""
    ext = viewer_ext(path)
    if _lookup(ext) is not None:
        return FormatSupport.SUPPORTED
    if any(e in DEFERRED_EXTS for e in MediaTypes.get_equivalent_exts(ext) | {ext}):
        return FormatSupport.DEFERRED
    return FormatSupport.OUT_OF_SCOPE


def is_viewable(path: Path) -> bool:
    """Return True if the viewer can attempt to decode this path."""
    return codec_for(path) is not None


def viewable_name_predicate() -> Callable[[str], bool]:
    """Return a fast version of `is_viewable()` for many relative POSIX path strings.

    The extension sets are computed once, from the current `MediaTypes` registrations, so each
    check is only a few string and set operations. Names whose result can depend on more than
    their last suffix (compressed files, compound extensions) use `is_viewable()` itself.
    """
    supported: set[str] = set()
    for ext in SUPPORTED_EXTS:
        supported |= MediaTypes.get_equivalent_exts(ext) | {ext}

    compressed_group = getattr(MediaTypes, "compressed", None)
    exact_suffixes: set[str] = set()
    if isinstance(compressed_group, MediaTypeGroup):
        for context_exts in compressed_group.context_sets.values():
            exact_suffixes |= context_exts
    for group in MediaTypes.all_groups:
        for file_type in group.types:
            compound_exts = (e for e in file_type.exts if e.count(".") > 1)
            exact_suffixes |= {f".{e.rsplit('.', 1)[1]}" for e in compound_exts}

    def predicate(name: str) -> bool:
        # Same rule as `PurePath.suffix`: a leading or trailing dot isn't a suffix.
        file_name = name[name.rfind("/") + 1 :]
        dot = file_name.rfind(".")
        if not 0 < dot < len(file_name) - 1:
            return False
        suffix = file_name[dot:].lower()
        if suffix in exact_suffixes:
            return is_viewable(Path(name))
        return suffix in supported

    return predicate
