# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only

"""Bounded image decoding for the image viewer.

Decoding returns the pixels and the image attributes from a single read of the file. Unlike the
thumbnail renderers, the result keeps transparency (no background is composited into the pixels)
and is never decorated (no rounded corners or masks).

Global Pillow settings (such as `Image.MAX_IMAGE_PIXELS`) are not read or modified here. Instead,
each request carries its own memory budget that is checked before the full image is loaded.
"""

import stat
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
import structlog
from PIL import Image, UnidentifiedImageError

from tagstudio.core.utils.decompression import decompressed
from tagstudio.core.viewer.formats import (
    PILLOW_FORMATS,
    ViewerCodec,
    codec_for,
    compressed_inner_ext,
    viewer_ext,
)

logger = structlog.get_logger(__name__)

MiB = 1024 * 1024

DEFAULT_MAX_DECODE_BYTES: int = 512 * MiB
"""Default estimated peak size of the pixel buffers allocated while decoding one image."""

MAX_TARGET_SIDE: int = 16384
"""The largest output width or height, regardless of the requested target size."""

HIGH_BIT_DEPTH_BYTES_PER_PIXEL: int = 12
"""Upper bound of the temporary arrays used to convert I, I;16 and F images to 8 bits."""

OUTPUT_BYTES_PER_PIXEL: int = 12
"""Upper bound of the output sized buffers alive at once while finishing a decode: the resized
32-bit image, a transposed or converted copy of it, and the returned bytes."""

OutputMode = Literal["L", "RGB", "RGBA"]

CancelCheck = Callable[[], bool]


class DecodeErrorKind(Enum):
    MISSING = "missing"
    """The file does not exist (or is not a regular file)."""
    UNSUPPORTED = "unsupported"
    """The file's format is not in the viewer's support table, or the codec can't read it."""
    CODEC_UNAVAILABLE = "codec_unavailable"
    """The format is supported, but the required codec/plugin is not installed."""
    CORRUPT = "corrupt"
    """The file could not be decoded."""
    OVER_BUDGET = "over_budget"
    """Decoding the file would exceed the memory budget."""
    CHANGED = "changed"
    """The file changed after the request was made."""
    CANCELED = "canceled"
    """The request was canceled before decoding finished."""


@dataclass(frozen=True, slots=True)
class FileVersion:
    """A cheap fingerprint of a file's content version."""

    mtime_ns: int
    size: int

    @classmethod
    def of(cls, path: Path) -> FileVersion | None:
        """Return the version of a regular file, or None if it can't be read."""
        try:
            st = path.stat()
        except OSError:
            return None
        if not stat.S_ISREG(st.st_mode):
            return None
        return cls(st.st_mtime_ns, st.st_size)


@dataclass(frozen=True, slots=True)
class DecodePolicy:
    """Options that affect the decoded pixels. Part of every cache key.

    Attributes:
        version: Bumped whenever the decoding output changes for the same options.
        upscale: Whether images smaller than the target are enlarged (they are not by default).
        apply_orientation: Whether EXIF orientation is applied.
        raw_camera_white_balance: Whether RAW files use the camera's white balance.
    """

    version: int = 1
    upscale: bool = False
    apply_orientation: bool = True
    raw_camera_white_balance: bool = True


DEFAULT_POLICY = DecodePolicy()


@dataclass(frozen=True, slots=True)
class DecodeRequest:
    """Everything a decoder needs; contains no library, ORM, or UI objects.

    Attributes:
        path: The absolute path of the file.
        target_size: The maximum output size in physical (device) pixels.
        expected_version: If set, the file must still match this version when decoding starts.
        policy: Options that affect the output pixels.
        max_decode_bytes: Budget for the estimated peak pixel buffers while decoding.
    """

    path: Path
    target_size: tuple[int, int]
    expected_version: FileVersion | None = None
    policy: DecodePolicy = DEFAULT_POLICY
    max_decode_bytes: int = DEFAULT_MAX_DECODE_BYTES


@dataclass(frozen=True, slots=True, eq=False)
class DecodedImage:
    """An immutable, self-owned pixel buffer and the attributes read with it.

    Attributes:
        data: Tightly packed rows of pixels (no padding), owned by this object.
        width: Width of `data` in pixels.
        height: Height of `data` in pixels.
        mode: Pixel layout of `data`.
        source_width: Width of the full image after orientation is applied.
        source_height: Height of the full image after orientation is applied.
        source_format: The decoder's name for the file format (e.g. "JPEG").
        is_animated: Whether the file has more than one frame. Only the first one is decoded.
        file_version: Version of the file that was decoded.
    """

    data: bytes = field(repr=False)
    width: int
    height: int
    mode: OutputMode
    source_width: int
    source_height: int
    source_format: str
    is_animated: bool = False
    file_version: FileVersion | None = None

    @property
    def channels(self) -> int:
        return {"L": 1, "RGB": 3, "RGBA": 4}[self.mode]

    @property
    def bytes_per_line(self) -> int:
        return self.width * self.channels

    @property
    def nbytes(self) -> int:
        return len(self.data)

    @property
    def has_alpha(self) -> bool:
        return self.mode == "RGBA"

    @property
    def is_full_resolution(self) -> bool:
        return self.width >= self.source_width and self.height >= self.source_height


@dataclass(frozen=True, slots=True)
class DecodeFailure:
    kind: DecodeErrorKind
    path: Path
    message: str = ""


DecodeOutcome = DecodedImage | DecodeFailure


class _DecodeError(Exception):
    def __init__(self, kind: DecodeErrorKind, message: str = "") -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


# Maps EXIF orientation values to the transposition that displays the image upright.
_ORIENTATION_TRANSPOSE: dict[int, Image.Transpose] = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}
_SWAPS_AXES = frozenset({5, 6, 7, 8})
_EXIF_ORIENTATION_TAG = 0x0112

# Bytes per pixel of Pillow's internal storage for each mode (RGB is stored as 32 bits).
_PILLOW_BYTES_PER_PIXEL: dict[str, int] = {
    "1": 1,
    "L": 1,
    "P": 1,
    "I;16": 2,
    "I;16B": 2,
    "I;16L": 2,
    "I;16N": 2,
}


def fit_size(
    source: tuple[int, int], box: tuple[int, int], upscale: bool = False
) -> tuple[int, int]:
    """Return the largest size with the source's aspect ratio that fits inside the box.

    Args:
        source: The (width, height) of the image.
        box: The (width, height) of the available area.
        upscale: Whether the result may be larger than the source.
    """
    sw, sh = max(1, source[0]), max(1, source[1])
    bw, bh = max(1, box[0]), max(1, box[1])
    scale = min(bw / sw, bh / sh)
    if not upscale:
        scale = min(scale, 1.0)
    return max(1, round(sw * scale)), max(1, round(sh * scale))


def _clamp_target(target: tuple[int, int]) -> tuple[int, int]:
    return (
        max(1, min(MAX_TARGET_SIDE, int(target[0]))),
        max(1, min(MAX_TARGET_SIDE, int(target[1]))),
    )


def _check_cancel(cancel: CancelCheck | None) -> None:
    if cancel is not None and cancel():
        raise _DecodeError(DecodeErrorKind.CANCELED)


def _check_budget(nbytes: int, budget: int, what: str) -> None:
    if nbytes > budget:
        raise _DecodeError(
            DecodeErrorKind.OVER_BUDGET,
            f"{what} needs an estimated {nbytes // MiB} MiB (budget: {budget // MiB} MiB)",
        )


def _is_high_bit_depth(mode: str) -> bool:
    return mode in ("I", "F") or mode.startswith("I;16")


def _normalize_mode(im: Image.Image) -> Image.Image:
    """Convert an image to L, RGB or RGBA without discarding transparency.

    High bit depth images (I, I;16, F) should be resized to the output size first: their
    conversion uses temporary arrays of up to `HIGH_BIT_DEPTH_BYTES_PER_PIXEL` per pixel.
    """
    mode = im.mode
    if mode in ("L", "RGB") and "transparency" in im.info:
        # A tRNS color key: the pixels with that value are transparent.
        return im.convert("RGBA")
    if mode in ("L", "RGB", "RGBA"):
        return im
    if mode in ("LA", "PA", "RGBa", "La"):
        return im.convert("RGBA")
    if mode == "P":
        return im.convert("RGBA" if "transparency" in im.info else "RGB")
    if mode == "1":
        return im.convert("L")
    if mode.startswith("I;16"):
        # Scale 16-bit samples down to 8 bits instead of clipping them.
        arr = np.array(im)
        arr >>= 8
        return Image.fromarray(arr.astype(np.uint8))
    if mode == "I":
        arr = np.array(im)
        if arr.max(initial=0) > 255:
            np.clip(arr, 0, 65535, out=arr)
            arr >>= 8
        np.clip(arr, 0, 255, out=arr)
        return Image.fromarray(arr.astype(np.uint8))
    if mode == "F":
        # Stretch the actual range of finite samples to 0-255, working in place.
        arr = np.array(im, dtype=np.float32)
        finite = np.isfinite(arr)
        lo = float(np.min(arr, where=finite, initial=np.inf))
        hi = float(np.max(arr, where=finite, initial=-np.inf))
        del finite
        if lo > hi:  # No finite samples
            lo = hi = 0.0
        scale = 255.0 / (hi - lo) if hi > lo else 0.0
        arr -= lo
        arr *= scale
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=255.0, neginf=0.0)
        np.clip(arr, 0, 255, out=arr)
        return Image.fromarray(arr.astype(np.uint8))
    # CMYK, YCbCr, LAB, HSV, etc. NOTE: Embedded ICC profiles are not applied.
    return im.convert("RGB")


def _drop_opaque_alpha(im: Image.Image) -> Image.Image:
    if im.mode == "RGBA":
        extrema = im.getchannel("A").getextrema()
        if extrema[0] == 255:
            return im.convert("RGB")
    return im


def _finish(
    im: Image.Image,
    decode_size: tuple[int, int],
    orientation: int,
    source_size: tuple[int, int],
    source_format: str,
    is_animated: bool,
    version: FileVersion | None,
    cancel: CancelCheck | None,
) -> DecodedImage:
    """Normalize, resize, orient and copy an already loaded image into a DecodedImage."""
    # High bit depth images are resized first, so their conversion works on the smaller image.
    # Other modes are normalized first (e.g. palettes and color keys can't be resampled).
    high_bit_depth = _is_high_bit_depth(im.mode)
    if not high_bit_depth:
        im = _normalize_mode(im)
    _check_cancel(cancel)
    if im.size != decode_size:
        # NOTE: Pillow's reduce() step (used with reducing_gap) doesn't support I;16 images.
        gap = None if high_bit_depth else 3.0
        im = im.resize(decode_size, Image.Resampling.LANCZOS, reducing_gap=gap)
    if high_bit_depth:
        im = _normalize_mode(im)
    if (transpose := _ORIENTATION_TRANSPOSE.get(orientation)) is not None:
        im = im.transpose(transpose)
    im = _drop_opaque_alpha(im)
    _check_cancel(cancel)
    mode: OutputMode = im.mode  # pyright: ignore[reportAssignmentType]
    return DecodedImage(
        data=im.tobytes(),
        width=im.width,
        height=im.height,
        mode=mode,
        source_width=source_size[0],
        source_height=source_size[1],
        source_format=source_format,
        is_animated=is_animated,
        file_version=version,
    )


def _oriented(size: tuple[int, int], orientation: int) -> tuple[int, int]:
    return (size[1], size[0]) if orientation in _SWAPS_AXES else size


def _decode_size(
    stored_size: tuple[int, int], orientation: int, request: DecodeRequest
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the (oriented full size, stored-orientation output size) for a request."""
    oriented = _oriented(stored_size, orientation)
    out = fit_size(oriented, _clamp_target(request.target_size), request.policy.upscale)
    return oriented, _oriented(out, orientation)


def _pillow_formats(path: Path) -> list[str]:
    """Return the Pillow plugins that may open a file, loading the JPEG XL plugin if needed."""
    ext = viewer_ext(path)
    if ext == ".jxl":
        try:
            import pillow_jxl  # noqa: F401 # pyright: ignore[reportUnusedImport]
        except ImportError as e:
            raise _DecodeError(DecodeErrorKind.CODEC_UNAVAILABLE, str(e)) from e
    Image.init()
    if ext == ".avif" and "AVIF" not in Image.OPEN:
        # Pillow only reads AVIF natively since 11.2, and pillow-heif has no AVIF decoder.
        raise _DecodeError(DecodeErrorKind.CODEC_UNAVAILABLE, "Pillow has no AVIF plugin")
    return [f for f in PILLOW_FORMATS if f in Image.OPEN]


def _decode_pillow(
    path: Path, request: DecodeRequest, version: FileVersion | None, cancel: CancelCheck | None
) -> DecodedImage:
    formats = _pillow_formats(request.path)
    try:
        with Image.open(path, formats=formats) as im:
            return _decode_opened_pillow(im, request, version, cancel)
    except UnidentifiedImageError as e:
        # The extension is supported, so content no plugin recognizes is treated as corrupt.
        raise _DecodeError(DecodeErrorKind.CORRUPT, str(e)) from e
    except Image.DecompressionBombError as e:
        raise _DecodeError(DecodeErrorKind.OVER_BUDGET, str(e)) from e


def _decode_opened_pillow(
    im: Image.Image,
    request: DecodeRequest,
    version: FileVersion | None,
    cancel: CancelCheck | None,
) -> DecodedImage:
    source_format = im.format or "unknown"
    # NOTE: `is_animated` stops after the second frame, while `n_frames` may scan the whole file.
    is_animated = bool(getattr(im, "is_animated", False))
    orientation = 1
    if request.policy.apply_orientation:
        try:
            orientation = int(im.getexif().get(_EXIF_ORIENTATION_TAG, 1) or 1)
        except Exception:
            orientation = 1
    stored_size = im.size
    oriented, out = _decode_size(stored_size, orientation, request)

    # Let decoders that can scale while decoding (e.g. JPEG) skip the full-size image.
    if out != stored_size and source_format in ("JPEG", "MPO"):
        im.draft(None, out)

    bpp = _PILLOW_BYTES_PER_PIXEL.get(im.mode, 4)
    loaded_pixels = im.size[0] * im.size[1]
    # Estimate the loaded buffer, plus a 32-bit copy for mode conversion or resampling, plus
    # the output sized buffers, plus the conversion arrays of high bit depth images (which are
    # made at the output size).
    estimate = loaded_pixels * (bpp + 4) + out[0] * out[1] * OUTPUT_BYTES_PER_PIXEL
    if _is_high_bit_depth(im.mode):
        estimate += out[0] * out[1] * HIGH_BIT_DEPTH_BYTES_PER_PIXEL
    _check_budget(estimate, request.max_decode_bytes, "Decoding this image")
    _check_cancel(cancel)

    im.load()
    _check_cancel(cancel)
    return _finish(im, out, orientation, oriented, source_format, is_animated, version, cancel)


def _decode_heif(
    path: Path, request: DecodeRequest, version: FileVersion | None, cancel: CancelCheck | None
) -> DecodedImage:
    try:
        import pillow_heif  # pyright: ignore[reportMissingTypeStubs]
    except ImportError as e:
        raise _DecodeError(DecodeErrorKind.CODEC_UNAVAILABLE, str(e)) from e

    try:
        heif_file = pillow_heif.open_heif(path, convert_hdr_to_8bit=True)
    except (ValueError, RuntimeError, OSError) as e:
        raise _DecodeError(DecodeErrorKind.CORRUPT, str(e)) from e

    # NOTE: libheif already applies the HEIF transformations (rotation/mirroring) while decoding,
    # so EXIF orientation is not applied a second time.
    stored_size: tuple[int, int] = tuple(heif_file.size)  # pyright: ignore[reportAssignmentType]
    oriented, out = _decode_size(stored_size, 1, request)
    # libheif's decoded buffer and its Pillow copy, plus the output sized buffers.
    _check_budget(
        stored_size[0] * stored_size[1] * 8 + out[0] * out[1] * OUTPUT_BYTES_PER_PIXEL,
        request.max_decode_bytes,
        "Decoding this image",
    )
    _check_cancel(cancel)
    try:
        im = heif_file.to_pillow()
    except (ValueError, RuntimeError, OSError) as e:
        raise _DecodeError(DecodeErrorKind.CORRUPT, str(e)) from e
    _check_cancel(cancel)
    return _finish(im, out, 1, oriented, "HEIF", len(heif_file) > 1, version, cancel)


def _decode_raw(
    path: Path, request: DecodeRequest, version: FileVersion | None, cancel: CancelCheck | None
) -> DecodedImage:
    try:
        import rawpy
    except ImportError as e:
        raise _DecodeError(DecodeErrorKind.CODEC_UNAVAILABLE, str(e)) from e

    try:
        with rawpy.imread(str(path)) as raw:
            sizes = raw.sizes
            flip = int(sizes.flip)
            # LibRaw's output size before and after applying the camera orientation.
            full = (int(sizes.width), int(sizes.height))
            oriented = (full[1], full[0]) if flip in (5, 6) else full
            target = _clamp_target(request.target_size)
            out_oriented = fit_size(oriented, target, request.policy.upscale)

            # Half-size demosaicing is 4x cheaper and is still large enough for small targets.
            half_size = out_oriented[0] * 2 <= oriented[0] and out_oriented[1] * 2 <= oriented[1]
            work_w, work_h = int(sizes.iwidth), int(sizes.iheight)
            if half_size and work_w >= full[0] and work_h >= full[1]:
                work_w, work_h = (work_w + 1) // 2, (work_h + 1) // 2
            # LibRaw's working image (4 x 16-bit per pixel) with its 8-bit RGB output, or that
            # output with its 32-bit Pillow copy, whichever is larger; plus the output sized
            # buffers.
            developed = (work_w * work_h) if half_size else oriented[0] * oriented[1]
            _check_budget(
                max(work_w * work_h * 8 + developed * 3, developed * 7)
                + out_oriented[0] * out_oriented[1] * OUTPUT_BYTES_PER_PIXEL,
                request.max_decode_bytes,
                "Developing this RAW image",
            )
            _check_cancel(cancel)

            rgb = raw.postprocess(
                use_camera_wb=request.policy.raw_camera_white_balance,
                half_size=half_size,
                output_bps=8,
            )
    except _DecodeError:
        raise
    except Exception as e:
        # rawpy raises several LibRaw* exception types that aren't all exported consistently.
        name = type(e).__name__
        kind = DecodeErrorKind.UNSUPPORTED if "Unsupported" in name else DecodeErrorKind.CORRUPT
        raise _DecodeError(kind, f"{name}: {e}") from e

    _check_cancel(cancel)
    im = Image.fromarray(np.ascontiguousarray(rgb))
    del rgb
    # Fit the actual developed size, which can differ slightly from LibRaw's reported size.
    out = fit_size(im.size, target, request.policy.upscale and not half_size)
    return _finish(im, out, 1, oriented, "RAW", is_animated=False, version=version, cancel=cancel)


_DECODERS: dict[
    ViewerCodec,
    Callable[[Path, DecodeRequest, FileVersion | None, CancelCheck | None], DecodedImage],
] = {
    ViewerCodec.PILLOW: _decode_pillow,
    ViewerCodec.HEIF: _decode_heif,
    ViewerCodec.RAW: _decode_raw,
}


def decode_image(request: DecodeRequest, cancel: CancelCheck | None = None) -> DecodeOutcome:
    """Decode an image for display, never raising for expected failures.

    The file is opened once; the returned image and its attributes come from the same read.
    All file handles and temporary files are closed before returning.

    Args:
        request: The file and constraints to decode with.
        cancel: Polled between decoding stages. Native codec calls can't be interrupted.
    """
    path = request.path
    codec = codec_for(path)
    if codec is None:
        return DecodeFailure(DecodeErrorKind.UNSUPPORTED, path, "Unsupported file type")

    version = FileVersion.of(path)
    if version is None:
        return DecodeFailure(DecodeErrorKind.MISSING, path, "File not found")
    if request.expected_version is not None and version != request.expected_version:
        return DecodeFailure(DecodeErrorKind.CHANGED, path, "File changed")

    try:
        _check_cancel(cancel)
        if compressed_inner_ext(path) is None:
            return _DECODERS[codec](path, request, version, cancel)
        with decompressed(path) as inner_path:
            if inner_path is None:
                raise _DecodeError(DecodeErrorKind.CORRUPT, "Could not decompress file")
            # Everything is decoded inside the context, before the temporary file is removed.
            return _DECODERS[codec](inner_path, request, version, cancel)
    except _DecodeError as e:
        return DecodeFailure(e.kind, path, e.message)
    except FileNotFoundError as e:
        return DecodeFailure(DecodeErrorKind.MISSING, path, str(e))
    except MemoryError as e:
        return DecodeFailure(DecodeErrorKind.OVER_BUDGET, path, f"Out of memory: {e}")
    except (OSError, SyntaxError, ValueError, EOFError) as e:
        # Pillow raises these for truncated or malformed files.
        return DecodeFailure(DecodeErrorKind.CORRUPT, path, f"{type(e).__name__}: {e}")
