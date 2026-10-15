# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only


import subprocess
import sys
from compression import gzip
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageFile

from tagstudio.core.viewer import decoding
from tagstudio.core.viewer.decoding import (
    DecodedImage,
    DecodeErrorKind,
    DecodeFailure,
    DecodeRequest,
    FileVersion,
    decode_image,
    fit_size,
)

RED = (255, 0, 0)
BLUE = (0, 0, 255)


def _ok(outcome: decoding.DecodeOutcome) -> DecodedImage:
    assert isinstance(outcome, DecodedImage), outcome
    return outcome


def _fail(outcome: decoding.DecodeOutcome, kind: DecodeErrorKind) -> DecodeFailure:
    assert isinstance(outcome, DecodeFailure), outcome
    assert outcome.kind == kind, outcome
    return outcome


def _pixel(image: DecodedImage, x: int, y: int) -> tuple[int, ...]:
    c = image.channels
    offset = (y * image.width + x) * c
    return tuple(image.data[offset : offset + c])


def _half_and_half(size: tuple[int, int]) -> Image.Image:
    """An image whose left half is red and right half is blue."""
    im = Image.new("RGB", size, BLUE)
    im.paste(RED, (0, 0, size[0] // 2, size[1]))
    return im


@pytest.mark.parametrize(
    ["source", "box", "expected"],
    [
        ((4000, 3000), (1000, 1000), (1000, 750)),
        ((3000, 4000), (1000, 1000), (750, 1000)),
        ((200, 100), (1000, 1000), (200, 100)),  # Small images aren't enlarged
        ((1, 10000), (100, 100), (1, 100)),
        ((0, 0), (100, 100), (1, 1)),
    ],
)
def test_fit_size(source: tuple[int, int], box: tuple[int, int], expected: tuple[int, int]):
    assert fit_size(source, box) == expected


def test_fit_size_upscale():
    assert fit_size((200, 100), (1000, 1000), upscale=True) == (1000, 500)


@pytest.mark.parametrize(
    ["size", "expected"], [((4000, 1000), (800, 200)), ((1000, 4000), (200, 800))]
)
def test_landscape_and_portrait(tmp_path: Path, size: tuple[int, int], expected: tuple[int, int]):
    path = tmp_path / "a.png"
    Image.new("RGB", size, RED).save(path)
    image = _ok(decode_image(DecodeRequest(path, (800, 800))))
    assert (image.width, image.height) == expected
    assert (image.source_width, image.source_height) == size
    assert image.mode == "RGB"
    assert not image.is_full_resolution
    assert image.nbytes == expected[0] * expected[1] * 3


def test_small_image_is_full_resolution(tmp_path: Path):
    path = tmp_path / "a.png"
    Image.new("RGB", (30, 20), RED).save(path)
    image = _ok(decode_image(DecodeRequest(path, (800, 800))))
    assert (image.width, image.height) == (30, 20)
    assert image.is_full_resolution
    assert image.file_version == FileVersion.of(path)


def test_exif_orientation_is_applied(tmp_path: Path):
    path = tmp_path / "a.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # Rotate 90 degrees clockwise to display
    _half_and_half((400, 200)).save(path, exif=exif.tobytes(), quality=95)

    image = _ok(decode_image(DecodeRequest(path, (1000, 1000))))
    assert (image.width, image.height) == (200, 400)
    assert (image.source_width, image.source_height) == (200, 400)
    # The left (red) half of the stored image ends up on top.
    top = _pixel(image, 100, 10)
    bottom = _pixel(image, 100, 390)
    assert top[0] > 200 and top[2] < 60
    assert bottom[2] > 200 and bottom[0] < 60


def test_exif_orientation_with_reduced_size(tmp_path: Path):
    path = tmp_path / "a.jpg"
    exif = Image.Exif()
    exif[0x0112] = 8
    _half_and_half((4000, 2000)).save(path, exif=exif.tobytes())
    image = _ok(decode_image(DecodeRequest(path, (500, 500))))
    assert (image.width, image.height) == (250, 500)
    assert (image.source_width, image.source_height) == (2000, 4000)


def test_transparency_is_kept(tmp_path: Path):
    path = tmp_path / "a.png"
    Image.new("RGBA", (40, 20), (0, 255, 0, 128)).save(path)
    image = _ok(decode_image(DecodeRequest(path, (100, 100))))
    assert image.mode == "RGBA"
    assert image.has_alpha
    assert _pixel(image, 5, 5) == (0, 255, 0, 128)  # No background composited


def test_opaque_alpha_channel_is_dropped(tmp_path: Path):
    path = tmp_path / "a.png"
    Image.new("RGBA", (40, 20), (0, 255, 0, 255)).save(path)
    image = _ok(decode_image(DecodeRequest(path, (100, 100))))
    assert image.mode == "RGB"


def test_palette_with_transparency(tmp_path: Path):
    path = tmp_path / "a.gif"
    im = Image.new("P", (10, 10), 1)
    im.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
    im.putpixel((0, 0), 0)
    im.save(path, transparency=0)
    image = _ok(decode_image(DecodeRequest(path, (100, 100))))
    assert image.mode == "RGBA"
    assert _pixel(image, 0, 0)[3] == 0
    assert _pixel(image, 5, 5) == (255, 0, 0, 255)


def test_palette_without_transparency(tmp_path: Path):
    path = tmp_path / "a.png"
    Image.new("RGB", (10, 10), RED).convert("P").save(path)
    image = _ok(decode_image(DecodeRequest(path, (100, 100))))
    assert image.mode == "RGB"
    assert _pixel(image, 1, 1) == RED


def test_cmyk(tmp_path: Path):
    path = tmp_path / "a.jpg"
    Image.new("CMYK", (10, 10), (0, 255, 255, 0)).save(path, quality=100)
    image = _ok(decode_image(DecodeRequest(path, (100, 100))))
    assert image.mode == "RGB"
    r, g, b = _pixel(image, 5, 5)
    assert r > 200 and g < 60 and b < 60


def test_16_bit_grayscale_is_scaled(tmp_path: Path):
    path = tmp_path / "a.png"
    Image.fromarray(np.full((8, 8), 0x8000, dtype=np.uint16)).save(path)
    image = _ok(decode_image(DecodeRequest(path, (100, 100))))
    assert image.mode == "L"
    assert _pixel(image, 1, 1) == (0x80,)


@pytest.mark.parametrize(["low", "high"], [(100.0, 200.0), (-200.0, -100.0), (0.0, 1.0)])
def test_float_image_uses_its_own_range(tmp_path: Path, low: float, high: float):
    path = tmp_path / "a.tif"
    arr = np.linspace(low, high, 64, dtype=np.float32).reshape(8, 8)
    Image.fromarray(arr).save(path)
    image = _ok(decode_image(DecodeRequest(path, (100, 100))))
    assert image.mode == "L"
    assert min(image.data) == 0
    assert max(image.data) == 255


def test_float_image_with_non_finite_samples(tmp_path: Path):
    path = tmp_path / "a.tif"
    arr = np.full((4, 4), 5.0, dtype=np.float32)
    arr[0, 0], arr[0, 1], arr[0, 2] = np.nan, np.inf, 10.0
    Image.fromarray(arr).save(path)
    image = _ok(decode_image(DecodeRequest(path, (100, 100))))
    assert _pixel(image, 2, 0) == (255,)
    assert _pixel(image, 3, 3) == (0,)


def test_animated_image_shows_first_frame(tmp_path: Path):
    path = tmp_path / "a.gif"
    frames = [Image.new("RGB", (10, 10), c) for c in (RED, BLUE)]
    frames[0].save(path, save_all=True, append_images=frames[1:])
    image = _ok(decode_image(DecodeRequest(path, (100, 100))))
    assert image.is_animated
    r, _, b = _pixel(image, 5, 5)[:3]
    assert r > 200 and b < 60


def test_heif(tmp_path: Path):
    pillow_heif = pytest.importorskip("pillow_heif")
    path = tmp_path / "a.heic"
    pillow_heif.from_pillow(Image.new("RGB", (64, 32), RED)).save(str(path))
    image = _ok(decode_image(DecodeRequest(path, (100, 100))))
    assert (image.width, image.height) == (64, 32)
    assert image.source_format == "HEIF"


def test_jpeg_xl(tmp_path: Path):
    pytest.importorskip("pillow_jxl")
    path = tmp_path / "a.jxl"
    Image.new("RGBA", (32, 16), (0, 0, 255, 100)).save(path)
    image = _ok(decode_image(DecodeRequest(path, (100, 100))))
    assert image.mode == "RGBA"


def test_misnamed_file_is_decoded_by_content(tmp_path: Path):
    path = tmp_path / "actually_png.jpg"
    Image.new("RGB", (10, 10), RED).save(path, format="PNG")
    assert _ok(decode_image(DecodeRequest(path, (100, 100)))).source_format == "PNG"


def test_compressed_image(tmp_path: Path):
    source = tmp_path / "a.png"
    Image.new("RGBA", (20, 10), (1, 2, 3, 4)).save(source)
    path = tmp_path / "a.png.gz"
    path.write_bytes(gzip.compress(source.read_bytes()))
    image = _ok(decode_image(DecodeRequest(path, (100, 100))))
    assert _pixel(image, 0, 0) == (1, 2, 3, 4)


def test_missing_file(tmp_path: Path):
    _fail(decode_image(DecodeRequest(tmp_path / "nope.png", (10, 10))), DecodeErrorKind.MISSING)


def test_directory_is_missing(tmp_path: Path):
    folder = tmp_path / "folder.png"
    folder.mkdir()
    _fail(decode_image(DecodeRequest(folder, (10, 10))), DecodeErrorKind.MISSING)


def test_unsupported_file(tmp_path: Path):
    path = tmp_path / "a.txt"
    path.write_text("hi")
    _fail(decode_image(DecodeRequest(path, (10, 10))), DecodeErrorKind.UNSUPPORTED)


def test_corrupt_file(tmp_path: Path):
    path = tmp_path / "a.png"
    path.write_bytes(b"not an image at all")
    _fail(decode_image(DecodeRequest(path, (10, 10))), DecodeErrorKind.CORRUPT)


def test_over_budget(tmp_path: Path):
    path = tmp_path / "a.png"
    Image.new("RGB", (1000, 1000)).save(path)
    request = DecodeRequest(path, (100, 100), max_decode_bytes=1_000_000)
    _fail(decode_image(request), DecodeErrorKind.OVER_BUDGET)


def test_jpeg_reduced_decoding_fits_budget(tmp_path: Path):
    """JPEG can be decoded at a reduced scale, so a large JPEG fits a small budget."""
    path = tmp_path / "a.jpg"
    Image.new("RGB", (4000, 4000), RED).save(path)
    request = DecodeRequest(path, (500, 500), max_decode_bytes=4_000_000)
    image = _ok(decode_image(request))
    assert (image.width, image.height) == (500, 500)
    assert (image.source_width, image.source_height) == (4000, 4000)


def test_changed_file(tmp_path: Path):
    path = tmp_path / "a.png"
    Image.new("RGB", (10, 10)).save(path)
    stale = FileVersion(1, 1)
    _fail(decode_image(DecodeRequest(path, (10, 10), stale)), DecodeErrorKind.CHANGED)


def test_cancel(tmp_path: Path):
    path = tmp_path / "a.png"
    Image.new("RGB", (10, 10)).save(path)
    _fail(decode_image(DecodeRequest(path, (10, 10)), lambda: True), DecodeErrorKind.CANCELED)


def test_cancel_between_stages(tmp_path: Path):
    path = tmp_path / "a.png"
    Image.new("RGB", (10, 10)).save(path)
    calls = 0

    def cancel() -> bool:
        nonlocal calls
        calls += 1
        return calls > 2

    _fail(decode_image(DecodeRequest(path, (10, 10)), cancel), DecodeErrorKind.CANCELED)


def test_file_is_opened_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Attributes come from the same read as the pixels; no second open for the size."""
    path = tmp_path / "a.png"
    Image.new("RGB", (100, 50)).save(path)
    opens: list[Path] = []
    real_open = Image.open

    def spy(fp, *args, **kwargs):  # pyright: ignore
        opens.append(fp)
        return real_open(fp, *args, **kwargs)

    monkeypatch.setattr(Image, "open", spy)
    image = _ok(decode_image(DecodeRequest(path, (100, 100))))
    assert (image.source_width, image.source_height) == (100, 50)
    assert opens == [path]


def test_file_and_globals_are_unchanged(tmp_path: Path):
    path = tmp_path / "a.png"
    Image.new("RGB", (100, 50)).save(path)
    content = path.read_bytes()
    version = FileVersion.of(path)
    max_pixels = Image.MAX_IMAGE_PIXELS
    load_truncated = ImageFile.LOAD_TRUNCATED_IMAGES

    _ok(decode_image(DecodeRequest(path, (10, 10))))
    assert path.read_bytes() == content
    assert FileVersion.of(path) == version
    assert max_pixels == Image.MAX_IMAGE_PIXELS
    assert load_truncated == ImageFile.LOAD_TRUNCATED_IMAGES


def test_decoded_buffer_is_immutable(tmp_path: Path):
    path = tmp_path / "a.png"
    Image.new("RGB", (10, 10)).save(path)
    image = _ok(decode_image(DecodeRequest(path, (10, 10))))
    assert isinstance(image.data, bytes)
    with pytest.raises(AttributeError):
        image.width = 5  # pyright: ignore[reportAttributeAccessIssue]


# RAW ==========================================================================


class FakeRaw:
    """Stands in for `rawpy.RawPy`, recording how it was developed."""

    def __init__(self, width: int, height: int, flip: int = 0):
        self.sizes = SimpleNamespace(
            width=width, height=height, iwidth=width, iheight=height, flip=flip
        )
        self.postprocess_calls: list[dict[str, object]] = []

    def __enter__(self):
        return self

    def __exit__(self, *args: object):
        return False

    def postprocess(self, **kwargs: object):
        self.postprocess_calls.append(kwargs)
        w, h = self.sizes.width, self.sizes.height
        if kwargs.get("half_size"):
            w, h = w // 2, h // 2
        if self.sizes.flip in (5, 6):
            w, h = h, w
        return np.zeros((h, w, 3), dtype=np.uint8)


@pytest.fixture
def raw_file(tmp_path: Path) -> Path:
    path = tmp_path / "photo.cr2"
    path.write_bytes(b"fake raw")
    return path


def _patch_rawpy(monkeypatch: pytest.MonkeyPatch, raw: FakeRaw):
    import rawpy

    monkeypatch.setattr(rawpy, "imread", lambda _path: raw)  # pyright: ignore


def test_raw_is_developed_once_with_half_size(raw_file: Path, monkeypatch: pytest.MonkeyPatch):
    raw = FakeRaw(6000, 4000)
    _patch_rawpy(monkeypatch, raw)
    image = _ok(decode_image(DecodeRequest(raw_file, (1920, 1080))))
    assert len(raw.postprocess_calls) == 1
    assert raw.postprocess_calls[0]["half_size"] is True
    assert raw.postprocess_calls[0]["use_camera_wb"] is True
    assert (image.width, image.height) == (1620, 1080)
    assert (image.source_width, image.source_height) == (6000, 4000)


def test_raw_full_size_for_large_targets(raw_file: Path, monkeypatch: pytest.MonkeyPatch):
    raw = FakeRaw(6000, 4000)
    _patch_rawpy(monkeypatch, raw)
    _ok(decode_image(DecodeRequest(raw_file, (5000, 5000))))
    assert raw.postprocess_calls[0]["half_size"] is False


def test_raw_orientation(raw_file: Path, monkeypatch: pytest.MonkeyPatch):
    raw = FakeRaw(6000, 4000, flip=6)
    _patch_rawpy(monkeypatch, raw)
    image = _ok(decode_image(DecodeRequest(raw_file, (1000, 1000))))
    assert (image.source_width, image.source_height) == (4000, 6000)
    assert image.height == 1000


def test_raw_over_budget_is_not_developed(raw_file: Path, monkeypatch: pytest.MonkeyPatch):
    raw = FakeRaw(6000, 4000)
    _patch_rawpy(monkeypatch, raw)
    request = DecodeRequest(raw_file, (5000, 5000), max_decode_bytes=1_000_000)
    _fail(decode_image(request), DecodeErrorKind.OVER_BUDGET)
    assert raw.postprocess_calls == []


def test_raw_unsupported(raw_file: Path, monkeypatch: pytest.MonkeyPatch):
    import rawpy

    class LibRawFileUnsupportedError(Exception):
        pass

    def fail(_path: str):
        raise LibRawFileUnsupportedError("nope")

    monkeypatch.setattr(rawpy, "imread", fail)
    _fail(decode_image(DecodeRequest(raw_file, (10, 10))), DecodeErrorKind.UNSUPPORTED)


def test_real_rawpy_rejects_garbage(raw_file: Path):
    outcome = decode_image(DecodeRequest(raw_file, (10, 10)))
    assert isinstance(outcome, DecodeFailure)
    assert outcome.kind in (DecodeErrorKind.UNSUPPORTED, DecodeErrorKind.CORRUPT)


# Isolation ====================================================================


def test_viewer_core_does_not_import_qt():
    """The core modules must stay usable without Qt (checked in a clean interpreter)."""
    code = (
        "import sys\n"
        "import tagstudio.core.viewer.cache, tagstudio.core.viewer.decoding\n"
        "import tagstudio.core.viewer.formats, tagstudio.core.viewer.library_reader\n"
        "import tagstudio.core.viewer.scheduler, tagstudio.core.viewer.session\n"
        "bad = [m for m in sys.modules if m.startswith(('PySide6', 'tagstudio.qt'))]\n"
        "assert not bad, bad\n"
    )
    src = Path(__file__).parents[3] / "src"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(src)},
        check=False,
    )
    assert result.returncode == 0, result.stderr
