# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only


import threading
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtCore import QThread
from pytestqt.qtbot import QtBot

from tagstudio.core.viewer.decoding import (
    CancelCheck,
    DecodedImage,
    DecodeErrorKind,
    DecodeFailure,
    DecodeOutcome,
    DecodeRequest,
    decode_image,
)
from tagstudio.qt.utils.image_decode_runner import ImageDecodeRunner

TIMEOUT_MS = 5000


class BlockingDecode:
    """A decode function that waits until released or canceled."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.threads: list[threading.Thread] = []

    def __call__(self, request: DecodeRequest, cancel: CancelCheck | None) -> DecodeOutcome:
        self.threads.append(threading.current_thread())
        self.started.set()
        while not self.release.wait(0.005):
            if cancel is not None and cancel():
                return DecodeFailure(DecodeErrorKind.CANCELED, request.path)
        return decode_image(request, cancel)


@pytest.fixture
def image_path(tmp_path: Path) -> Path:
    path = tmp_path / "a.png"
    Image.new("RGB", (64, 32), (255, 0, 0)).save(path)
    return path


@pytest.fixture
def runner(qtbot: QtBot):
    runners: list[ImageDecodeRunner] = []

    def make(decode=decode_image) -> ImageDecodeRunner:  # pyright: ignore
        r = ImageDecodeRunner(decode)
        runners.append(r)
        return r

    yield make
    for r in runners:
        r.shutdown(timeout=1)


def test_success_is_delivered_on_gui_thread(qtbot: QtBot, runner, image_path: Path):  # pyright: ignore
    r: ImageDecodeRunner = runner()
    gui_thread = QThread.currentThread()
    received: list[tuple[int, DecodeOutcome, bool]] = []
    r.finished.connect(lambda i, o: received.append((i, o, QThread.currentThread() is gui_thread)))
    with qtbot.waitSignal(r.finished, timeout=TIMEOUT_MS):
        r.start(1, DecodeRequest(image_path, (100, 100)))
    request_id, outcome, on_gui_thread = received[0]
    assert request_id == 1
    assert isinstance(outcome, DecodedImage)
    assert on_gui_thread
    assert r.active_count == 0


def test_decode_runs_off_gui_thread(qtbot: QtBot, runner, image_path: Path):  # pyright: ignore
    decode = BlockingDecode()
    decode.release.set()
    r: ImageDecodeRunner = runner(decode)
    with qtbot.waitSignal(r.finished, timeout=TIMEOUT_MS):
        r.start(1, DecodeRequest(image_path, (100, 100)))
    assert decode.threads[0] is not threading.main_thread()


def test_exception_reports_failure_and_runner_keeps_working(
    qtbot: QtBot,
    runner,  # pyright: ignore
    image_path: Path,
):
    calls = 0

    def flaky(request: DecodeRequest, cancel: CancelCheck | None) -> DecodeOutcome:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("boom")
        return decode_image(request, cancel)

    r: ImageDecodeRunner = runner(flaky)
    with qtbot.waitSignal(r.finished, timeout=TIMEOUT_MS) as blocker:
        r.start(1, DecodeRequest(image_path, (10, 10)))
    outcome = blocker.args[1]  # pyright: ignore[reportOptionalSubscript]
    assert isinstance(outcome, DecodeFailure)
    assert outcome.kind == DecodeErrorKind.CORRUPT

    with qtbot.waitSignal(r.finished, timeout=TIMEOUT_MS) as blocker:
        r.start(2, DecodeRequest(image_path, (10, 10)))
    assert isinstance(blocker.args[1], DecodedImage)  # pyright: ignore[reportOptionalSubscript]


def test_cancel_finishes_exactly_once(qtbot: QtBot, runner, image_path: Path):  # pyright: ignore
    decode = BlockingDecode()
    r: ImageDecodeRunner = runner(decode)
    received: list[int] = []
    r.finished.connect(lambda i, o: received.append(i))

    r.start(1, DecodeRequest(image_path, (10, 10)))
    assert decode.started.wait(5)
    with qtbot.waitSignal(r.finished, timeout=TIMEOUT_MS) as blocker:
        r.cancel(1)
        r.cancel(1)
    outcome = blocker.args[1]  # pyright: ignore[reportOptionalSubscript]
    assert isinstance(outcome, DecodeFailure)
    assert outcome.kind == DecodeErrorKind.CANCELED

    # Wait for another request to make sure no second signal for request 1 is queued.
    decode.release.set()
    with qtbot.waitSignal(r.finished, timeout=TIMEOUT_MS):
        r.start(2, DecodeRequest(image_path, (10, 10)))
    assert received == [1, 2]
    assert r.max_active == 1


def test_duplicate_start_is_rejected(runner, image_path: Path):  # pyright: ignore
    decode = BlockingDecode()
    r: ImageDecodeRunner = runner(decode)
    r.start(1, DecodeRequest(image_path, (10, 10)))
    with pytest.raises(ValueError):
        r.start(1, DecodeRequest(image_path, (10, 10)))
    decode.release.set()


def test_nothing_is_emitted_after_shutdown(qtbot: QtBot, runner, image_path: Path):  # pyright: ignore
    decode = BlockingDecode()
    r: ImageDecodeRunner = runner(decode)
    r.start(1, DecodeRequest(image_path, (10, 10)))
    assert decode.started.wait(5)

    r.shutdown()  # Doesn't wait for the running decode
    assert r.is_shut_down
    with qtbot.assertNotEmitted(r.finished, wait=200):
        decode.release.set()
    r.start(2, DecodeRequest(image_path, (10, 10)))  # Ignored
    assert r.started_count == 1
