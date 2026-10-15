# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only

# NOTE: These tests drive the scheduler with a fake executor; no Qt, threads, or database.


from dataclasses import dataclass, field

from tagstudio.core.viewer.scheduler import LoadRequest, LoadScheduler


@dataclass
class FakeExecutor:
    """Carries out scheduler actions and records them."""

    scheduler: LoadScheduler = field(default_factory=LoadScheduler)
    running: list[LoadRequest] = field(default_factory=list)
    cancels: list[int] = field(default_factory=list)
    applied: list[tuple[int, str]] = field(default_factory=list)
    max_parallel: int = 0

    def _start(self, request: LoadRequest | None):
        if request is not None:
            self.running.append(request)
            self.max_parallel = max(self.max_parallel, len(self.running))

    def _cancel(self, request: LoadRequest | None):
        if request is not None:
            self.cancels.append(request.request_id)

    def submit(self, key: str) -> LoadRequest:
        actions = self.scheduler.submit(key)
        self._cancel(actions.cancel)
        self._start(actions.start)
        return actions.request

    def complete(self, request: LoadRequest) -> bool:
        self.running.remove(request)
        actions = self.scheduler.finish(request.request_id)
        if actions.apply:
            self.applied.append((request.request_id, str(request.key)))
        self._start(actions.start)
        return actions.apply


def test_single_request_runs_and_applies():
    ex = FakeExecutor()
    a = ex.submit("A")
    assert ex.running == [a]
    assert ex.complete(a)
    assert ex.applied == [(a.request_id, "A")]
    assert ex.scheduler.is_idle


def test_rapid_submissions_keep_one_running_and_latest_pending():
    ex = FakeExecutor()
    a = ex.submit("A")
    ex.submit("B")
    ex.submit("C")
    d = ex.submit("D")
    assert ex.running == [a]
    assert ex.scheduler.pending == d
    assert ex.cancels == [a.request_id]  # Cancel is requested only once

    assert not ex.complete(a)  # A is no longer wanted
    assert ex.running == [d]  # B and C never ran
    assert ex.complete(d)
    assert ex.applied == [(d.request_id, "D")]
    assert ex.max_parallel == 1
    assert ex.scheduler.stats.started == 2


def test_back_and_forth_does_not_apply_old_result():
    """A -> B -> A: the first A's result must not be shown."""
    ex = FakeExecutor()
    a1 = ex.submit("A")
    ex.submit("B")
    a2 = ex.submit("A")
    assert a2.request_id != a1.request_id
    assert not ex.complete(a1)
    assert ex.running == [a2]
    assert ex.complete(a2)
    assert ex.applied == [(a2.request_id, "A")]


def test_same_key_reuses_running_request():
    ex = FakeExecutor()
    a = ex.submit("A")
    again = ex.submit("A")
    assert again == a
    assert ex.scheduler.stats.reused == 1
    assert ex.complete(a)


def test_finish_is_idempotent_and_ignores_unknown_ids():
    ex = FakeExecutor()
    a = ex.submit("A")
    b = ex.submit("B")
    assert not ex.scheduler.finish(12345).apply
    assert ex.scheduler.running == a

    ex.complete(a)
    assert ex.scheduler.running == b
    # A second finish for A must not release B's slot.
    assert not ex.scheduler.finish(a.request_id).apply
    assert ex.scheduler.running == b


def test_out_of_order_completion_is_ignored():
    ex = FakeExecutor()
    a = ex.submit("A")
    b = ex.submit("B")
    # B hasn't started; a stray finish for it changes nothing.
    assert not ex.scheduler.finish(b.request_id).apply
    assert ex.scheduler.running == a
    assert ex.scheduler.pending == b


def test_satisfy_drops_pending_and_cancels_running():
    ex = FakeExecutor()
    a = ex.submit("A")
    ex.submit("B")
    cancel = ex.scheduler.satisfy()  # e.g. C was found in the cache
    assert cancel is None  # A already had a cancel requested
    assert ex.scheduler.pending is None
    assert not ex.complete(a)
    assert ex.running == []
    assert ex.applied == []


def test_invalidate_keeps_slot_until_finished():
    """Closing and reopening never runs two decodes at once."""
    ex = FakeExecutor()
    a = ex.submit("A")
    ex.submit("B")
    cancel = ex.scheduler.invalidate()  # viewer closed
    assert cancel is None  # A was already asked to cancel when B was submitted
    assert ex.scheduler.pending is None

    c = ex.submit("C")  # viewer reopened
    assert ex.running == [a]
    assert ex.scheduler.pending == c
    assert not ex.complete(a)
    assert ex.running == [c]
    assert ex.complete(c)
    assert ex.max_parallel == 1


def test_invalidate_rejects_result_of_previous_generation():
    ex = FakeExecutor()
    a = ex.submit("A")
    cancel = ex.scheduler.invalidate()
    assert cancel == a
    assert not ex.complete(a)
    assert ex.applied == []


def test_failed_request_still_releases_slot():
    ex = FakeExecutor()
    a = ex.submit("A")
    b = ex.submit("B")
    # Whether A succeeded, failed, or was canceled, finish() is called once.
    ex.complete(a)
    assert ex.running == [b]
