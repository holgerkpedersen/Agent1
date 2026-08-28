"""Tests for the autonomous self-improvement driver.

Covers the safety rails that make "fully autonomous" safe:
- refuses to run without AGENT_AUTONOMOUS=1 (or --auto),
- honours the STOP_AUTONOMOUS kill-switch between iterations,
- commits only an ACCEPTED repair and stops on any non-accepted verdict.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest import mock

import pytest

import scripts.autonomous_self_improve as drv


def _fake_summary(verdict: str, *, repair: str = "tool-interface-error-detail",
                  accepted: bool = False) -> dict:
    return {
        "verdict": verdict,
        "proposed_repair": repair,
        "accepted": accepted,
        "tests_passed": True,
        "security_passed": True,
        "baseline_rate": None,
        "post_rate": None,
    }


def test_refuses_without_autonomous_env(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_AUTONOMOUS", raising=False)
    rc = drv.main(["--model", "x"])
    assert rc == 2


def test_engages_with_auto_flag(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_AUTONOMOUS", raising=False)
    calls = []

    def fake_run_iteration(iteration, *, model, profile, trace_dir, output_dir,
                            no_benchmark=False):
        calls.append(iteration)
        return _fake_summary("no_repair_catalogued")

    with mock.patch.object(drv, "run_iteration", fake_run_iteration), \
         mock.patch.object(drv, "_git", _noop_git), \
         mock.patch.object(drv, "_stop_requested", lambda: False):
        rc = drv.main(["--auto", "--model", "x", "--max-iterations", "3"])
    # One iteration, then stop because no repair was catalogued.
    assert rc == 0
    assert calls == [1]


def _accepted_summary() -> dict:
    return _fake_summary("accepted", accepted=True)


def _noop_git(*_args, **_kw):
    return subprocess.CompletedProcess([], 0, "", "")


def test_stops_on_kill_switch(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_AUTONOMOUS", raising=False)
    stops = iter([True, False])  # stop before iteration 1
    with mock.patch.object(drv, "_stop_requested", lambda: next(stops)), \
         mock.patch.object(drv, "_git", _noop_git), \
         mock.patch.object(drv, "run_iteration", lambda *a, **k: _accepted_summary()):
        # --source catalog so the kill-switch is consulted before the catalog
        # loop (the issue loop would consume the first stop flag otherwise).
        rc = drv.main(["--auto", "--source", "catalog", "--max-iterations", "3"])
    assert rc == 0


def test_no_benchmark_disables_model(tmp_path):
    """--no-benchmark forces the model to None so the loop never spawns the
    benchmark subprocess; the offline harness-quality gate is used instead."""
    import os as _os
    _os.environ.pop("AGENT_AUTONOMOUS", None)
    seen = {}

    def fake_run_iteration(iteration, *, model, profile, trace_dir, output_dir,
                           no_benchmark=False):
        # Mirror the real run_iteration: --no-benchmark forces model=None.
        eff_model = None if no_benchmark else model
        seen["model"] = eff_model
        seen["no_benchmark"] = no_benchmark
        return _fake_summary("no_repair_catalogued")

    with mock.patch.object(drv, "run_iteration", fake_run_iteration), \
         mock.patch.object(drv, "_git", _noop_git), \
         mock.patch.object(drv, "_stop_requested", lambda: False):
        rc = drv.main(["--auto", "--model", "qwen3", "--no-benchmark"])
    assert rc == 0
    assert seen["no_benchmark"] is True
    assert seen["model"] is None


def test_commits_accepted_repair_then_stops(monkeypatch, tmp_path):
    """An accepted repair is committed; the loop then stops (single iteration)."""
    monkeypatch.delenv("AGENT_AUTONOMOUS", raising=False)
    git_calls: list[list[str]] = []
    has_changes = {"v": True}

    def fake_git(args, check=True):
        git_calls.append(args)
        if args[:2] == ["status", "--porcelain"]:
            changed = " M agent_core/llm/tool_loop.py\n" if has_changes["v"] else ""
            return subprocess.CompletedProcess(args, 0, changed)
        return subprocess.CompletedProcess(args, 0, "", "")

    with mock.patch.object(drv, "_git", fake_git), \
         mock.patch.object(drv, "_stop_requested", lambda: False), \
         mock.patch.object(drv, "run_iteration", lambda *a, **k: _accepted_summary()):
        rc = drv.main(["--auto", "--max-iterations", "3"])
    assert rc == 0
    # Accepted -> add + commit happened.
    assert any(c[:1] == ["add"] for c in git_calls)
    assert any(c[:1] == ["commit"] for c in git_calls)


def test_stops_on_rejected_verdict(monkeypatch, tmp_path):
    """A rejected verdict ends the loop (no second iteration)."""
    monkeypatch.delenv("AGENT_AUTONOMOUS", raising=False)
    iters: list[int] = []

    def fake_iter(iteration, **k):
        iters.append(iteration)
        return _fake_summary("rejected_and_reverted")

    with mock.patch.object(drv, "_stop_requested", lambda: False), \
         mock.patch.object(drv, "_git", _noop_git), \
         mock.patch.object(drv, "run_iteration", fake_iter):
        rc = drv.main(["--auto", "--max-iterations", "5"])
    assert rc == 0
    assert iters == [1]


def test_iteration_exception_is_caught_and_stops(monkeypatch, tmp_path):
    """A non-SystemExit exception in an iteration must NOT crash the driver
    with a traceback; it should be logged, the checkpoint restored, and the
    loop should stop (return 1)."""
    monkeypatch.delenv("AGENT_AUTONOMOUS", raising=False)
    git_calls: list[list[str]] = []

    def spy_git(args, check=True):
        git_calls.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    def boom(iteration, **k):
        raise RuntimeError("boom in iteration")

    with mock.patch.object(drv, "_stop_requested", lambda: False), \
         mock.patch.object(drv, "_git", spy_git), \
         mock.patch.object(drv, "run_iteration", boom):
        rc = drv.main(["--auto", "--max-iterations", "5"])
    assert rc == 1
    # The checkpoint stash was popped so the tree is not left dirty.
    assert any(c[:2] == ["stash", "pop"] for c in git_calls)


def test_keyboard_interrupt_restores_checkpoint_and_reraises(monkeypatch):
    """A Ctrl+C (KeyboardInterrupt, a BaseException) during an iteration must
    restore the checkpoint so the tree is not left dirty, and must still
    propagate so the process actually stops."""
    monkeypatch.delenv("AGENT_AUTONOMOUS", raising=False)
    git_calls: list[list[str]] = []

    def spy_git(args, check=True):
        git_calls.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    def interrupt(iteration, **k):
        raise KeyboardInterrupt()

    with mock.patch.object(drv, "_stop_requested", lambda: False), \
         mock.patch.object(drv, "_git", spy_git), \
         mock.patch.object(drv, "run_iteration", interrupt):
        with pytest.raises(KeyboardInterrupt):
            drv.main(["--auto", "--max-iterations", "5"])
    # The checkpoint stash was popped before re-raising.
    assert any(c[:2] == ["stash", "pop"] for c in git_calls)


def test_git_missing_raises_clear_error(monkeypatch):
    """If git is not on PATH, _git must raise a clear RuntimeError — not a bare
    FileNotFoundError pointing at an opaque call site (e.g. line 146)."""
    monkeypatch.delenv("AGENT_AUTONOMOUS", raising=False)

    def missing_run(*_a, **_k):
        raise FileNotFoundError(2, "No such file or directory", "git")

    # Patch the real subprocess.run that _git() delegates to, so the
    # try/except FileNotFoundError branch inside _git() is exercised.
    with mock.patch.object(drv.subprocess, "run", missing_run), \
         mock.patch.object(drv, "_stop_requested", lambda: False), \
         mock.patch.object(drv, "run_iteration", lambda *a, **k: _accepted_summary()):
        with pytest.raises(RuntimeError, match="git executable not found"):
            drv.main(["--auto", "--max-iterations", "3"])


def test_failed_stash_push_does_not_pop_unrelated_stash(monkeypatch):
    """A `git stash push` that fails (rc != 0) must NOT be followed by a
    `git stash pop` — otherwise an unrelated stash would be popped and the
    tree corrupted."""
    monkeypatch.delenv("AGENT_AUTONOMOUS", raising=False)
    git_calls: list[list[str]] = []

    def fake_git(args, check=True):
        git_calls.append(args)
        # The checkpoint push "fails" (e.g. nothing to stash / no initial
        # commit).  Every other git call (status/add/commit) succeeds.
        if args[:2] == ["stash", "push"]:
            return subprocess.CompletedProcess(args, 1, "", "nothing to stash")
        return subprocess.CompletedProcess(args, 0, "", "")

    with mock.patch.object(drv, "_stop_requested", lambda: False), \
         mock.patch.object(drv, "_git", fake_git), \
         mock.patch.object(drv, "run_iteration", lambda *a, **k: _accepted_summary()):
        rc = drv.main(["--auto", "--source", "catalog", "--max-iterations", "3"])
    # The fake iteration returns "accepted" but the tree is unchanged, so the
    # driver fails closed (no commit possible) rather than spinning.
    assert rc == 1
    # No stash pop should have been issued because no checkpoint was created.
    assert not any(c[:2] == ["stash", "pop"] for c in git_calls)
    assert not any(c[:2] == ["stash", "drop"] for c in git_calls)


def test_commit_stages_only_repair_files_not_blanket_add(monkeypatch, tmp_path):
    """Regression: the autonomous driver must NOT `git add -A` (which would
    sweep unrelated/scratch files into the commit).  It stages ONLY the repair's
    own source file(s)."""
    monkeypatch.delenv("AGENT_AUTONOMOUS", raising=False)
    git_calls: list[list[str]] = []

    def fake_git(args, check=True):
        git_calls.append(args)
        if args[:1] == ["status"] and "--porcelain" in args:
            # Only the repair's file is reported as changed; a scratch file
            # (_tmp_zen_out.txt) exists in the real tree but must be ignored.
            if "tool_loop.py" in args[-1]:
                return subprocess.CompletedProcess(
                    args, 0, " M agent_core/llm/tool_loop.py\n"
                )
            return subprocess.CompletedProcess(args, 0, "")
        return subprocess.CompletedProcess(args, 0, "", "")

    with mock.patch.object(drv, "_stop_requested", lambda: False), \
         mock.patch.object(drv, "_git", fake_git), \
         mock.patch.object(drv, "run_iteration", lambda *a, **k: _accepted_summary()):
        rc = drv.main(["--auto", "--max-iterations", "3"])
    assert rc == 0
    # No blanket add.
    assert not any(c == ["add", "-A"] for c in git_calls)
    # Every accepted iteration stages ONLY the repair's file via a scoped
    # `git add -- <file>` (never a blanket `git add -A`).  The fake iteration
    # returns "accepted" each round without mutating the tree, so we expect
    # one scoped add per committed iteration.
    add_calls = [c for c in git_calls if c[:1] == ["add"]]
    norm = [tuple(p.replace("\\", "/") for p in c) for c in add_calls]
    assert all(c == ("add", "--", "agent_core/llm/tool_loop.py") for c in norm)
    assert len(norm) >= 1
    assert any(c[:1] == ["commit"] for c in git_calls)


def test_commit_skips_when_no_repair_files_changed(monkeypatch, tmp_path):
    """If the accepted repair's own files are NOT in the change set (e.g. it
    was a no-op or the diff is unrelated), the driver must NOT commit — and
    must NOT fall back to a blanket `git add -A`."""
    monkeypatch.delenv("AGENT_AUTONOMOUS", raising=False)
    git_calls: list[list[str]] = []

    def fake_git(args, check=True):
        git_calls.append(args)
        if args[:2] == ["status", "--porcelain", "--"]:
            # The repair's file is NOT changed; only an unrelated scratch file.
            return subprocess.CompletedProcess(args, 0, "")
        return subprocess.CompletedProcess(args, 0, "", "")

    with mock.patch.object(drv, "_stop_requested", lambda: False), \
         mock.patch.object(drv, "_git", fake_git), \
         mock.patch.object(drv, "run_iteration", lambda *a, **k: _accepted_summary()):
        rc = drv.main(["--auto", "--source", "catalog", "--max-iterations", "3"])
    # Accepted but nothing to commit -> fail closed (rc 1), not a silent 0.
    assert rc == 1
    # Nothing staged, nothing committed.
    assert not any(c[:1] == ["add"] for c in git_calls)
    assert not any(c[:1] == ["commit"] for c in git_calls)


def test_repair_files_returns_empty_for_unknown_repair():
    """_repair_files returns () for an unknown repair id so the driver can
    refuse a blanket add rather than guessing.  This is the safety backstop
    that prevents committing unrelated work."""
    assert drv._repair_files("does-not-exist") == ()
    assert drv._repair_files(None) == ()


def test_gate_runs_the_real_functional_suite(monkeypatch):
    """The autonomous test gate must run the real functional suite (not the
    HarnessFix loop/repair self-tests, which are isolated via the root conftest
    fixture so they never mutate the real tree mid-gate)."""
    import subprocess as _sp

    from harnessfix import gates

    captured: dict = {}

    def fake_run(args, **kw):
        captured["args"] = list(args)
        return _sp.CompletedProcess(args, 0, "passed\n", "")

    monkeypatch.setattr(gates.subprocess, "run", fake_run)
    gates.collect_test_failures()
    args = captured["args"]
    # The gate invokes pytest as a subprocess (the real functional suite).
    assert "pytest" in args

