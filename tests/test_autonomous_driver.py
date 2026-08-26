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

    def fake_run_iteration(iteration, *, model, profile, trace_dir, output_dir):
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
        rc = drv.main(["--auto", "--max-iterations", "3"])
    assert rc == 0


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
