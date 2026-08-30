"""Regression tests for test isolation of the autonomous harness.

These pin the two root causes behind a silent, unproductive autonomous run:

1. Repair edits (apply()/revert()) and the live dashboard beacons must be
   redirected to a temp dir for every test, so a test can never mutate the
   real ``agent_core/llm/tool_loop.py`` nor write fake records into
   ``reports/harnessfix/run_status.json`` / ``run_history.jsonl``.  When a
   loop test instead edits the real file, the post-repair ``pytest`` gate can
   revert the very repair the loop just applied, leaving nothing to commit and
   causing the loop to spin (accepting a phantom repair every iteration).

2. The driver must fail closed when an accepted verdict leaves nothing to
   commit: previously it printed "already clean" and kept iterating, silently
   re-applying a repair that could never persist.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

import harnessfix.progress as progress
import harnessfix.repairs.abandonment_resume as abandonment_resume
import harnessfix.repairs.stuck_repeat as stuck_repeat
import harnessfix.repairs.tool_interface as tool_interface
import scripts.autonomous_self_improve as drv
from harnessfix.loop import run_loop
from harnessfix.repairs import CATALOG
from harnessfix.tracing import KIND_LOOP_END, KIND_TOOL_ERROR, TraceWriter

# These are repair/loop self-tests: they exercise apply()/revert() on the real
# tool_loop.py and stub the gates, so running them inside the autonomous gate
# (which applies the repair to the real file first) would revert the very
# change the gate is validating.  Excluded from the gate via the same
# harnessfix_self_test marker used by test_harnessfix_loop.py /
# test_repairs_*.py (see gates.py: the gate runs `-m not harnessfix_self_test`
# so the repaired functional suite is validated while the change is present).
pytestmark = pytest.mark.harnessfix_self_test


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_tool_error_trace(traces_dir: Path, task_id: str) -> None:
    writer = TraceWriter(task_id=task_id, directory=traces_dir)
    writer.emit(
        {
            "kind": KIND_TOOL_ERROR,
            "layer": "tool_interface",
            "exception": "ValidationError",
            "message": "schema validation failed for path",
        }
    )
    writer.emit(
        {"kind": KIND_LOOP_END, "layer": "lifecycle", "outcome": "completed",
         "termination_reason": "answer"}
    )
    writer.close()


def test_repair_apply_revert_never_touches_tracked_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """With the isolation fixture active, exercising every catalog repair's
    apply()/revert() leaves the real ``tool_loop.py`` byte-identical and points
    each module's ``_TARGET`` at a temp copy."""
    real = Path("agent_core/llm/tool_loop.py")
    before = _sha(real)

    # The fixture already redirected _TARGET; assert that happened.
    assert tool_interface._TARGET.resolve() != real.resolve()
    assert stuck_repeat._TARGET.resolve() != real.resolve()
    assert abandonment_resume._TARGET.resolve() != real.resolve()

    for repair in CATALOG.values():
        if repair.is_applied():
            repair.revert()
        repair.apply()  # idempotent against its own copy
        repair.revert()

    assert _sha(real) == before, "a repair edit leaked into the real source tree"


def test_run_loop_does_not_pollute_real_beacons(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A real ``run_loop`` (which applies + reverts the tool_interface repair
    and writes progress) must not touch the live reports/harnessfix files; the
    isolation fixture routes those to temp.  We snapshot the REAL beacon files
    (resolved from the repo root, independent of the redirected progress paths)
    before and after to prove they are unchanged."""
    from harnessfix import gates

    real_root = Path("reports") / "harnessfix"
    real_status = real_root / "run_status.json"
    real_history = real_root / "run_history.jsonl"
    status_before = real_status.read_text(encoding="utf-8") if real_status.exists() else None
    history_before = real_history.read_text(encoding="utf-8") if real_history.exists() else None

    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    _write_tool_error_trace(traces_dir, "sbx")

    # Ensure the tool_interface repair is NOT already applied, so run_loop can
    # exercise the apply+accept path (it is skipped as already-applied once
    # landed in the real tree by the autonomous driver).
    if tool_interface.is_applied():
        tool_interface.revert()

    monkeypatch.setattr(gates, "get_baseline_failures", lambda *a, **k: frozenset())
    monkeypatch.setattr(gates, "run_test_gate", lambda *a, **k: (True, "passed"))
    monkeypatch.setattr(gates, "run_security_gate", lambda: (True, "ok"))
    monkeypatch.setattr(gates, "run_benchmark_gate", lambda model, profile=None: None)
    (tmp_path / "no_tests").mkdir()
    monkeypatch.setattr(
        "harnessfix.repairs.collisions.DEFAULT_TESTS_DIR", tmp_path / "no_tests"
    )

    out = tmp_path / "out"
    summary = run_loop(traces_dir, approve=True, model=None, output_dir=out)
    # The repair applies + passes the (stubbed) gates -> accepted; the point of
    # this test is that the live beacons are untouched, not the verdict.
    assert summary["verdict"] == "accepted"

    status_after = real_status.read_text(encoding="utf-8") if real_status.exists() else None
    history_after = real_history.read_text(encoding="utf-8") if real_history.exists() else None
    assert status_before == status_after, "run_loop polluted the live beacon"
    assert history_before == history_after, "run_loop polluted run history"


def test_accepted_but_uncommitted_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """If a verdict is accepted yet the repair leaves nothing to commit, the
    driver stops (return 1) instead of silently re-applying a phantom repair
    for every remaining iteration."""
    monkeypatch.delenv("AGENT_AUTONOMOUS", raising=False)
    iters: list[int] = []

    def fake_iteration(iteration, **kwargs):
        iters.append(iteration)
        return {
            "verdict": "accepted",
            "proposed_repair": "tool-interface-error-detail",
            "accepted": True,
            "tests_passed": True,
            "security_passed": True,
            "baseline_rate": None,
            "post_rate": None,
        }

    def fake_git(args, check=True):
        # Pretend nothing is ever changed on disk -> _commit_repair returns False.
        return subprocess.CompletedProcess(args, 0, "", "")

    with monkeypatch.context() as m:
        m.setattr(drv, "_stop_requested", lambda: False)
        m.setattr(drv, "_git", fake_git)
        m.setattr(drv, "run_iteration", fake_iteration)
        rc = drv.main(["--auto", "--max-iterations", "5"])

    assert rc == 1
    # Must stop after the first iteration, not loop to max_iterations.
    assert iters == [1]
