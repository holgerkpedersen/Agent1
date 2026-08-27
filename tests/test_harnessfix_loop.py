"""Phase 4 tests: the HarnessFix closed loop and its gates.

Covers the spec's loop test: "the loop rejects a repair that fails tests",
plus the human-review gate (fail-closed headless) and the acceptance path.
"""
from __future__ import annotations

import json
from pathlib import Path

from harnessfix import gates
from harnessfix.gates import should_accept
from harnessfix.loop import run_loop
from harnessfix.repairs.tool_interface import TOOL_INTERFACE_REPAIR_ID, revert
from harnessfix.tracing import KIND_LOOP_END, KIND_TOOL_ERROR, TraceWriter

_OLD = 'result_str = f"Tool error: {exc}"'
_NEW = 'result_str = f"Tool error ({type(exc).__name__}): {exc}"'


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
        {"kind": KIND_LOOP_END, "layer": "lifecycle", "outcome": "completed", "termination_reason": "answer"}
    )
    writer.close()


def _loop_source() -> str:
    return Path("agent_core/llm/tool_loop.py").read_text(encoding="utf-8")


def test_should_accept_requires_tests_and_security():
    assert should_accept(False, True, None, None) is False
    assert should_accept(True, False, None, None) is False
    assert should_accept(True, True, None, None) is True


def test_should_accept_rejects_benchmark_regression():
    assert should_accept(True, True, 60.0, 55.0) is False
    assert should_accept(True, True, 60.0, 62.0) is True
    assert should_accept(True, True, 60.0, 59.9, regression_tolerance=0.2) is True


def test_benchmark_key_is_profile_aware():
    """Decision #055: benchmark results are keyed by model|profile so the
    same model under different profiles is compared like with like."""
    from harnessfix.gates import _benchmark_key

    assert _benchmark_key("qwen3.8-27b") == "qwen3.8-27b"
    assert _benchmark_key("qwen3.8-27b", "deep-analysis") == "qwen3.8-27b|deep-analysis"
    assert _benchmark_key("m", "deep-analysis") != _benchmark_key("m", "fast-codegen")


def test_benchmark_gate_reads_list_form_report(tmp_path, monkeypatch):
    """Regression: benchmark.py's --output file stores models as a LIST
    (save_json_report); the gate used to call .get(key) on the list and
    crash with AttributeError."""
    import json as _json

    from harnessfix import gates as _gates

    payload = {
        "models": [
            {"model": "m", "profile": "deep-analysis",
             "display_name": "m|deep-analysis", "overall_accuracy": 87.5},
        ]
    }

    def fake_run(cmd, **kw):
        out = Path("reports") / "benchmark_harnessfix.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_json.dumps(payload), encoding="utf-8")
        return type("P", (), {"returncode": 0})()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_gates.subprocess, "run", fake_run)
    assert _gates.run_benchmark_gate("m", "deep-analysis") == 87.5
    assert _gates.run_benchmark_gate("m", "fast-codegen") is None
    assert _gates.run_benchmark_gate(None) is None


def test_loop_rejects_repair_that_fails_tests(tmp_path, monkeypatch):
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    _write_tool_error_trace(traces_dir, "tr1")

    monkeypatch.setattr(gates, "get_baseline_failures", lambda *a, **k: frozenset())
    monkeypatch.setattr(gates, "run_test_gate", lambda *a, **k: (False, "1 failed"))
    monkeypatch.setattr(gates, "run_security_gate", lambda: (True, "ok"))
    monkeypatch.setattr(gates, "run_benchmark_gate", lambda model, profile=None: None)
    # The collision guard must not skip these flows: point it at an empty dir.
    (tmp_path / "no_tests").mkdir()
    monkeypatch.setattr(
        "harnessfix.repairs.collisions.DEFAULT_TESTS_DIR", tmp_path / "no_tests"
    )

    out = tmp_path / "out"
    summary = run_loop(traces_dir, approve=True, model=None, output_dir=out)

    assert summary["proposed_repair"] == TOOL_INTERFACE_REPAIR_ID
    assert summary["tests_passed"] is False
    assert summary["accepted"] is False
    assert summary["verdict"] == "rejected_and_reverted"
    # Repair was reverted: the original error string is back in the source.
    assert _OLD in _loop_source()
    assert _NEW not in _loop_source()
    # summary.json + per-task diagnoses are persisted.
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["verdict"] == "rejected_and_reverted"
    assert list((out / "diagnoses").glob("*.json"))


def test_loop_fail_closed_without_approval(tmp_path):
    traces_dir = tmp_path / "traces2"
    traces_dir.mkdir()
    _write_tool_error_trace(traces_dir, "tr2")

    out = tmp_path / "out2"
    summary = run_loop(traces_dir, approve=False, model=None, output_dir=out)

    assert summary["verdict"] == "review_required_fail_closed"
    assert summary["proposed_repair"] == TOOL_INTERFACE_REPAIR_ID
    assert _OLD in _loop_source()
    assert _NEW not in _loop_source()


def test_loop_accepts_repair_when_all_gates_pass(tmp_path, monkeypatch):
    traces_dir = tmp_path / "traces3"
    traces_dir.mkdir()
    _write_tool_error_trace(traces_dir, "tr3")

    monkeypatch.setattr(gates, "get_baseline_failures", lambda *a, **k: frozenset())
    monkeypatch.setattr(gates, "run_test_gate", lambda *a, **k: (True, "passed"))
    monkeypatch.setattr(gates, "run_security_gate", lambda: (True, "ok"))
    monkeypatch.setattr(gates, "run_benchmark_gate", lambda model, profile=None: None)
    # Collision guard: empty tests dir so the apply/accept path is exercised.
    (tmp_path / "no_tests2").mkdir()
    monkeypatch.setattr(
        "harnessfix.repairs.collisions.DEFAULT_TESTS_DIR", tmp_path / "no_tests2"
    )

    out = tmp_path / "out3"
    try:
        summary = run_loop(traces_dir, approve=True, model=None, output_dir=out)
        assert summary["accepted"] is True
        assert summary["verdict"] == "accepted"
        assert _NEW in _loop_source()
    finally:
        # Keep the working tree clean regardless of the outcome.
        revert()
    assert _OLD in _loop_source()


def test_auto_approve_accepts_when_all_gates_pass(tmp_path, monkeypatch):
    """Fully autonomous mode applies + accepts a repair when the gates are green."""
    traces_dir = tmp_path / "traces_auto_ok"
    traces_dir.mkdir()
    _write_tool_error_trace(traces_dir, "tra_ok")

    monkeypatch.setattr(gates, "get_baseline_failures", lambda *a, **k: frozenset())
    monkeypatch.setattr(gates, "run_test_gate", lambda *a, **k: (True, "passed"))
    monkeypatch.setattr(gates, "run_security_gate", lambda: (True, "ok"))
    monkeypatch.setattr(gates, "run_benchmark_gate", lambda model, profile=None: None)
    (tmp_path / "no_tests_auto").mkdir()
    monkeypatch.setattr(
        "harnessfix.repairs.collisions.DEFAULT_TESTS_DIR", tmp_path / "no_tests_auto"
    )

    out = tmp_path / "out_auto_ok"
    try:
        summary = run_loop(traces_dir, approve=False, auto_approve=True,
                           model=None, output_dir=out)
        assert summary["accepted"] is True
        assert summary["verdict"] == "accepted"
        assert _NEW in _loop_source()
    finally:
        revert()
    assert _OLD in _loop_source()


def test_auto_approve_reverts_on_failing_test(tmp_path, monkeypatch):
    """Autonomous mode must NOT keep a repair that fails the test gate — it
    reverts, exactly like the manual --approve path."""
    traces_dir = tmp_path / "traces_auto_fail"
    traces_dir.mkdir()
    _write_tool_error_trace(traces_dir, "tra_fail")

    monkeypatch.setattr(gates, "get_baseline_failures", lambda *a, **k: frozenset())
    monkeypatch.setattr(gates, "run_test_gate", lambda *a, **k: (False, "1 failed"))
    monkeypatch.setattr(gates, "run_security_gate", lambda: (True, "ok"))
    monkeypatch.setattr(gates, "run_benchmark_gate", lambda model, profile=None: None)
    (tmp_path / "no_tests_af").mkdir()
    monkeypatch.setattr(
        "harnessfix.repairs.collisions.DEFAULT_TESTS_DIR", tmp_path / "no_tests_af"
    )

    out = tmp_path / "out_auto_fail"
    summary = run_loop(traces_dir, approve=False, auto_approve=True,
                       model=None, output_dir=out)
    assert summary["accepted"] is False
    assert summary["verdict"] == "rejected_and_reverted"
    # The tree is byte-identical to before: the repair never stuck.
    assert _OLD in _loop_source()
    assert _NEW not in _loop_source()


class TestMainExitCode:
    """main() must surface the loop outcome via its exit code, not always 0
    (regression: a hardcoded `return 0` masked rejected/reverted repairs as
    success to any caller of `raise SystemExit(main())`).

    main() returns the int directly; the module-level
    ``raise SystemExit(main())`` is what turns it into a process exit code.
    """

    def test_accepted_exits_zero(self, tmp_path, monkeypatch):
        from harnessfix.loop import main

        traces_dir = tmp_path / "traces_ok"
        traces_dir.mkdir()
        _write_tool_error_trace(traces_dir, "exit_ok")

        monkeypatch.setattr(gates, "get_baseline_failures", lambda *a, **k: frozenset())
        monkeypatch.setattr(gates, "run_test_gate", lambda *a, **k: (True, "passed"))
        monkeypatch.setattr(gates, "run_security_gate", lambda: (True, "ok"))
        monkeypatch.setattr(gates, "run_benchmark_gate", lambda model, profile=None: None)
        (tmp_path / "no_tests_ec").mkdir()
        monkeypatch.setattr(
            "harnessfix.repairs.collisions.DEFAULT_TESTS_DIR", tmp_path / "no_tests_ec"
        )
        out = tmp_path / "out_ec"
        try:
            code = main(["--traces", str(traces_dir), "--approve", "--output", str(out)])
            assert code == 0
        finally:
            revert()

    def test_rejected_and_reverted_exits_nonzero(self, tmp_path, monkeypatch):
        from harnessfix.loop import main

        traces_dir = tmp_path / "traces_rej"
        traces_dir.mkdir()
        _write_tool_error_trace(traces_dir, "exit_rej")

        monkeypatch.setattr(gates, "get_baseline_failures", lambda *a, **k: frozenset())
        monkeypatch.setattr(gates, "run_test_gate", lambda *a, **k: (False, "failed"))
        monkeypatch.setattr(gates, "run_security_gate", lambda: (True, "ok"))
        monkeypatch.setattr(gates, "run_benchmark_gate", lambda model, profile=None: None)
        (tmp_path / "no_tests_rej").mkdir()
        monkeypatch.setattr(
            "harnessfix.repairs.collisions.DEFAULT_TESTS_DIR", tmp_path / "no_tests_rej"
        )
        out = tmp_path / "out_rej"
        try:
            code = main(["--traces", str(traces_dir), "--approve", "--output", str(out)])
            assert code == 1
        finally:
            revert()

    def test_fail_closed_exits_zero(self, tmp_path):
        from harnessfix.loop import main

        traces_dir = tmp_path / "traces_fc"
        traces_dir.mkdir()
        _write_tool_error_trace(traces_dir, "exit_fc")

        out = tmp_path / "out_fc"
        code = main(["--traces", str(traces_dir), "--output", str(out)])
        assert code == 0


def _write_abandonment_trace(traces_dir: Path, task_id: str) -> None:
    """A lifecycle trace (mutation + non-completion, no loop_end) that maps to
    the lifecycle layer, which has TWO catalog repairs (stuck-repeat before
    abandonment-resume).  Mirrors the proven writer in
    test_repairs_abandonment_resume.py so it diagnoses as lifecycle."""
    from harnessfix.tracing import (
        KIND_TOOL_CALL,
        KIND_TOOL_RESULT,
        LAYER_TOOL_INTERFACE,
    )

    writer = TraceWriter(task_id=task_id, directory=traces_dir)
    writer.emit({"kind": KIND_TOOL_CALL, "layer": LAYER_TOOL_INTERFACE,
                 "tool": "write", "args_hash": "a"})
    writer.emit({"kind": KIND_TOOL_RESULT, "layer": LAYER_TOOL_INTERFACE,
                 "tool": "write", "affected_files": ["a.py"]})
    # A third event so the corpus counts it as failed (>= MIN_ACTIVITY_EVENTS);
    # no loop_end -> interrupted after mutation (decision #052).
    writer.emit({"kind": KIND_TOOL_CALL, "layer": LAYER_TOOL_INTERFACE,
                 "tool": "read", "args_hash": "r"})
    writer.close()


def test_loop_falls_through_to_next_repair_on_rejection(tmp_path, monkeypatch):
    """When the highest-priority repair is rejected by the gates, the loop must
    try the NEXT catalog repair for the same layer instead of giving up — so a
    single bad repair does not stall the whole autonomous run."""
    from harnessfix.repairs.abandonment_resume import ABANDONMENT_RESUME_REPAIR_ID
    from harnessfix.repairs.stuck_repeat import STUCK_REPEAT_REPAIR_ID, revert as revert_stuck

    traces_dir = tmp_path / "traces_fb"
    traces_dir.mkdir()
    _write_abandonment_trace(traces_dir, "fb1")

    # The collision guard must not skip these flows: point it at an empty dir.
    (tmp_path / "no_tests_fb").mkdir()
    monkeypatch.setattr(
        "harnessfix.repaused.collisions.DEFAULT_TESTS_DIR"
        if False else "harnessfix.repairs.collisions.DEFAULT_TESTS_DIR",
        tmp_path / "no_tests_fb",
    )
    monkeypatch.setattr(gates, "get_baseline_failures", lambda *a, **k: frozenset())

    # First candidate (stuck-repeat) is rejected; second (abandonment-resume)
    # is accepted.  The loop should land on the accepted one.
    states = {"call": 0}

    def _test_gate(*a, **k):
        states["call"] += 1
        # Reject the first call (stuck-repeat), accept the second.
        return (False, "1 failed") if states["call"] == 1 else (True, "passed")

    monkeypatch.setattr(gates, "run_test_gate", _test_gate)
    monkeypatch.setattr(gates, "run_security_gate", lambda: (True, "ok"))
    monkeypatch.setattr(gates, "run_benchmark_gate", lambda model, profile=None: None)

    out = tmp_path / "out_fb"
    try:
        summary = run_loop(traces_dir, approve=True, model=None, output_dir=out)
        assert summary["verdict"] == "accepted"
        # The first candidate was rejected and the loop fell through to the
        # second (abandonment-resume), which was accepted.
        assert summary["proposed_repair"] == STUCK_REPEAT_REPAIR_ID
        assert summary["accepted_repair"] == ABANDONMENT_RESUME_REPAIR_ID
        assert summary.get("attempted_repairs") == [STUCK_REPEAT_REPAIR_ID]
    finally:
        revert_stuck()


def test_loop_accepts_repair_that_adds_no_new_failures(tmp_path, monkeypatch):
    """The repo carries pre-existing test failures (corpus-drift /
    environment-specific).  The regression-aware gate must ACCEPT a repair
    that introduces no NEW failures, so the autonomous loop is not permanently
    blocked by a non-100%-green suite.  This is the exact scenario that made
    the loop reject everything and stop at iteration 1."""
    from harnessfix.repairs.stuck_repeat import STUCK_REPEAT_REPAIR_ID, revert as revert_stuck

    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    _write_abandonment_trace(traces_dir, "fb2")

    (tmp_path / "no_tests_pf").mkdir()
    monkeypatch.setattr(
        "harnessfix.repairs.collisions.DEFAULT_TESTS_DIR", tmp_path / "no_tests_pf"
    )
    # Baseline: the suite already has 4 failing tests (pre-existing).
    baseline = frozenset(
        {
            "tests/test_autoreview.py::test_real_corpus_matches_pre050_labels",
            "tests/test_flow_control.py::TestShowPatchVerdict::test_verdict_ignores_pre_existing_errors",
            "tests/test_workflow_cmd.py::TestAnalysisFlagGate::test_flagged_without_force_confirms",
            "tests/test_workflow_cmd.py::TestExtractDecisionsGate::test_warned_candidate_recorded_with_explicit_yes",
        }
    )
    monkeypatch.setattr(gates, "get_baseline_failures", lambda *a, **k: baseline)

    # Post-repair run: same 4 failures, nothing new -> regression gate passes.
    def _test_gate(*a, **k):
        # collect_test_failures() would return these; run_test_gate compares
        # against the baseline and accepts (no new failures).
        return True, "4 failed, 1843 passed"

    monkeypatch.setattr(gates, "run_test_gate", _test_gate)
    monkeypatch.setattr(gates, "run_security_gate", lambda: (True, "ok"))
    monkeypatch.setattr(gates, "run_benchmark_gate", lambda model, profile=None: None)

    out = tmp_path / "out_pf"
    try:
        summary = run_loop(traces_dir, approve=True, model=None, output_dir=out)
        assert summary["verdict"] == "accepted"
        assert summary["accepted_repair"] == STUCK_REPEAT_REPAIR_ID
        assert summary["tests_passed"] is True
    finally:
        revert_stuck()


def test_loop_consults_offline_harness_gate_when_no_model(tmp_path, monkeypatch):
    """With no --model, the loop must gate on the offline harness-quality
    signal (target alignment) instead of the LLM benchmark, and record both
    the baseline/post snapshots and the harness verdict in the summary."""
    from harnessfix.repairs.tool_interface import revert

    traces_dir = tmp_path / "traces_hg"
    traces_dir.mkdir()
    _write_tool_error_trace(traces_dir, "hg1")

    monkeypatch.setattr(gates, "get_baseline_failures", lambda *a, **k: frozenset())
    monkeypatch.setattr(gates, "run_test_gate", lambda *a, **k: (True, "passed"))
    monkeypatch.setattr(gates, "run_security_gate", lambda: (True, "ok"))
    monkeypatch.setattr(gates, "run_benchmark_gate", lambda model, profile=None: None)
    (tmp_path / "no_tests_hg").mkdir()
    monkeypatch.setattr(
        "harnessfix.repairs.collisions.DEFAULT_TESTS_DIR", tmp_path / "no_tests_hg"
    )

    out = tmp_path / "out_hg"
    try:
        summary = run_loop(traces_dir, approve=True, model=None, output_dir=out)
        # The corpus evidences tool_interface, so the offline gate passes and
        # the repair is accepted (benchmark is None -> non-blocking).
        assert summary["accepted"] is True
        assert summary["harness_accepted"] is True
        assert summary["harness_baseline"] is not None
        assert summary["harness_post"] is not None
        assert summary["harness_baseline"]["layer_counts"].get("tool_interface", 0) >= 1
    finally:
        revert()


def test_loop_benchmark_is_optional_cross_check(tmp_path, monkeypatch):
    """When a --model IS supplied, the LLM benchmark is the quality signal and
    the offline harness gate is NOT the deciding factor (benchmark wins).  This
    demotes the benchmark from de-facto criterion to an optional cross-check
    while keeping the deterministic offline gate as the default offline path."""
    from harnessfix.repairs.tool_interface import revert

    traces_dir = tmp_path / "traces_bc"
    traces_dir.mkdir()
    _write_tool_error_trace(traces_dir, "bc1")

    monkeypatch.setattr(gates, "get_baseline_failures", lambda *a, **k: frozenset())
    monkeypatch.setattr(gates, "run_test_gate", lambda *a, **k: (True, "passed"))
    monkeypatch.setattr(gates, "run_security_gate", lambda: (True, "ok"))
    # Benchmark supplied but REGRESSING: baseline (first call) 60.0, post
    # (second call) 55.0 -> should_accept is False -> rejected.
    _bm_calls = {"n": 0}

    def _benchmark(model, profile=None):
        if not model:
            return None
        _bm_calls["n"] += 1
        return 60.0 if _bm_calls["n"] == 1 else 55.0

    monkeypatch.setattr(gates, "run_benchmark_gate", _benchmark)
    (tmp_path / "no_tests_bc").mkdir()
    monkeypatch.setattr(
        "harnessfix.repairs.collisions.DEFAULT_TESTS_DIR", tmp_path / "no_tests_bc"
    )

    out = tmp_path / "out_bc"
    try:
        summary = run_loop(traces_dir, approve=True, model="some-model", output_dir=out)
        # Benchmark regressed (baseline None -> post 55.0 is treated as a
        # regression), so the offline harness pass does NOT override it.
        assert summary["accepted"] is False
        assert summary["verdict"] == "rejected_and_reverted"
        # The harness gate still ran and recorded its (passing) verdict.
        assert summary["harness_accepted"] is True
    finally:
        revert()

