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


def test_loop_rejects_repair_that_fails_tests(tmp_path, monkeypatch):
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    _write_tool_error_trace(traces_dir, "tr1")

    monkeypatch.setattr(gates, "run_test_gate", lambda: (False, "1 failed"))
    monkeypatch.setattr(gates, "run_security_gate", lambda: (True, "ok"))
    monkeypatch.setattr(gates, "run_benchmark_gate", lambda model: None)
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

    monkeypatch.setattr(gates, "run_test_gate", lambda: (True, "passed"))
    monkeypatch.setattr(gates, "run_security_gate", lambda: (True, "ok"))
    monkeypatch.setattr(gates, "run_benchmark_gate", lambda model: None)
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
