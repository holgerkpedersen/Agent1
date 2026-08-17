"""Tests for the harnessfix.dashboard CLI, incl. the --task detail view."""
from __future__ import annotations

import json

from harnessfix.dashboard import (
    _find_trace,
    _load_diagnosis,
    _show_task,
    main,
)
from harnessfix.reader import TraceValidationError

_GOOD_TRACE = "\n".join(
    json.dumps(ev)
    for ev in [
        {"task_id": "t1", "ts": 1.0, "correlation_id": "", "kind": "step_start", "layer": "lifecycle", "iteration": 0, "budget_remaining": 4},
        {"task_id": "t1", "ts": 1.1, "correlation_id": "", "kind": "tool_call", "layer": "tool_interface", "iteration": 0, "tool": "run", "args_hash": "{}"},
        {"task_id": "t1", "ts": 1.2, "correlation_id": "", "kind": "tool_error", "layer": "tool_interface", "iteration": 0, "tool": "run", "exception": "ValueError", "message": "boom"},
        {"task_id": "t1", "ts": 1.3, "correlation_id": "", "kind": "loop_end", "layer": "lifecycle", "iteration": 0, "outcome": "completed", "termination_reason": "answer"},
    ]
) + "\n"

_DIAGNOSIS = {
    "task_id": "t1",
    "root_layer": "tool_interface",
    "mechanism": "unclassified tool error",
    "evidence": ["step:1"],
    "confidence": 0.85,
    "repair_proposal": "inspect tool_interface harness mechanism",
}


def _setup(tmp_path) -> tuple[object, object]:
    traces = tmp_path / "traces"
    diagnoses = tmp_path / "diagnoses"
    traces.mkdir()
    diagnoses.mkdir()
    (traces / "t1.jsonl").write_text(_GOOD_TRACE, encoding="utf-8")
    (diagnoses / "t1.json").write_text(json.dumps(_DIAGNOSIS), encoding="utf-8")
    return traces, diagnoses


def test_find_trace_accepts_bare_id_and_suffix(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "t1.jsonl").write_text(_GOOD_TRACE, encoding="utf-8")
    assert _find_trace(traces, "t1") == traces / "t1.jsonl"
    assert _find_trace(traces, "t1.jsonl") == traces / "t1.jsonl"
    assert _find_trace(traces, "missing") is None


def test_load_diagnosis_includes_evidence_and_repair(tmp_path):
    traces, diagnoses = _setup(tmp_path)
    diag = _load_diagnosis(diagnoses, "t1")
    assert diag is not None
    assert diag["layer"] == "tool_interface"
    assert diag["evidence"] == ["step:1"]
    assert diag["repair_proposal"] == "inspect tool_interface harness mechanism"
    assert _load_diagnosis(diagnoses, "unknown") is None


def test_task_detail_view_prints_explanation(capsys, tmp_path):
    traces, diagnoses = _setup(tmp_path)
    rc = _show_task(traces, diagnoses, "t1", as_json=False)
    out = capsys.readouterr().out
    assert rc == 0
    assert "HarnessFix task detail — t1" in out
    assert "Timeline:" in out
    assert "exception=ValueError" in out
    assert "<-- FAIL" in out
    assert "Tool error at event 2: run raised ValueError: boom" in out
    assert "Diagnosis: tool_interface — unclassified tool error (confidence 0.85)." in out
    assert "Suggested repair: inspect tool_interface harness mechanism." in out


def test_task_detail_json(capsys, tmp_path):
    traces, diagnoses = _setup(tmp_path)
    rc = _show_task(traces, diagnoses, "t1", as_json=True)
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert data["task_id"] == "t1"
    assert data["summary"]["outcome"] == "completed"
    assert data["diagnosis"]["layer"] == "tool_interface"
    assert data["events"][2]["kind"] == "tool_error"


def test_main_task_unknown_returns_1(capsys, tmp_path):
    traces, diagnoses = _setup(tmp_path)
    rc = main(["--traces", str(traces), "--diagnoses", str(diagnoses), "--task", "missing"])
    assert rc == 1
    assert "Unknown task: 'missing'" in capsys.readouterr().out


def test_main_task_missing_diagnosis_degrades_gracefully(capsys, tmp_path):
    traces, diagnoses = _setup(tmp_path)
    (diagnoses / "t1.json").unlink()
    rc = main(["--traces", str(traces), "--diagnoses", str(diagnoses), "--task", "t1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "diagnosis  : none stored" in out
    assert "No stored diagnosis for this task." in out


def test_main_summary_mode_untouched(capsys, tmp_path):
    traces, diagnoses = _setup(tmp_path)
    rc = main(["--traces", str(traces), "--diagnoses", str(diagnoses), "--limit", "5"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "recent traces from" in out
    assert "t1" in out
