"""Unit tests for harnessfix.dashboard --quality wiring (read-only)."""
import json

from harnessfix import dashboard
from harnessfix.tracing import (
    LAYER_EXECUTION,
    LAYER_VERIFICATION,
    LAYER_OBSERVABILITY,
)


def _write_trace(tmp_path, task_id, outcome, duration=0.0):
    p = tmp_path / f"{task_id}.jsonl"
    events = [
        {"kind": "task_begin", "task_id": task_id, "layer": LAYER_VERIFICATION, "user_input": "x"},
        {"kind": "tool_result", "task_id": task_id, "layer": LAYER_EXECUTION, "duration_s": duration},
        {"kind": "loop_end", "task_id": task_id, "layer": LAYER_OBSERVABILITY, "outcome": outcome},
    ]
    p.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    return p


class TestQualityReport:
    def test_empty_dir_yields_zero_average(self, tmp_path):
        rep = dashboard._quality_report(tmp_path)
        assert rep["inspected"] == 0
        assert rep["windowed_average"] == 0.0
        assert rep["should_evolve"] is False

    def test_completed_runs_raise_average(self, tmp_path):
        _write_trace(tmp_path, "a", "completed", duration=0.0)  # score 1.0
        _write_trace(tmp_path, "b", "completed", duration=0.0)  # score 1.0
        rep = dashboard._quality_report(tmp_path, threshold=0.7)
        assert rep["inspected"] == 2
        assert rep["windowed_average"] == 1.0
        assert rep["should_evolve"] is False

    def test_failed_runs_trigger_evolution(self, tmp_path):
        _write_trace(tmp_path, "a", "error")   # score 0.0
        _write_trace(tmp_path, "b", "error")   # score 0.0
        rep = dashboard._quality_report(tmp_path, threshold=0.7)
        assert rep["windowed_average"] == 0.0
        assert rep["should_evolve"] is True

    def test_incomplete_scores_zero(self, tmp_path):
        _write_trace(tmp_path, "a", "incomplete")  # no loop_end -> 0.0
        rep = dashboard._quality_report(tmp_path)
        assert rep["runs"][0]["score"] == 0.0
        assert rep["runs"][0]["outcome"] == "incomplete"

    def test_limit_respected(self, tmp_path):
        for i in range(5):
            _write_trace(tmp_path, f"t{i}", "completed", duration=0.0)
        rep = dashboard._quality_report(tmp_path, limit=3)
        assert rep["inspected"] == 3


class TestMainQualityFlag:
    def test_quality_table_mode(self, tmp_path, capsys):
        _write_trace(tmp_path, "a", "completed", duration=0.0)
        rc = dashboard.main([
            "--traces", str(tmp_path), "--quality", "--limit", "1",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "windowed average" in out
        assert "should evolve" in out

    def test_quality_json_mode(self, tmp_path, capsys):
        _write_trace(tmp_path, "a", "completed", duration=0.0)
        rc = dashboard.main([
            "--traces", str(tmp_path), "--quality", "--json", "--limit", "1",
        ])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["windowed_average"] == 1.0
        assert data["should_evolve"] is False
