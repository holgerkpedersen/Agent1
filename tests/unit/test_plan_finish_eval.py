"""Phase 2 wiring: plan_finish writes the fact-based evaluation artifact.

Covers agent_core/plan_execution/plan_finish_eval.py and its hook in
_nlp_plan_finish: evaluation.json + one executions.jsonl ledger line per
finish, with graceful degradation when traces are missing or corrupt.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_core.plan_execution.plan_finish_eval import (
    collect_task_counts,
    collect_trace_token_facts,
    write_evaluation_for_plan,
)

PLAN_MD = """# Plan

## Tasks

- t1: do the thing (role: implementer)
- t2: test the thing (role: tester, deps: t1)
"""


def _make_plan_dir(tmp_path: Path, *, report: str | None = None) -> Path:
    plan_dir = tmp_path / ".docs" / "2026-11-08_12-00-00"
    plan_dir.mkdir(parents=True)
    (plan_dir / "plan_executing.md").write_text(PLAN_MD, encoding="utf-8")
    if report is not None:
        (plan_dir / "plan_execution_report.md").write_text(report, encoding="utf-8")
    return plan_dir


def _write_trace(workspace: Path, task_id: str, events: list[dict]) -> None:
    trace_dir = workspace / "reports" / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(ev) for ev in events]
    (trace_dir / f"{task_id}.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


class TestCollectTaskCounts:
    def test_counts_from_plan_and_report(self, tmp_path: Path) -> None:
        plan_dir = _make_plan_dir(
            tmp_path,
            report=(
                "- [t1] completed (role: implementer)\n"
                "- [t2] failed (role: tester)\n"
            ),
        )
        planned, completed = collect_task_counts(plan_dir)
        assert planned == 2
        assert completed == 1

    def test_missing_sources_are_none(self, tmp_path: Path) -> None:
        plan_dir = tmp_path / "empty"
        plan_dir.mkdir()
        planned, completed = collect_task_counts(plan_dir)
        assert planned is None
        assert completed is None


class TestCollectTraceTokenFacts:
    def test_token_usage_event_is_a_fact(self, tmp_path: Path) -> None:
        _write_trace(
            tmp_path,
            "run1",
            [{"kind": "token_usage", "prompt_tokens": 100, "completion_tokens": 40,
              "total_tokens": 140}],
        )
        facts = collect_trace_token_facts(tmp_path)
        assert facts is not None
        assert facts["total_tokens"] == 140

    def test_totals_sum_across_runs(self, tmp_path: Path) -> None:
        _write_trace(tmp_path, "run1", [{"kind": "token_usage", "total_tokens": 100}])
        _write_trace(tmp_path, "run2", [{"kind": "token_usage", "total_tokens": 250}])
        facts = collect_trace_token_facts(tmp_path)
        assert facts is not None and facts["total_tokens"] == 350

    def test_no_traces_yields_none(self, tmp_path: Path) -> None:
        assert collect_trace_token_facts(tmp_path) is None

    def test_corrupt_trace_is_skipped_not_guessed(self, tmp_path: Path) -> None:
        trace_dir = tmp_path / "reports" / "traces"
        trace_dir.mkdir(parents=True)
        (trace_dir / "bad.jsonl").write_text("not json\n{{{\n", encoding="utf-8")
        assert collect_trace_token_facts(tmp_path) is None


class TestWriteEvaluationForPlan:
    def test_writes_artifact_and_one_ledger_line(self, tmp_path: Path) -> None:
        plan_dir = _make_plan_dir(
            tmp_path, report="- [t1] completed (role: implementer)\n"
        )
        _write_trace(tmp_path, "run1", [{"kind": "token_usage", "total_tokens": 420}])

        summary = write_evaluation_for_plan(plan_dir, tmp_path, success=True)

        eval_path = plan_dir / "evaluation.json"
        assert eval_path.is_file()
        record = json.loads(eval_path.read_text(encoding="utf-8"))
        assert record["kind"] == "plan_evaluation"
        assert record["schema_version"]
        assert record["tasks_planned"] == 2
        assert record["tasks_completed"] == 1
        assert record["tokens"] is not None
        assert record["tokens"]["total_tokens"] == 420
        assert isinstance(record["score"], (int, float))

        ledger = tmp_path / "reports" / "history" / "executions.jsonl"
        lines = ledger.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["command"] == "plan_eval"
        assert entry["outcome"] == "success"

        assert "Evaluation:" in summary
        assert "420" in summary

    def test_failed_plan_scores_zero(self, tmp_path: Path) -> None:
        plan_dir = _make_plan_dir(
            tmp_path, report="- [t1] completed (role: implementer)\n"
        )
        _write_trace(tmp_path, "run1", [{"kind": "token_usage", "total_tokens": 10}])

        write_evaluation_for_plan(plan_dir, tmp_path, success=False)

        record = json.loads((plan_dir / "evaluation.json").read_text(encoding="utf-8"))
        assert record["success"] is False
        assert record["score"] == 0.0

    def test_missing_trace_degrades_to_null_tokens(self, tmp_path: Path) -> None:
        plan_dir = _make_plan_dir(tmp_path)
        summary = write_evaluation_for_plan(plan_dir, tmp_path, success=True)

        record = json.loads((plan_dir / "evaluation.json").read_text(encoding="utf-8"))
        assert record["tokens"] is None
        assert "Evaluation:" in summary
        assert "tokens=None" in summary

    def test_unwritable_workspace_is_silent(self, tmp_path: Path) -> None:
        plan_dir = _make_plan_dir(tmp_path)
        missing = tmp_path / "no-such-workspace"
        assert write_evaluation_for_plan(plan_dir, missing, success=True) == ""


class TestPlanFinishHook:
    """The real code path: _nlp_plan_finish appends the evaluation summary."""

    @pytest.mark.asyncio
    async def test_plan_finish_writes_evaluation(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from agent_core.handlers import nlp_tools

        plan_dir = _make_plan_dir(
            tmp_path, report="- [t1] completed (role: implementer)\n"
        )
        _write_trace(tmp_path, "run1", [{"kind": "token_usage", "total_tokens": 77}])

        class FakeAgent:
            workspace = str(tmp_path)

            def is_plan_mode(self) -> bool:
                return False

        state = {
            "status": "executing",
            "plan_dir": str(plan_dir),
            "plan_path": str(plan_dir / "plan_executing.md"),
            "plan_text": PLAN_MD,
            "tasks": [],
        }

        async def fake_state(agent):  # noqa: ANN001 - test double
            return state

        monkeypatch.setattr(nlp_tools, "_get_plan_state", fake_state)

        result = await nlp_tools._nlp_plan_finish(FakeAgent(), {"success": True})

        assert "Plan marked as executed." in result
        assert "Evaluation:" in result
        assert (plan_dir / "evaluation.json").is_file()
        ledger = tmp_path / "reports" / "history" / "executions.jsonl"
        assert len(ledger.read_text(encoding="utf-8").strip().splitlines()) == 1
