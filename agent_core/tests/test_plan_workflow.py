"""Unit tests for the plan workflow modules.

Covers:
* Schema validation (plan_schema)
* Lifecycle atomicity and JSONL logging (plan_lifecycle)
* Dry-run blocking of mutating tools (plan_dry_run)
* File protection for plan files (file_protection)
* Decision gating against .decisions.json (plan_decision_gate)
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from agent_core.commands.plan_schema import (
    PlanStatus,
    PlanTransition,
    PlanMetadata,
    DryRunResult,
    DecisionGateResult,
    PlanLogEntry,
    validate_plan_markdown,
)
from agent_core.commands.plan_lifecycle import PlanLifecycleManager, append_log
from agent_core.commands.plan_dry_run import PlanDryRunner, MUTATING_TOOLS
from agent_core.commands.plan_decision_gate import PlanDecisionGate
from agent_core.commands.plan_lifecycle import PlanLifecycleManager
from agent_core.file_protection import is_protected
from agent_core.plan_execution.runner import run_plan, build_and_validate_graph

import asyncio
from agent import Agent


# ── plan_schema ──────────────────────────────────────────────────────────

class TestPlanSchema:
    """Tests for plan schema enums, dataclasses, and markdown validation."""

    def test_plan_status_values(self):
        assert PlanStatus.PROPOSED.value == "proposed"
        assert PlanStatus.EXECUTING.value == "executing"
        assert PlanStatus.EXECUTED.value == "executed"
        assert PlanStatus.FAILED.value == "failed"

    def test_plan_transition_values(self):
        assert PlanTransition.SUBMIT.value == "submit"
        assert PlanTransition.START.value == "start"
        assert PlanTransition.FINISH.value == "finish"
        assert PlanTransition.ERROR.value == "error"

    def test_plan_metadata_defaults(self):
        meta = PlanMetadata(id="test-001", title="Test Plan")
        assert meta.status == PlanStatus.PROPOSED
        assert meta.author == "agent"
        assert meta.tags == []

    def test_dry_run_result_defaults(self):
        result = DryRunResult(valid=True)
        assert result.errors == []
        assert result.warnings == []
        assert result.affected_files == []

    def test_decision_gate_result_defaults(self):
        result = DecisionGateResult(passed=True)
        assert result.violations == []
        assert result.metadata is None

    def test_plan_log_entry_to_dict(self):
        entry = PlanLogEntry(
            plan_id="test.md",
            timestamp="2026-01-01T00:00:00Z",
            transition=PlanTransition.SUBMIT,
            status=PlanStatus.PROPOSED,
        )
        d = entry.to_dict()
        assert d["plan_id"] == "test.md"
        assert d["transition"] == "submit"
        assert d["status"] == "proposed"

    def test_validate_plan_markdown_valid(self):
        content = "# Proposed Plan\n\n## Tasks\n\n- [T1] Do something\n- [T2] Do another thing\n"
        ok, errors = validate_plan_markdown(content)
        assert ok is True
        assert errors == []

    def test_validate_plan_markdown_missing_heading(self):
        content = "## Tasks\n\n- [T1] Do something\n"
        ok, errors = validate_plan_markdown(content)
        assert ok is False
        assert any("heading" in e.lower() for e in errors)

    def test_validate_plan_markdown_no_tasks(self):
        content = "# Proposed Plan\n\n## Tasks\n\nNo tasks here.\n"
        ok, errors = validate_plan_markdown(content)
        assert ok is False
        assert any("no tasks" in e.lower() for e in errors)


# ── plan_lifecycle ───────────────────────────────────────────────────────

class TestPlanLifecycle:
    """Tests for lifecycle transitions and JSONL audit logging."""

    def test_start_plan_renames_file(self, tmp_path: Path):
        plan_dir = tmp_path / "docs"
        plan_dir.mkdir()
        (plan_dir / "plan_proposed.md").write_text("# Proposed Plan\n")

        lm = PlanLifecycleManager(plan_dir, tmp_path)
        result = lm.start_plan()

        assert result.name == "plan_executing.md"
        assert result.exists()
        assert not (plan_dir / "plan_proposed.md").exists()

    def test_finish_plan_renames_to_timestamped(self, tmp_path: Path):
        plan_dir = tmp_path / "docs"
        plan_dir.mkdir()
        (plan_dir / "plan_executing.md").write_text("# Plan\n")

        lm = PlanLifecycleManager(plan_dir, tmp_path)
        result = lm.finish_plan()

        assert result.name.startswith("plan_executed_")
        assert result.exists()
        assert not (plan_dir / "plan_executing.md").exists()

    def test_lifecycle_writes_jsonl_log(self, tmp_path: Path):
        plan_dir = tmp_path / "docs"
        plan_dir.mkdir()
        (plan_dir / "plan_proposed.md").write_text("# Plan\n")

        lm = PlanLifecycleManager(plan_dir, tmp_path)
        lm.start_plan()
        lm.finish_plan()

        log_file = plan_dir / ".plans.jsonl"
        assert log_file.exists()
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 2

        entry1 = json.loads(lines[0])
        assert entry1["transition"] == "start"
        assert entry1["status"] == "executing"

        entry2 = json.loads(lines[1])
        assert entry2["transition"] == "finish"
        assert entry2["status"] == "executed"

    def test_start_plan_raises_if_no_proposed(self, tmp_path: Path):
        plan_dir = tmp_path / "docs"
        plan_dir.mkdir()
        lm = PlanLifecycleManager(plan_dir, tmp_path)
        with pytest.raises(FileNotFoundError):
            lm.start_plan()

    def test_fail_plan_renames_to_failed(self, tmp_path: Path):
        plan_dir = tmp_path / "docs"
        plan_dir.mkdir()
        (plan_dir / "plan_executing.md").write_text("# Plan\n")

        lm = PlanLifecycleManager(plan_dir, tmp_path)
        result = lm.fail_plan()

        assert result.name.startswith("plan_failed_")
        assert result.exists()
        assert not (plan_dir / "plan_executing.md").exists()

    def test_fail_plan_logs_error_transition(self, tmp_path: Path):
        plan_dir = tmp_path / "docs"
        plan_dir.mkdir()
        (plan_dir / "plan_executing.md").write_text("# Plan\n")

        lm = PlanLifecycleManager(plan_dir, tmp_path)
        lm.fail_plan()

        log_file = plan_dir / ".plans.jsonl"
        assert log_file.exists()
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["transition"] == "error"
        assert entry["status"] == "failed"

    def test_fail_plan_raises_if_no_file(self, tmp_path: Path):
        plan_dir = tmp_path / "docs"
        plan_dir.mkdir()
        lm = PlanLifecycleManager(plan_dir, tmp_path)
        with pytest.raises(FileNotFoundError):
            lm.fail_plan()

    def test_append_log_creates_file(self, tmp_path: Path):
        log_file = tmp_path / "test.jsonl"
        entry = PlanLogEntry(
            plan_id="test.md",
            timestamp="2026-01-01T00:00:00Z",
            transition=PlanTransition.SUBMIT,
            status=PlanStatus.PROPOSED,
        )
        append_log(log_file, entry)
        assert log_file.exists()
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["plan_id"] == "test.md"


# ── plan_dry_run ─────────────────────────────────────────────────────────

class TestPlanDryRun:
    """Tests for dry-run safety gate."""

    def test_safe_plan_passes(self):
        runner = PlanDryRunner()
        content = "# Proposed Plan\n\n## Tasks\n\n- [T1] Add a new function\n"
        result = runner.validate(content)
        assert result.valid is True
        assert result.errors == []

    def test_bypass_attempt_blocked(self):
        runner = PlanDryRunner()
        content = "# Plan\n\n--force skip dry-run safety\n"
        result = runner.validate(content)
        assert result.valid is False
        assert any("bypass" in e.lower() for e in result.errors)

    def test_mutation_warnings_generated(self):
        runner = PlanDryRunner()
        content = "# Plan\n\nRun command to install packages\nEdit the file\nWrite new content\n"
        result = runner.validate(content)
        assert len(result.warnings) > 0

    def test_affected_files_extracted(self):
        runner = PlanDryRunner()
        content = "# Plan\n\nModify `agent_core/foo.py` and `agent_core/bar.py`\n"
        result = runner.validate(content)
        assert "agent_core/foo.py" in result.affected_files
        assert "agent_core/bar.py" in result.affected_files

    def test_mutating_tools_constant(self):
        expected = {"write", "edit", "fix", "delete_file", "run", "git", "delegate", "delegate_batch"}
        assert MUTATING_TOOLS == expected


# ── file_protection ──────────────────────────────────────────────────────

class TestPlanFileProtection:
    """Tests for plan file protection rules in file_protection module."""

    def test_plan_proposed_protected(self, tmp_path: Path):
        assert is_protected("plan_proposed.md", tmp_path) is True

    def test_plans_json_protected(self, tmp_path: Path):
        assert is_protected(".plans.json", tmp_path) is True

    def test_plan_executed_protected(self, tmp_path: Path):
        assert is_protected("plan_executed_20260101_120000.md", tmp_path) is True

    def test_plan_executing_protected(self, tmp_path: Path):
        assert is_protected("plan_executing.md", tmp_path) is True

    def test_env_protected(self, tmp_path: Path):
        assert is_protected(".env", tmp_path) is True

    def test_reports_protected(self, tmp_path: Path):
        assert is_protected("reports/summary.md", tmp_path) is True

    def test_normal_file_not_protected(self, tmp_path: Path):
        assert is_protected("agent_core/foo.py", tmp_path) is False


# ── plan_decision_gate ───────────────────────────────────────────────────

class TestPlanDecisionGate:
    """Tests for decision gate validation against .decisions.json."""

    def test_clean_plan_passes(self, tmp_path: Path):
        decisions = [
            {
                "id": "001",
                "title": "Use Pydantic for validation",
                "tags": ["input-validation"],
                "affected_files": [],
                "contradictions": [],
                "resolved_by": None,
            }
        ]
        (tmp_path / ".decisions.json").write_text(json.dumps(decisions))
        gate = PlanDecisionGate(tmp_path)
        content = "# Proposed Plan\n\n## Tasks\n\n- [T1] Add feature\n"
        result = gate.validate(content)
        assert result.passed is True
        assert result.violations == []

    def test_violation_detected(self, tmp_path: Path):
        decisions = [
            {
                "id": "002",
                "title": "Self-modification guard",
                "tags": ["self-modification-prevention", "integrity-protection"],
                "affected_files": ["agent_core/security/guard.py"],
                "contradictions": [],
                "resolved_by": None,
            }
        ]
        (tmp_path / ".decisions.json").write_text(json.dumps(decisions))
        gate = PlanDecisionGate(tmp_path)
        content = "# Plan\n\nModify `agent_core/security/guard.py` to update security logic\n"
        result = gate.validate(content)
        assert result.passed is False
        assert len(result.violations) > 0

    def test_no_decisions_file_passes(self, tmp_path: Path):
        gate = PlanDecisionGate(tmp_path)
        content = "# Plan\n\n## Tasks\n\n- [T1] Do something\n"
        result = gate.validate(content)
        assert result.passed is True

    def test_resolved_decision_flagged(self, tmp_path: Path):
        decisions = [
            {
                "id": "010",
                "title": "Old decision",
                "tags": [],
                "affected_files": [],
                "contradictions": [],
                "resolved_by": "new_decision",
            }
        ]
        (tmp_path / ".decisions.json").write_text(json.dumps(decisions))
        gate = PlanDecisionGate(tmp_path)
        content = "# Plan\n\nReimplement new_decision approach\n"
        result = gate.validate(content)
        assert result.passed is False
        assert any("resolved" in v.lower() for v in result.violations)


# ── NLP plan tools registration ─────────────────────────────────────────

class TestPlanNlpToolsRegistration:
    """Verify plan tools appear in schemas and handler dict."""

    def test_plan_tools_in_nlp_tool_schemas(self):
        from agent_core.tool_schemas import NLP_TOOL_SCHEMAS
        names = {t["function"]["name"] for t in NLP_TOOL_SCHEMAS}
        for tool in ("plan_status", "plan_start", "plan_step", "plan_finish"):
            assert tool in names, f"{tool} missing from NLP_TOOL_SCHEMAS"

    def test_plan_tools_in_nlp_tool_names(self):
        from agent_core.tool_schemas import NLP_TOOL_NAMES
        for tool in ("plan_status", "plan_start", "plan_step", "plan_finish"):
            assert tool in NLP_TOOL_NAMES, f"{tool} missing from NLP_TOOL_NAMES"

    def test_plan_handlers_registered(self):
        from agent import Agent
        agent = Agent.__new__(Agent)
        agent.workspace = tempfile.mkdtemp()
        handlers = agent._nlp_tool_handlers()
        for tool in ("plan_status", "plan_start", "plan_step", "plan_finish"):
            assert tool in handlers, f"{tool} handler not registered"


# ── Regression tests for bug fixes ─────────────────────────────────────

class TestLifecycleRegression:
    """Regression: start_plan must log the *destination* path, not source."""

    def test_start_plan_logs_executing_path(self, tmp_path: Path):
        """Bug fix: _log_transition used `src` (proposed) instead of `dst`
        (executing) after the file had already been moved."""
        plan_dir = tmp_path / "docs"
        plan_dir.mkdir()
        (plan_dir / "plan_proposed.md").write_text("# Plan\n")

        lm = PlanLifecycleManager(plan_dir, tmp_path)
        lm.start_plan()

        log_file = plan_dir / ".plans.jsonl"
        assert log_file.exists()
        lines = log_file.read_text().strip().splitlines()
        entry = json.loads(lines[0])
        # The logged plan_id must be the executing file, not the proposed one
        assert "plan_executing" in entry["plan_id"]
        assert "plan_proposed" not in entry["plan_id"]


class TestPlanStepRegression:
    """Regression: plan_step must pass dependency results and detect failures.

    ``plan_step`` (``agent._nlp_plan_step``) is filesystem-driven: it reads the
    plan from ``.docs/<stamp>/plan_executing.md``, completed task ids from
    ``plan_execution_report.md``, and dependency results from
    ``dep_<id>_result.json``.  The plan must use the ``## Tasks`` block format
    that ``parse_plan_tasks`` expects (see ``plan exec`` in plan_cmd.py).
    """

    def _make_run(self, tmp_path: Path) -> Path:
        """Create ``.docs/<stamp>/plan_executing.md`` with T1 and T2 (T2 deps T1)."""
        from agent_core.commands.doc_paths import run_stamp

        run_dir = tmp_path / ".docs" / run_stamp()
        run_dir.mkdir(parents=True)
        (run_dir / "plan_executing.md").write_text(
            "# Plan\n\n"
            "## Tasks\n"
            "- [T1] task one (role: implementer)\n"
            "- [T2] task two (role: implementer), deps: T1\n",
            encoding="utf-8",
        )
        return run_dir

    @staticmethod
    def _agent(tmp_path: Path):
        agent = Agent.__new__(Agent)
        agent.mode = "build"
        agent.workspace = str(tmp_path)
        return agent

    def test_step_rejects_unmet_dependency(self, tmp_path: Path):
        """T2 depends on T1 — executing T2 before T1 completed must fail."""
        self._make_run(tmp_path)
        agent = self._agent(tmp_path)

        result = asyncio.run(agent._nlp_plan_step({"task_id": "T2"}))
        assert "unmet" in result.lower()

    def test_step_reports_executor_failure(self, tmp_path: Path):
        """An empty subagent result must mark the task failed, not completed."""
        run_dir = self._make_run(tmp_path)
        agent = self._agent(tmp_path)

        class EmptySub:
            async def respond(self, desc):
                return ""  # empty => executor failure

        agent.spawn_subagent = lambda **kw: EmptySub()

        result = asyncio.run(agent._nlp_plan_step({"task_id": "T1"}))
        assert "failed" in result.lower()
        report = (run_dir / "plan_execution_report.md").read_text(encoding="utf-8")
        assert "[T1] failed" in report

    def test_step_passes_dep_results(self, tmp_path: Path):
        """Dependency results from completed tasks must be forwarded to the subagent."""
        run_dir = self._make_run(tmp_path)
        # T1 is completed and produced a result.
        (run_dir / "plan_execution_report.md").write_text(
            "- [T1] completed (role: implementer)\n", encoding="utf-8"
        )
        (run_dir / "dep_T1_result.json").write_text(
            json.dumps({"summary": "T1 done"}), encoding="utf-8"
        )
        agent = self._agent(tmp_path)

        captured: dict[str, str] = {}

        class FakeSub:
            async def respond(self, desc):
                captured["desc"] = desc
                return "ok"

        agent.spawn_subagent = lambda **kw: FakeSub()

        result = asyncio.run(agent._nlp_plan_step({"task_id": "T2"}))
        assert "completed" in result.lower()
        # The description passed to the subagent must contain the upstream result.
        assert "T1" in captured.get("desc", "")
        assert "T1 done" in captured.get("desc", "")


class TestRunPlanPersistence:
    """Regression for issue #108: run_plan must persist task completion state and
    dependency results to disk so a subsequent session can resume via plan_step.

    Previously, ``plan_start`` ran all tasks through ``run_plan()`` but never wrote
    ``dep_<id>_result.json`` or ``plan_execution_report.md``, leaving no filesystem
    trace for cross-session resumption (``_nlp_plan_step`` reads both).  This test
    verifies the persistence contract is now honoured.
    """

    @staticmethod
    def _tasks():
        from agent_core.plan_execution.parser import parse_plan_tasks, PlanTask
        return [
            PlanTask(id="T1", description="task one", role="implementer"),
            PlanTask(
                id="T2", description="task two", role="implementer", depends_on=["T1"],
            ),
        ]

    def test_run_plan_writes_dep_results_and_report(self, tmp_path: Path):
        """After batch execution with plan_dir set, dep results + report must exist."""
        run_dir = tmp_path / ".docs" / "run_001"
        run_dir.mkdir(parents=True)

        captured_desc: dict[str, str] = {}

        class FakeSub:
            async def respond(self, desc):
                captured_desc["desc"] = desc
                return f"done for {desc[:20]}"  # non-empty => completed

            def get_context_summary(self, max_messages=3):
                return "summary"

        agent = Agent.__new__(Agent)
        agent.mode = "build"
        agent.workspace = str(tmp_path)
        agent.spawn_subagent = lambda **kw: FakeSub()

        tasks = self._tasks()
        asyncio.run(run_plan(agent, tasks, plan_dir=str(run_dir)))

        # T1 completed -> dep_T1_result.json must exist with the executor result.
        dep_t1 = run_dir / "dep_T1_result.json"
        assert dep_t1.exists(), f"Expected {dep_t1} to be written by run_plan"
        t1_data = json.loads(dep_t1.read_text(encoding="utf-8"))
        assert t1_data.get("success") is True

        # T2 completed -> dep_T2_result.json must exist.
        dep_t2 = run_dir / "dep_T2_result.json"
        assert dep_t2.exists(), f"Expected {dep_t2} to be written by run_plan"

        # Execution report must contain one line per terminal task.
        report = (run_dir / "plan_execution_report.md").read_text(encoding="utf-8")
        assert "[T1] completed" in report
        assert "[T2] completed" in report

    def test_run_plan_passes_dep_results_to_downstream(self, tmp_path: Path):
        """Dependency results written by T1 must flow into T2's input_data."""
        run_dir = tmp_path / ".docs" / "run_002"
        run_dir.mkdir(parents=True)

        captured_desc: dict[str, str] = {}

        class FakeSub:
            async def respond(self, desc):
                captured_desc["desc"] = desc
                return "ok"

            def get_context_summary(self, max_messages=3):
                return "summary"

        agent = Agent.__new__(Agent)
        agent.mode = "build"
        agent.workspace = str(tmp_path)
        agent.spawn_subagent = lambda **kw: FakeSub()

        tasks = self._tasks()
        snap = asyncio.run(run_plan(agent, tasks, plan_dir=str(run_dir)))

        # T2's input_data must contain the upstream result from T1.
        t2_rec = next(r for r in snap["tasks"] if r["task_id"] == "T2")
        assert "input_data" in t2_rec
        assert "T1" in t2_rec["input_data"], f"Expected T1 in input_data: {t2_rec['input_data']}"

    def test_run_plan_no_persistence_when_dir_none(self, tmp_path: Path):
        """When plan_dir is None (default), no files should be written."""
        class FakeSub:
            async def respond(self, desc):
                return "ok"

            def get_context_summary(self, max_messages=3):
                return "summary"

        agent = Agent.__new__(Agent)
        agent.mode = "build"
        agent.workspace = str(tmp_path)
        agent.spawn_subagent = lambda **kw: FakeSub()

        tasks = self._tasks()
        asyncio.run(run_plan(agent, tasks))  # no plan_dir

        run_dir = tmp_path / ".docs"
        if run_dir.exists():
            for f in run_dir.iterdir():
                assert not f.name.startswith("dep_"), \
                    f"dep result should not be written without plan_dir: {f}"
                assert f.name != "plan_execution_report.md", \
                    "execution report should not be written without plan_dir"
