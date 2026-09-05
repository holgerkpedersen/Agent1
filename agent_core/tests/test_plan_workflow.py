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
from agent_core.file_protection import is_protected


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
