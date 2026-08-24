"""Regression tests: deterministic regression-gate for generated plan docs.

The ``analyze`` stage already verified its claims claim-by-claim
(analysis_verifier); ``plan`` / ``entities`` / ``taskplan`` output used to be
written straight from LLM text. These tests pin the new gate:

* :mod:`agent_core.commands.plan_verifier` — deterministic checks
  (path existence vs [NEW]/create-intent, python-fence parsability,
  duplicate definitions across taskplan-referenced files / entity blocks),
* the standalone commands (``plan`` / ``entities`` / ``taskplan``) halt on
  flagged output in autonomous mode (safe default) and append a Verification
  Report when ``--force`` writes anyway,
* ``workflow_cmd._plan_doc_gate`` — the shared gate used by all nine inline
  workflow write sites.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agent_core.commands.entities_cmd import EntitiesCommand
from agent_core.commands.plan_cmd import PlanCommand
from agent_core.commands.plan_verifier import (
    apply_report,
    check_doc,
    report_section,
    summarize,
    verify_entities_doc,
    verify_plan_doc,
    verify_taskplan_doc,
)
from agent_core.commands.taskplan_cmd import TaskplanCommand
from agent_core.commands.workflow_cmd import _plan_doc_gate
from agent_core.commands.base import set_autonomous


@pytest.fixture(autouse=True)
def _autonomous_decline():
    """Gates must auto-DECLINE without stdin (safe default)."""
    set_autonomous(True)
    yield
    set_autonomous(None)


@pytest.fixture()
def ws(tmp_path: Path) -> Path:
    """Workspace with one real module the generated docs can reference."""
    (tmp_path / "existing.py").write_text("def parse(): ...\n", encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# plan_verifier: path existence vs creation intent
# ---------------------------------------------------------------------------

class TestPlanPathChecks:
    def test_existing_path_is_clean(self, ws: Path) -> None:
        result = verify_plan_doc("Update `existing.py` for speed.\n", str(ws))
        assert result.clean
        assert result.checked == 1

    def test_missing_path_without_intent_is_flagged(self, ws: Path) -> None:
        result = verify_plan_doc("Touch `missing_thing.py`.\n", str(ws))
        assert not result.clean
        assert "not found" in result.findings[0]

    def test_create_wording_marks_new_file_ok(self, ws: Path) -> None:
        result = verify_plan_doc(
            "1. Create `utils/forgotten_helper.py` for caching.\n", str(ws))
        assert result.clean

    def test_new_tag_marks_new_file_ok(self, ws: Path) -> None:
        result = verify_taskplan_doc(
            "1. [NEW] `agent_core/newmod.py` — add module\n", str(ws))
        assert result.clean

    def test_modify_tag_on_missing_file_is_flagged(self, ws: Path) -> None:
        result = verify_taskplan_doc(
            "1. [MODIFY] `ghost.py` — extend handler\n", str(ws))
        assert not result.clean
        assert "modify target" in result.findings[0]

    def test_stale_report_is_not_rechecked(self, ws: Path) -> None:
        doc = ("Touch `missing_thing.py`.\n\n---\n\n## Verification Report\n\n"
               "- plan claims checked: 1 — 0 verified, 1 flagged.\n"
               "- [UNVERIFIED] `missing_thing.py` — not found\n")
        result = verify_plan_doc(doc, str(ws))
        # Only the real claim is counted — the report's own copy is ignored.
        assert result.checked == 1


# ---------------------------------------------------------------------------
# plan_verifier: duplicate definitions
# ---------------------------------------------------------------------------

class TestDuplicateDefinitions:
    def test_taskplan_flags_dup_defs_in_same_dir(self, ws: Path) -> None:
        (ws / "a_mod.py").write_text("def handle(): ...\n", encoding="utf-8")
        (ws / "b_mod.py").write_text("def handle(): ...\n", encoding="utf-8")
        doc = "1. `a_mod.py` — update\n2. `b_mod.py` — update\n"
        result = verify_taskplan_doc(doc, str(ws))
        assert any("`handle` defined" in f for f in result.findings)

    def test_same_name_in_different_dirs_is_ok(self, ws: Path) -> None:
        (ws / "a_mod.py").write_text("def handle(): ...\n", encoding="utf-8")
        sub = ws / "sub"
        sub.mkdir()
        (sub / "b_mod.py").write_text("def handle(): ...\n", encoding="utf-8")
        doc = "1. `a_mod.py` — update\n2. `sub/b_mod.py` — update\n"
        result = verify_taskplan_doc(doc, str(ws))
        assert result.clean


# ---------------------------------------------------------------------------
# plan_verifier: entity blocks must parse and define unique names
# ---------------------------------------------------------------------------

class TestEntitiesChecks:
    CLEAN = "```python\n@dataclass\nclass Config:\n    name: str\n```\n"
    BROKEN = "```python\ndef broken(:\n```\n"

    def test_clean_block_passes(self, tmp_path: Path) -> None:
        result = verify_entities_doc(self.CLEAN, str(tmp_path))
        assert result.clean and result.checked == 1

    def test_syntax_error_flagged_with_block_number(self, tmp_path: Path) -> None:
        result = verify_entities_doc(self.CLEAN + self.BROKEN, str(tmp_path))
        assert not result.clean
        assert "block #2" in result.findings[0]

    def test_duplicate_name_across_blocks_flagged(self, tmp_path: Path) -> None:
        dup = self.CLEAN + "```python\nclass Config:\n    other: int\n```\n"
        result = verify_entities_doc(dup, str(tmp_path))
        assert any("`Config` defined 2" in f for f in result.findings)


# ---------------------------------------------------------------------------
# dispatch + report rendering
# ---------------------------------------------------------------------------

def test_check_doc_unknown_kind_raises() -> None:
    with pytest.raises(ValueError):
        check_doc("spec", "text", ".")


def test_apply_report_is_idempotent_under_recheck(ws: Path) -> None:
    doc = "Touch `missing_thing.py`.\n"
    first = check_doc("plan", doc, str(ws))
    written = apply_report(doc, first)
    second = check_doc("plan", written, str(ws))
    assert second.flagged == first.flagged == 1
    assert written.count("## Verification Report") == 1


def test_report_section_caps_findings(ws: Path) -> None:
    from agent_core.commands.plan_verifier import PlanCheckResult

    result = PlanCheckResult("plan", checked=30, flagged=25,
                             findings=[f"f{i}" for i in range(25)])
    section = report_section(result)
    assert section.count("[UNVERIFIED]") == 20
    assert "5 more finding(s)" in section


def test_summarize_prints_findings(capsys) -> None:
    from agent_core.commands.plan_verifier import PlanCheckResult

    summarize(PlanCheckResult("plan", checked=2, flagged=1, findings=["boom"]), "plan")
    out = capsys.readouterr().out
    assert "Regression-checked 2 claims (1 flagged)" in out
    assert "[UNVERIFIED] boom" in out


# ---------------------------------------------------------------------------
# standalone commands halt / force-write through the gate
# ---------------------------------------------------------------------------

def _make_agent(ws: Path, reply: str) -> AsyncMock:
    agent = AsyncMock()
    agent.workspace = str(ws)
    agent.llm.chat = AsyncMock(return_value=reply)
    return agent


class TestStandaloneCommands:
    def test_plan_halts_on_hallucinated_modify_target(self, ws: Path) -> None:
        analysis = ws / "analysis.md"
        analysis.write_text("The module `existing.py` parses.\n", encoding="utf-8")
        plan_md = ws / "plan.md"
        agent = _make_agent(ws, "1. Update `ghost.py` — refactor internals\n")
        rc = asyncio.run(PlanCommand().execute(
            [str(analysis), str(plan_md)], agent))
        assert rc is True
        assert not plan_md.exists()  # autonomous default DECLINED

    def test_plan_force_writes_with_report(self, ws: Path) -> None:
        analysis = ws / "analysis.md"
        analysis.write_text("The module `existing.py` parses.\n", encoding="utf-8")
        plan_md = ws / "plan.md"
        agent = _make_agent(ws, "Touch `missing_thing.py`.\n")
        rc = asyncio.run(PlanCommand().execute(
            [str(analysis), str(plan_md), "--force"], agent))
        assert rc is True
        content = plan_md.read_text(encoding="utf-8")
        assert "Verification Report" in content

    def test_entities_halt_on_broken_blocks(self, ws: Path) -> None:
        analysis = ws / "analysis.md"
        analysis.write_text("x\n", encoding="utf-8")
        ent_md = ws / "entities.md"
        agent = _make_agent(ws, "```python\ndef broken(:\n```\n")
        rc = asyncio.run(EntitiesCommand().execute(
            [str(analysis), str(ent_md)], agent))
        assert rc is True
        assert not ent_md.exists()

    def test_taskplan_halt_on_duplicate_defs(self, ws: Path) -> None:
        (ws / "a_mod.py").write_text("def handle(): ...\n", encoding="utf-8")
        (ws / "b_mod.py").write_text("def handle(): ...\n", encoding="utf-8")
        analysis = ws / "analysis.md"
        analysis.write_text("x\n", encoding="utf-8")
        plan_md = ws / "plan.md"
        plan_md.write_text("# Plan\n", encoding="utf-8")
        tasks_md = ws / "tasks.md"
        agent = _make_agent(ws, "1. `a_mod.py` — update\n2. `b_mod.py` — update\n")
        rc = asyncio.run(TaskplanCommand().execute(
            [str(analysis), str(plan_md), str(tasks_md)], agent))
        assert rc is True
        assert not tasks_md.exists()

    def test_taskplan_clean_writes_without_report(self, ws: Path) -> None:
        analysis = ws / "analysis.md"
        analysis.write_text("x\n", encoding="utf-8")
        plan_md = ws / "plan.md"
        plan_md.write_text("# Plan\n", encoding="utf-8")
        tasks_md = ws / "tasks.md"
        agent = _make_agent(ws, "1. `existing.py` — extend parser\n")
        rc = asyncio.run(TaskplanCommand().execute(
            [str(analysis), str(plan_md), str(tasks_md)], agent))
        assert rc is True
        content = tasks_md.read_text(encoding="utf-8")
        assert "Verification Report" not in content


# ---------------------------------------------------------------------------
# workflow shared gate
# ---------------------------------------------------------------------------

class TestWorkflowPlanDocGate:
    def test_clean_content_returned_unchanged(self, ws: Path) -> None:
        content = "1. `existing.py` — extend parser\n"
        out = _plan_doc_gate("taskplan", content, str(ws), force=False)
        assert out == content

    def test_flagged_force_returns_content_with_report(self, ws: Path) -> None:
        content = "Touch `missing_thing.py`.\n"
        out = _plan_doc_gate("plan", content, str(ws), force=True)
        assert out is not None
        assert "Verification Report" in out

    def test_flagged_autonomous_halts(self, ws: Path) -> None:
        assert _plan_doc_gate(
            "taskplan", "1. [MODIFY] `ghost.py` — x\n", str(ws), force=False,
        ) is None

    def test_all_workflow_write_sites_use_the_gate(self) -> None:
        import inspect

        src = inspect.getsource(
            __import__("agent_core.commands.workflow_cmd",
                       fromlist=["_plan_doc_gate"]))
        # 9 inline write sites (3 branches × plan/entities/taskplan) + 1 def.
        assert src.count("_plan_doc_gate(") >= 10
        assert 'f.write(strip_reasoning(r, mode="light"))' not in src
