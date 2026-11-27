"""The planner -> executor contract must be self-consistent.

Plan mode saves its final answer to ``.docs/<ts>/plan_proposed.md`` and
``plan exec`` then reads that same file expecting a ``## Tasks`` block.  Both
halves are pinned here so the two can never drift apart again:

* the persisted header must satisfy :func:`validate_plan_markdown`;
* a ``## Tasks`` body written by a planner must survive
  :func:`parse_tasks_from_markdown` (i.e. ``plan exec`` sees real tasks);
* the plan-mode prompt must actually ask for that block.

Regression: the persisted title was ``# Proposed plan`` (lowercase "p") while
the schema required ``# Proposed Plan``, and nothing in the prompt ever asked
for a ``## Tasks`` block — so every plan-mode artifact failed its own schema
and ``plan exec`` reported "No tasks found".
"""

from __future__ import annotations

import asyncio

import pytest

from agent import Agent
from agent_core.commands.plan_schema import validate_plan_markdown
from agent_core.modes import plan_mode_system_suffix, plan_mode_turn_note
from agent_core.plan_execution.parser import parse_tasks_from_markdown

#: A plan body shaped the way plan mode is instructed to end its answer.
PLAN_BODY = (
    "Rename the resolver so plan exec can consume this plan.\n"
    "\n"
    "## Tasks\n"
    "\n"
    "- [T1] Extract the resolver into workspace.py (role: implementer)\n"
    "- [T2] Wire the CLI flag to the resolver (role: implementer) deps: T1\n"
    "- [T3] Cover both halves with tests (role: tester) deps: T2\n"
)


@pytest.fixture()
def persisted_plan(tmp_path):
    """Persist *PLAN_BODY* through the real agent path; return (path, text)."""
    planner = Agent(workspace=str(tmp_path))
    planner._persist_plan_answer(PLAN_BODY)
    run_dirs = [d for d in (tmp_path / ".docs").iterdir() if d.is_dir()]
    assert run_dirs, "expected a fresh .docs run folder"
    out = run_dirs[0] / "plan_proposed.md"
    return out, out.read_text(encoding="utf-8")


class TestPersistedPlanMeetsSchema:
    def test_persisted_plan_passes_its_own_validator(self, persisted_plan) -> None:
        _path, text = persisted_plan
        ok, errors = validate_plan_markdown(text)
        assert ok is True, f"persisted plan failed its own schema: {errors}"

    def test_persisted_plan_keeps_the_planner_body(self, persisted_plan) -> None:
        _path, text = persisted_plan
        assert "Rename the resolver" in text
        assert "## Tasks" in text


class TestPlanExecCanConsumePersistedPlan:
    def test_persisted_plan_yields_the_planner_tasks(self, persisted_plan) -> None:
        """``plan exec`` parses exactly this file — it must not be empty."""
        _path, text = persisted_plan
        tasks = parse_tasks_from_markdown(text)
        assert [t.id for t in tasks] == ["T1", "T2", "T3"]
        assert tasks[1].depends_on == ["T1"]
        assert tasks[2].role == "tester"


class TestPlanGenerateWritesAValidPlan:
    """``plan <analysis.md> <plan.md>`` must not halt on its own output.

    The command wraps the LLM answer in a hard-coded title and then runs
    :func:`validate_plan_markdown` over the result.  The title used to be
    ``# Coding Plan``, which fails the ``# Proposed Plan`` requirement — so
    every generate run stopped at "Plan schema validation failed".
    """

    def test_generated_plan_validates_and_parses(self, tmp_path, monkeypatch) -> None:
        workspace = tmp_path
        (workspace / "analysis.md").write_text("# Analysis\n\nNeeds work.\n")

        planner = Agent(workspace=str(workspace))

        class _FakeLlm:
            async def chat(self, messages):
                return (
                    "Do the work.\n\n## Tasks\n\n"
                    "- [T1] Extract the resolver (role: implementer)\n"
                    "- [T2] Cover it with tests (role: tester) deps: T1\n"
                )

        monkeypatch.setattr(planner, "llm", _FakeLlm(), raising=False)
        # Autonomy would silently decline the schema prompt; the assertion is
        # that the prompt is never reached, and no halt happens either way.
        monkeypatch.setattr(
            "agent_core.commands.plan_cmd.auto_choice", lambda *a, **k: "n"
        )

        async def _run() -> None:
            from agent_core.commands.plan_cmd import PlanCommand

            assert await PlanCommand().execute(["analysis.md", "plan.md"], planner)

        asyncio.run(_run())

        written = list((workspace / ".docs").rglob("plan.md"))
        assert written, "expected the generated plan to be written"
        text = written[0].read_text(encoding="utf-8")
        ok, errors = validate_plan_markdown(text)
        assert ok is True, f"generate wrote a schema-invalid plan: {errors}"

        tasks = parse_tasks_from_markdown(text)
        assert [t.id for t in tasks] == ["T1", "T2"]


class TestSchemaHeadingCase:
    """The title check must not hinge on capitalisation alone."""

    def test_sniffed_lowercase_title_is_accepted(self) -> None:
        ok, errors = validate_plan_markdown(
            "# Proposed plan\n\n## Tasks\n\n- [T1] Do it\n"
        )
        assert ok is True, errors

    def test_missing_tasks_heading_still_rejected(self) -> None:
        ok, errors = validate_plan_markdown("# Proposed Plan\n\n- [T1] Do it\n")
        assert ok is False
        assert any("tasks" in e.lower() for e in errors)

    def test_missing_title_still_rejected(self) -> None:
        ok, errors = validate_plan_markdown("## Tasks\n\n- [T1] Do it\n")
        assert ok is False
        assert any("heading" in e.lower() for e in errors)


class TestDecisionGateStatusNormalization:
    """A decision without an explicit ``status`` is ACTIVE, not invisible.

    ``agent_core.decisions.load_decisions`` normalizes a missing or unknown
    status to ``active``.  The gate used to filter the raw JSON for an exact
    ``== "active"``, so every hand-written or older ``.decisions.json`` (no
    ``status`` key) yielded ``decisions_checked: 0`` — the gate silently
    passed plans it was meant to block.
    """

    @staticmethod
    def _gate(tmp_path, decisions):
        import json

        from agent_core.commands.plan_decision_gate import PlanDecisionGate

        (tmp_path / ".decisions.json").write_text(json.dumps(decisions))
        return PlanDecisionGate(tmp_path)

    def test_missing_status_is_treated_as_active(self, tmp_path) -> None:
        gate = self._gate(
            tmp_path,
            [
                {
                    "id": "002",
                    "title": "Self-modification guard",
                    "tags": ["self-modification-prevention"],
                    "affected_files": ["agent_core/security/guard.py"],
                    "contradictions": [],
                    "resolved_by": None,
                }
            ],
        )
        result = gate.validate(
            "# Proposed Plan\n\n## Tasks\n\n"
            "- [T1] Modify `agent_core/security/guard.py`\n"
        )
        assert result.metadata["decisions_checked"] == 1
        assert result.passed is False

    def test_inactive_statuses_stay_out_of_scope(self, tmp_path) -> None:
        gate = self._gate(
            tmp_path,
            [
                {"id": "010", "title": "Old", "affected_files": ["a.py"],
                 "status": "superseded"},
                {"id": "011", "title": "Older", "affected_files": ["b.py"],
                 "status": "archived"},
                {"id": "012", "title": "Tombstone", "affected_files": ["c.py"],
                 "status": "void"},
            ],
        )
        result = gate.validate(
            "# Proposed Plan\n\n## Tasks\n\n- [T1] Touch `a.py`, `b.py`, `c.py`\n"
        )
        assert result.metadata["decisions_checked"] == 0
        assert result.passed is True

    def test_unknown_status_falls_back_to_active(self, tmp_path) -> None:
        gate = self._gate(
            tmp_path,
            [
                {"id": "013", "title": "Typo", "affected_files": ["d.py"],
                 "status": "activ"},
            ],
        )
        assert gate.validate(
            "# Proposed Plan\n\n## Tasks\n\n- [T1] Touch `d.py`\n"
        ).metadata["decisions_checked"] == 1


class TestPlanModePromptAsksForTasks:
    """The prompt is the only thing that can make the body parseable."""

    def test_system_suffix_requires_a_tasks_block(self) -> None:
        suffix = plan_mode_system_suffix()
        assert "## Tasks" in suffix

    def test_turn_note_requires_a_tasks_block(self) -> None:
        assert "## Tasks" in plan_mode_turn_note()
