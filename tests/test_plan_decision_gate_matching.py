"""Regression tests for PlanDecisionGate boundary-aware matching.

Both bugs guarded here made a safe plan unexecutable, because a failed
decision gate aborts the whole ``plan exec`` run before any subagent starts.

Bug 1 — bare extension matched every guarded path
-------------------------------------------------
``_extract_plan_paths`` collected any backticked token containing a ``.``, so
prose such as "an ``ast`` walk of all 487 ``.py`` files" produced the plan path
``.py``. ``_path_matches`` then used raw substring containment, and
``".py" in "agent_core/file_system.py"`` is True — so the plan was reported as
modifying every decision-guarded ``*.py`` file, including the self-modification
guarded ``agent_core/security/allowlist.py``.

Bug 2 — three-character decision id matched inside a line number
---------------------------------------------------------------
Decision 035 has ``resolved_by == "007"``. The check was
``"007" in plan_text_lower``, which is True for the unrelated line number
``3007`` (a real ``_resolve_display_mode()`` call site in ``agent.py``), so the
gate claimed the plan "references resolved decision '007'".

The real regression
-------------------
``docs/refactor_agent_py_taskplan.md`` genuinely contains both patterns: it
mentions ``.py`` in prose and lists line number 3007. The gate must pass it
while still catching a plan that really targets a security-guarded path.

Pure in-memory checks against the real gate class: no LLM, no subagents.
"""
from __future__ import annotations

import pathlib

import pytest

from agent_core.commands.plan_decision_gate import (
    PlanDecisionGate,
    _is_path_like,
    _references_decision,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TASKPLAN = REPO_ROOT / "docs" / "refactor_agent_py_taskplan.md"


@pytest.fixture()
def gate() -> PlanDecisionGate:
    """The real gate pointed at this workspace's real .decisions.json."""
    return PlanDecisionGate(REPO_ROOT)


class TestBareExtensionIsNotAPath:
    """Bug 1: an extension names a file type, not a file."""

    @pytest.mark.parametrize("token", [".py", ".md", ".json"])
    def test_extension_only_token_rejected(self, token: str) -> None:
        assert _is_path_like(token) is False

    @pytest.mark.parametrize(
        "token",
        ["agent.py", "agent_core/file_system.py", "tests/test_x.py"],
    )
    def test_real_paths_still_accepted(self, token: str) -> None:
        assert _is_path_like(token) is True

    @pytest.mark.parametrize("token", ["", "   ", "487 .py files", "http://x/y.py"])
    def test_non_paths_rejected(self, token: str) -> None:
        assert _is_path_like(token) is False

    def test_bare_extension_not_collected_as_plan_path(self, gate: PlanDecisionGate) -> None:
        """The exact prose from the taskplan must not yield a ``.py`` plan path."""
        content = "measured by an `ast` walk of all 487 `.py` files (`backups/` excluded)"
        paths = gate._extract_plan_paths(content)
        assert ".py" not in paths
        assert "backups/" not in paths

    def test_bare_extension_does_not_match_guarded_file(self, gate: PlanDecisionGate) -> None:
        """``.py`` must never be read as touching a guarded .py file."""
        assert gate._path_matches("agent_core/file_system.py", {".py"}) is False
        assert gate._path_matches("agent_core/security/allowlist.py", {".py"}) is False


class TestPathMatchingIsBoundaryAware:
    """Guarded paths match on component boundaries, not raw substrings."""

    def test_exact_match(self, gate: PlanDecisionGate) -> None:
        assert gate._path_matches("agent.py", {"agent.py"}) is True

    def test_suffix_within_path(self, gate: PlanDecisionGate) -> None:
        assert gate._path_matches("agent.py", {"pkg/agent.py"}) is True

    def test_windows_separators_normalized(self, gate: PlanDecisionGate) -> None:
        assert gate._path_matches("agent.py", {"C:\\Dev\\Agent1\\agent.py"}) is True

    def test_partial_filename_does_not_match(self, gate: PlanDecisionGate) -> None:
        """``agent.py`` must not match the unrelated ``not_agent.py``."""
        assert gate._path_matches("agent.py", {"not_agent.py"}) is False

    def test_guarded_directory_covers_children(self, gate: PlanDecisionGate) -> None:
        assert (
            gate._path_matches(
                "agent_core/security/", {"agent_core/security/allowlist.py"}
            )
            is True
        )

    def test_similar_directory_name_does_not_match(self, gate: PlanDecisionGate) -> None:
        """``security/`` must not match the unrelated ``security_notes.md``."""
        assert gate._path_matches("agent_core/security/", {"security_notes.md"}) is False

    def test_empty_paths_never_match(self, gate: PlanDecisionGate) -> None:
        assert gate._path_matches("", {"agent.py"}) is False
        assert gate._path_matches("agent.py", set()) is False


class TestResolvedDecisionIdBoundary:
    """Bug 2: a short decision id must not match inside a longer token."""

    def test_id_inside_line_number_does_not_match(self) -> None:
        """The real trigger: ``007`` inside the line number ``3007``."""
        assert _references_decision("007", "| 2481, 2792, 3007, 3053 |") is False

    def test_id_inside_longer_id_does_not_match(self) -> None:
        assert _references_decision("007", "id 0070") is False

    def test_standalone_id_matches(self) -> None:
        assert _references_decision("007", "uses decision 007 here") is True

    def test_empty_id_never_matches(self) -> None:
        assert _references_decision("", "anything") is False


class TestRealTaskplanPassesGate:
    """End-to-end guard: the real refactor taskplan must be executable."""

    def test_taskplan_file_exists(self) -> None:
        assert TASKPLAN.is_file(), f"missing taskplan: {TASKPLAN}"

    def test_taskplan_passes_decision_gate(self, gate: PlanDecisionGate) -> None:
        """This is the regression: it used to fail with 3 false violations."""
        content = TASKPLAN.read_text(encoding="utf-8")
        result = gate.validate(content)
        assert result.violations == [], (
            "taskplan must pass the decision gate; false positives found: "
            f"{result.violations}"
        )
        assert result.passed is True
        assert result.metadata is not None
        assert result.metadata["decisions_checked"] > 0

    def test_taskplan_contains_the_triggering_patterns(self) -> None:
        """Prove the fixture really exercises both bugs, so the test cannot rot."""
        content = TASKPLAN.read_text(encoding="utf-8")
        assert "`.py`" in content, "fixture must contain a bare `.py` token"
        assert "3007" in content, "fixture must contain line number 3007"


class TestTruePositivesStillCaught:
    """The fix must not blind the gate to genuine violations."""

    def test_security_guarded_write_is_still_flagged(self, gate: PlanDecisionGate) -> None:
        bad = "Write `agent_core/security/allowlist.py` and bypass the security guard."
        result = gate.validate(bad)
        assert result.passed is False
        assert any("allowlist.py" in v for v in result.violations)

    def test_raw_json_loads_without_pydantic_is_still_flagged(
        self, gate: PlanDecisionGate
    ) -> None:
        """Decision 001 guards `agent_core/llm/tool_loop.py` (not `agent.py`).

        The guarded path matters: the json.loads check only fires once the plan
        actually touches a file the decision protects.
        """
        bad = "Call json.loads on the payload in `agent_core/llm/tool_loop.py`."
        result = gate.validate(bad)
        assert result.passed is False
        assert any("json.loads" in v for v in result.violations)

    def test_decision_001_guard_path_is_the_one_used_above(
        self, gate: PlanDecisionGate
    ) -> None:
        """Keep the fixture honest: decision 001 must still guard that path."""
        decisions = gate._load_decisions()
        guarded = {
            path
            for d in decisions
            if {"pydantic-schema", "input-validation"} & set(d.get("tags") or [])
            for path in d.get("affected_files") or []
        }
        assert "agent_core/llm/tool_loop.py" in guarded

    def test_referencing_resolved_decision_is_still_flagged(
        self, gate: PlanDecisionGate
    ) -> None:
        """A real reference to resolved decision 007 must still be caught."""
        bad = "We revert to the old design from decision 007 in `agent.py`."
        result = gate.validate(bad)
        assert result.passed is False
        assert any("007" in v for v in result.violations)
