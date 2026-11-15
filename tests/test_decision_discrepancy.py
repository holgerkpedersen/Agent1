"""Phase 2 of the autonomous-decision plan: discrepancy detection.

Card ``9bc4327f`` ("Improve Autonomous Decisions in Agent1"), plan
``.decisions_plan.md`` — Phase 2 is the foundation phase: a detector that
compares an executed tool call against the active decisions in
``.decisions.json``, a JSONL ledger that collects what it finds, and the
hook in ``AgentCore._execute_tool_call`` that runs the check on every tool
call.

These tests drive the REAL modules (``agent_core.discrepancy``,
``agent_core.decisions``, ``agent_core.agent.AgentCore``) against a temp
workspace with its own ``.decisions.json`` — no LLM, no network.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_core.decisions import (
    DISCREPANCIES_FILE,
    load_discrepancies,
    record_discrepancy,
)
from agent_core.discrepancy import Discrepancy, check_decision_compliance

#: Decision #002 of the real ledger: tag what self-modification decisions
#: carry, so the guard is derived from the ledger, not hard-coded per id.
SELF_MOD_TAGS = ["self-modification-prevention", "integrity-protection"]


def _workspace(tmp_path, decisions):
    (tmp_path / ".decisions.json").write_text(
        json.dumps(decisions), encoding="utf-8"
    )
    return tmp_path


def _guarded_decision(decision_id="002", files=("agent_core/security/allowlist.py",)):
    return {
        "id": decision_id,
        "title": "Self-modification guard to prevent agent tampering",
        "decision": "Block all write operations targeting agent_core/security/.",
        "rationale": "Guardrails must not be editable from within.",
        "affected_files": list(files),
        "tags": list(SELF_MOD_TAGS),
        "contradictions": [],
        "status": "active",
        "category": "security",
    }


# ── check_decision_compliance: guard violations ────────────────────────────


def test_flags_blocked_write_to_guarded_path(tmp_path):
    """A mutating call aimed at a self-modification-guarded file, refused by
    the guard, is a `blocked_action` warning attributed to that decision."""
    ws = _workspace(tmp_path, [_guarded_decision()])

    found = check_decision_compliance(
        ws,
        "edit",
        {"path": "agent_core/security/allowlist.py", "content": "x"},
        "Error: self-modification guard refused the write",
    )

    assert len(found) == 1
    disc = found[0]
    assert isinstance(disc, Discrepancy)
    assert disc.decision_id == "002"
    assert disc.action_tool == "edit"
    assert disc.discrepancy_type == "blocked_action"
    assert disc.severity == "warning"
    assert disc.suggestion


def test_guarded_write_that_succeeded_is_a_critical_contradiction(tmp_path):
    """If the guarded write actually WENT THROUGH, a written decision is being
    contradicted in practice — that is critical, not a warning."""
    ws = _workspace(tmp_path, [_guarded_decision()])

    found = check_decision_compliance(
        ws,
        "write",
        {"path": "agent_core/security/allowlist.py", "content": "x"},
        "Written agent_core/security/allowlist.py\n[verify] py_compile OK",
    )

    assert len(found) == 1
    assert found[0].discrepancy_type == "contradiction"
    assert found[0].severity == "critical"


def test_guarded_path_and_tag_match_without_existing_file(tmp_path):
    """Regression guard: ``normalize_affected_files`` drops paths that do not
    exist, so a fresh checkout used to lose the guarded directory entirely and
    the guard silently never fired.  The tag-derived guard must still apply."""
    decision = _guarded_decision(files=())
    ws = _workspace(tmp_path, [decision])
    assert not (ws / "agent_core" / "security").exists()

    found = check_decision_compliance(
        ws,
        "edit",
        {"path": "agent_core/security/allowlist.py", "content": "x"},
        "Written agent_core/security/allowlist.py",
    )

    assert [d.severity for d in found] == ["critical"]


def test_read_of_guarded_path_is_not_a_discrepancy(tmp_path):
    """Reading the guarded tree is explicitly allowed — no discrepancy."""
    ws = _workspace(tmp_path, [_guarded_decision()])

    assert (
        check_decision_compliance(
            ws,
            "read",
            {"path": "agent_core/security/allowlist.py"},
            "def check_shell_command(cmd: str) -> str: ...",
        )
        == []
    )


def test_unrelated_write_is_clean(tmp_path):
    ws = _workspace(tmp_path, [_guarded_decision()])

    assert (
        check_decision_compliance(
            ws,
            "write",
            {"path": "docs/notes.md", "content": "hi"},
            "Written docs/notes.md",
        )
        == []
    )


def test_no_decisions_means_no_discrepancies(tmp_path):
    assert check_decision_compliance(tmp_path, "write", {"path": "a.py"}, "ok") == []


# ── check_decision_compliance: plan-mode rejection ─────────────────────────


def test_flags_plan_mode_rejection(tmp_path):
    """A mutating call the executor rejected because the session is read-only
    is an autonomous-decision conflict worth recording."""
    ws = _workspace(tmp_path, [])

    found = check_decision_compliance(
        ws,
        "write",
        {"path": "agent.py", "content": "x"},
        "[plan mode] 'write' is blocked: Plan mode is read-only — no file in "
        "the workspace may be changed.",
    )

    assert len(found) == 1
    assert found[0].decision_id == "plan-mode"
    assert found[0].discrepancy_type == "blocked_action"
    assert found[0].severity == "warning"


# ── check_decision_compliance: unresolved context ──────────────────────────


def test_flags_action_in_area_with_open_contradiction(tmp_path):
    """A decision still carrying unresolved contradictions is unsettled; acting
    on its files is flagged as `unresolved_context`."""
    decision = _guarded_decision(decision_id="054")
    decision.update(
        affected_files=["agent_core/decisions.py"],
        tags=["ledger"],
        contradictions=["\u2194 #003"],
    )
    ws = _workspace(tmp_path, [decision])

    found = check_decision_compliance(
        ws,
        "edit",
        {"path": "agent_core/decisions.py", "content": "x"},
        "Edited agent_core/decisions.py",
    )

    assert len(found) == 1
    assert found[0].decision_id == "054"
    assert found[0].discrepancy_type == "unresolved_context"
    assert found[0].severity == "warning"


# ── JSONL ledger ───────────────────────────────────────────────────────────


def test_record_and_load_discrepancy_roundtrip(tmp_path):
    record_discrepancy(
        tmp_path,
        Discrepancy(
            decision_id="002",
            decision_title="guard",
            action_tool="edit",
            action_args={"path": "agent_core/security/allowlist.py"},
            discrepancy_type="blocked_action",
            severity="warning",
        ),
    )
    record_discrepancy(
        tmp_path,
        {
            "decision_id": "054",
            "decision_title": "ledger",
            "action_tool": "write",
            "action_args": {},
            "discrepancy_type": "unresolved_context",
            "severity": "critical",
        },
    )

    lines = (tmp_path / DISCREPANCIES_FILE).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["decision_id"] == "002"

    assert len(load_discrepancies(tmp_path)) == 2
    assert [d["decision_id"] for d in load_discrepancies(tmp_path, "054")] == ["054"]
    critical = load_discrepancies(tmp_path, severity="critical")
    assert [d["severity"] for d in critical] == ["critical"]


def test_load_discrepancies_missing_file_returns_empty(tmp_path):
    assert load_discrepancies(tmp_path) == []


def test_load_discrepancies_skips_corrupt_lines(tmp_path):
    (tmp_path / DISCREPANCIES_FILE).write_text(
        '{"decision_id": "001"}\nnot json at all\n', encoding="utf-8"
    )
    assert [d["decision_id"] for d in load_discrepancies(tmp_path)] == ["001"]


# ── AgentCore integration ──────────────────────────────────────────────────


def _agent(tmp_path):
    from agent import Agent

    return Agent(workspace=str(tmp_path))


def test_execute_tool_call_logs_plan_mode_rejection(tmp_path):
    """The real executor hook: a plan-mode rejection during a tool call must
    land in the ledger without changing the returned result."""
    agent = _agent(tmp_path)
    agent.set_mode("plan")

    result = asyncio.run(
        agent._execute_tool_call(
            "write", {"path": str(tmp_path / "x.py"), "content": "x = 1\n"}
        )
    )

    assert "[plan mode]" in result
    logged = load_discrepancies(tmp_path)
    assert len(logged) == 1
    assert logged[0]["action_tool"] == "write"
    assert logged[0]["decision_id"] == "plan-mode"


def test_execute_tool_call_creates_no_ledger_when_clean(tmp_path):
    """A clean tool call must not leave a `.discrepancies.jsonl` behind —
    otherwise every workspace gets a spurious dirty file."""
    agent = _agent(tmp_path)

    result = asyncio.run(
        agent._execute_tool_call(
            "write", {"path": str(tmp_path / "ok.py"), "content": "y = 2\n"}
        )
    )

    assert "Written" in result
    assert not (tmp_path / DISCREPANCIES_FILE).exists()


def test_execute_tool_call_survives_broken_decision_ledger(tmp_path):
    """A corrupt `.decisions.json` must not break tool execution."""
    (tmp_path / ".decisions.json").write_text("{not json", encoding="utf-8")
    agent = _agent(tmp_path)
    agent.set_mode("plan")

    result = asyncio.run(
        agent._execute_tool_call(
            "write", {"path": str(tmp_path / "z.py"), "content": "z = 3\n"}
        )
    )

    assert "[plan mode]" in result


@pytest.mark.parametrize("tool", ["edit", "write", "fix", "run", "git"])
def test_mutating_tools_are_covered(tmp_path, tool):
    """Every tool that can change the workspace counts as an action for the
    guard check, not just write/edit."""
    ws = _workspace(tmp_path, [_guarded_decision()])

    found = check_decision_compliance(
        ws, tool, {"path": "agent_core/security/allowlist.py"}, "blocked"
    )

    assert len(found) == 1
