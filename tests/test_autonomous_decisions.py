"""Regression tests for autonomous-decision capture (Phase 1).

Covers ``agent_core.decisions`` (detection, the observation log, promotion to
the ledger) and the end-to-end hook in ``Agent._execute_tool_call``:

- detection: file targets the user never named, git commit/merge (both the
  ``run``-a-command shape and the ``git`` tool shape), delegation, plan
  transitions
- non-decisions: read-only tools, refusals, plan-mode rejections, error text,
  and paths the user asked for by name
- promotion: idempotent (a repeated detection is logged but never duplicated),
  one ledger entry per target, ``source="agent"`` + ``autonomous`` tagging
- prompt block: recent autonomous decisions are re-stated, and can be opted out
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agent import Agent
from agent_core.decisions import (
    AUTONOMOUS_CATEGORY,
    AUTONOMOUS_LOG_FILE,
    AUTONOMOUS_TAG,
    SOURCE_AUTONOMOUS,
    decisions_as_system_prompt,
    detect_autonomous_decisions,
    load_autonomous_log,
    load_decisions,
    record_autonomous_decision,
    record_autonomous_decisions,
    save_decisions,
)


# ---------------------------------------------------------------------------
# Detection — what counts as a decision
# ---------------------------------------------------------------------------


def test_detects_file_target_the_user_never_named(tmp_path: Path) -> None:
    found = detect_autonomous_decisions(
        tmp_path,
        "write",
        {"path": str(tmp_path / "notes" / "plan.md"), "content": "x"},
        "Written 1 bytes to notes/plan.md",
        user_input="add a hello world function",
    )
    assert len(found) == 1
    assert found[0]["title"] == "Change notes/plan.md"
    assert found[0]["affected_files"] == ["notes/plan.md"]
    assert AUTONOMOUS_TAG in found[0]["tags"]
    assert "write" in found[0]["tags"]


def test_skips_a_path_the_user_named(tmp_path: Path) -> None:
    """A path in the user's own request is an instruction, not a choice."""
    found = detect_autonomous_decisions(
        tmp_path,
        "write",
        {"path": str(tmp_path / "notes" / "plan.md"), "content": "x"},
        "Written 1 bytes to notes/plan.md",
        user_input="please write notes/plan.md with the outline",
    )
    assert found == []


def test_reports_a_deletion(tmp_path: Path) -> None:
    """A shell delete has no path argument — the target comes from the result.

    Titled "Delete" (not "Change") because the result says so: the verb is the
    only thing distinguishing a removal from a modification here.
    """
    found = detect_autonomous_decisions(
        tmp_path,
        "run",
        {"command": "del gone.txt"},
        "Deleted gone.txt",
    )
    assert [c["title"] for c in found] == ["Delete gone.txt"]
    assert found[0]["affected_files"] == ["gone.txt"]


def test_git_tool_commit_is_a_decision(tmp_path: Path) -> None:
    """The ``git`` tool passes the verb in ``subcommand``, not inside ``args``.

    Regression: reading only ``args`` would miss every commit made through the
    git tool and silently record nothing.
    """
    found = detect_autonomous_decisions(
        tmp_path, "git", {"subcommand": "commit", "args": '-m "wip"'}, "..."
    )
    assert [c["title"] for c in found] == ["Ran git commit autonomously"]
    assert found[0]["severity"] == "warning"


def test_shell_git_merge_is_a_decision(tmp_path: Path) -> None:
    found = detect_autonomous_decisions(
        tmp_path, "run", {"command": "git -C . merge feature"}, "..."
    )
    assert [c["title"] for c in found] == ["Ran git merge autonomously"]


def test_git_status_is_not_a_decision(tmp_path: Path) -> None:
    found = detect_autonomous_decisions(
        tmp_path, "git", {"subcommand": "status", "args": ""}, "..."
    )
    assert found == []


def test_delegation_is_a_decision(tmp_path: Path) -> None:
    found = detect_autonomous_decisions(
        tmp_path, "delegate", {"role": "tester", "task": "run tests"}, "..."
    )
    assert [c["title"] for c in found] == ["Delegated work to tester"]


def test_plan_transition_is_a_decision(tmp_path: Path) -> None:
    found = detect_autonomous_decisions(tmp_path, "plan_start", {}, "...")
    assert [c["title"] for c in found] == ["Started a plan autonomously"]


def test_every_target_gets_its_own_candidate(tmp_path: Path) -> None:
    found = detect_autonomous_decisions(
        tmp_path,
        "edit",
        {"path": str(tmp_path / "a.py"), "old_text": "x", "new_text": "y"},
        "Edited a.py",
    )
    assert len(found) == 1
    assert found[0]["affected_files"] == ["a.py"]


# ---------------------------------------------------------------------------
# Detection — what must NOT be recorded
# ---------------------------------------------------------------------------


def test_read_only_tool_is_not_a_decision(tmp_path: Path) -> None:
    assert detect_autonomous_decisions(tmp_path, "read", {"path": "a.py"}, "hi") == []


def test_plan_mode_rejection_is_not_a_decision(tmp_path: Path) -> None:
    found = detect_autonomous_decisions(
        tmp_path,
        "write",
        {"path": str(tmp_path / "a.py")},
        "[plan mode] write is a mutating tool",
    )
    assert found == []


@pytest.mark.parametrize(
    "result",
    [
        "Blocked by sandbox: rm refused",
        "write error: permission denied",
        "unknown tool 'frobnicate'",
        "No changes made.",
    ],
)
def test_failed_call_is_not_a_decision(tmp_path: Path, result: str) -> None:
    found = detect_autonomous_decisions(
        tmp_path, "write", {"path": str(tmp_path / "a.py")}, result
    )
    assert found == []


# ---------------------------------------------------------------------------
# Promotion to the ledger + observation log
# ---------------------------------------------------------------------------


def test_promotes_to_ledger_with_agent_provenance(tmp_path: Path) -> None:
    # The write succeeded, so the file exists: `normalize_affected_files`
    # drops absent paths on purpose (design decision, see
    # tests/test_decision_paths.py), which is the right rule for a record that
    # describes what was actually changed.
    (tmp_path / "x.py").write_text("x = 1\n", encoding="utf-8")
    candidates = detect_autonomous_decisions(
        tmp_path, "write", {"path": str(tmp_path / "x.py")}, "Written x.py"
    )
    stored = record_autonomous_decisions(tmp_path, candidates)
    assert len(stored) == 1

    ledger = load_decisions(tmp_path)
    assert len(ledger) == 1
    record = ledger[0]
    assert record["category"] == AUTONOMOUS_CATEGORY
    assert record["source"] == SOURCE_AUTONOMOUS
    assert AUTONOMOUS_TAG in record["tags"]
    assert record["affected_files"] == ["x.py"]


def test_repeated_detection_is_logged_but_never_duplicated(tmp_path: Path) -> None:
    """The same change seen twice must not create two ledger entries."""
    args = {"path": str(tmp_path / "x.py")}
    first = detect_autonomous_decisions(tmp_path, "write", args, "Written x.py")
    second = detect_autonomous_decisions(tmp_path, "write", args, "Written x.py")

    assert len(record_autonomous_decisions(tmp_path, first)) == 1
    assert record_autonomous_decisions(tmp_path, second) == []

    assert len(load_decisions(tmp_path)) == 1
    log = load_autonomous_log(tmp_path)
    assert len(log) == 2
    assert log[0]["decision_id"]
    assert log[1]["decision_id"] is None


def test_log_is_appended_as_json_lines(tmp_path: Path) -> None:
    candidates = detect_autonomous_decisions(
        tmp_path, "write", {"path": str(tmp_path / "x.py")}, "Written x.py"
    )
    record_autonomous_decisions(tmp_path, candidates)
    raw = (tmp_path / AUTONOMOUS_LOG_FILE).read_text(encoding="utf-8")
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["title"] == "Change x.py"
    assert parsed["detected_at"]


def test_truncated_log_line_is_skipped(tmp_path: Path) -> None:
    (tmp_path / AUTONOMOUS_LOG_FILE).write_text('{"title": "ok"}\n{"broken\n')
    assert [e["title"] for e in load_autonomous_log(tmp_path)] == ["ok"]


def test_empty_candidates_do_not_create_files(tmp_path: Path) -> None:
    assert record_autonomous_decisions(tmp_path, []) == []
    assert not (tmp_path / AUTONOMOUS_LOG_FILE).exists()


def test_record_autonomous_decision_normalises_tags(tmp_path: Path) -> None:
    record = record_autonomous_decision(
        tmp_path, "Picked SQLite", tags=["Autonomous", "storage", "storage"]
    )
    assert record["tags"] == [AUTONOMOUS_TAG, "storage"]
    assert record["source"] == SOURCE_AUTONOMOUS


def test_record_autonomous_decision_needs_a_title(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        record_autonomous_decision(tmp_path, "   ")


# ---------------------------------------------------------------------------
# Prompt block
# ---------------------------------------------------------------------------


def test_prompt_block_restates_recent_autonomous_decisions(tmp_path: Path) -> None:
    record_autonomous_decision(
        tmp_path, "Chose SQLite", decision="Use SQLite for the cache."
    )
    block = decisions_as_system_prompt(tmp_path, [])
    assert "RECENT AUTONOMOUS DECISIONS" in block
    assert "Chose SQLite" in block


def test_prompt_block_can_opt_out(tmp_path: Path) -> None:
    record_autonomous_decision(tmp_path, "Chose SQLite")
    assert decisions_as_system_prompt(tmp_path, [], include_autonomous=False) == ""


def test_prompt_block_is_empty_without_decisions(tmp_path: Path) -> None:
    assert decisions_as_system_prompt(tmp_path, []) == ""


# ---------------------------------------------------------------------------
# End-to-end through the real executor hook
# ---------------------------------------------------------------------------


def test_executor_records_an_autonomous_write(tmp_path: Path) -> None:
    """The choke point must capture a write the user never asked for."""
    agent = Agent(workspace=str(tmp_path))
    target = tmp_path / "scratch.py"
    result = asyncio.run(
        agent._execute_tool_call("write", {"path": str(target), "content": "x = 1"})
    )
    assert target.exists(), result

    ledger = load_decisions(tmp_path)
    assert [r["title"] for r in ledger] == ["Change scratch.py"]
    assert ledger[0]["source"] == SOURCE_AUTONOMOUS
    assert load_autonomous_log(tmp_path)


def test_executor_records_a_user_named_write_as_nothing(tmp_path: Path) -> None:
    """A path the user named in the turn belongs to the user, not the agent."""
    agent = Agent(workspace=str(tmp_path))
    target = tmp_path / "asked.py"
    agent._last_user_input = f"create {target}"
    asyncio.run(
        agent._execute_tool_call("write", {"path": str(target), "content": "y = 2"})
    )
    assert load_decisions(tmp_path) == []


# ---------------------------------------------------------------------------
# Regression: scratch probes must not enter the ledger (audit invariant)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["tmp_qw6_probe.py", "_tmp_probe.py", "scratch_notes.md", "run.bak"],
)
def test_scratch_targets_are_never_recorded(tmp_path: Path, name: str) -> None:
    """A scratch probe is deleted by design, so remembering it goes stale.

    Regression: decision #141 recorded ``tmp_qw6_probe.py``; deleting that
    probe left a stale affected_file and broke the stale-reference audit
    (decision #054), failing ``scripts/audit_invariants.py`` on a clean tree.
    """
    found = detect_autonomous_decisions(
        tmp_path,
        "write",
        {"path": str(tmp_path / name), "content": "x"},
        f"Written 1 bytes to {name}",
        user_input="investigate the bug",
    )
    assert found == []


def test_real_source_targets_are_still_recorded(tmp_path: Path) -> None:
    """The scratch guard must not silence genuine source decisions."""
    found = detect_autonomous_decisions(
        tmp_path,
        "write",
        {"path": str(tmp_path / "agent_core" / "cache.py"), "content": "x"},
        "Written 1 bytes to agent_core/cache.py",
        user_input="investigate the bug",
    )
    assert len(found) == 1
    assert found[0]["affected_files"] == ["agent_core/cache.py"]


def test_scratch_refs_are_not_reported_stale(tmp_path: Path) -> None:
    """An already-recorded scratch probe must not fail the stale audit."""
    from agent_core.decisions import find_stale_decisions

    ledger = [
        {
            "id": "141",
            "title": "Change tmp_qw6_probe.py",
            "affected_files": ["tmp_qw6_probe.py"],
            "status": "active",
        }
    ]
    assert find_stale_decisions(tmp_path, ledger) == []


def test_missing_real_file_is_still_reported_stale(tmp_path: Path) -> None:
    """The scratch exemption must not weaken the real stale check."""
    from agent_core.decisions import find_stale_decisions

    ledger = [
        {
            "id": "7",
            "title": "Change agent_core/gone.py",
            "affected_files": ["agent_core/gone.py"],
            "status": "active",
        }
    ]
    stale = find_stale_decisions(tmp_path, ledger)
    assert len(stale) == 1
    assert stale[0]["_missing_files"] == ["agent_core/gone.py"]


# ---------------------------------------------------------------------------
# Regression: a tombstone must survive a load/save round-trip
# ---------------------------------------------------------------------------


def test_void_status_survives_a_roundtrip(tmp_path: Path) -> None:
    """``void`` is a real status (tombstone, decision #043).

    Regression: ``_STATUSES`` omitted it, so ``load_decisions`` classified the
    record as "unknown" and rewrote the tombstone to ``active``.  One later
    ``save_decisions`` (e.g. ``decide set-status``) then silently resurrected a
    reserved ID as an authoritative architectural decision.
    """
    from agent_core.decisions import STATUS_VOID

    save_decisions(
        tmp_path,
        [{"id": "043", "title": "ID 043 is void", "status": "void"}],
    )
    loaded = load_decisions(tmp_path)
    assert loaded[0]["status"] == STATUS_VOID == "void"


def test_unknown_status_still_falls_back_to_active(tmp_path: Path) -> None:
    """The ``void`` fix must not weaken the legacy normalisation."""
    save_decisions(
        tmp_path,
        [{"id": "001", "title": "Legacy", "status": "wat"}],
    )
    assert load_decisions(tmp_path)[0]["status"] == "active"


# ---------------------------------------------------------------------------
# Regression: a foreign path is not a decision about THIS repository
# ---------------------------------------------------------------------------


def test_out_of_workspace_target_is_never_recorded(tmp_path: Path) -> None:
    """A write into ``%TEMP%`` leaves a permanently stale affected_file.

    Regression: the plan-mode executor work wrote to a pytest ``tmp_path`` and
    the call was recorded as a decision about the repo, adding ledger entries
    no checkout could ever satisfy (stale-reference audit, decision #054).
    """
    outside = tmp_path / "elsewhere" / "case.txt"
    found = detect_autonomous_decisions(
        tmp_path / "ws",
        "write",
        {"path": str(outside), "content": "x"},
        f"Written 1 bytes to {outside}",
        user_input="investigate the bug",
    )
    assert found == []


def test_relative_escape_target_is_never_recorded(tmp_path: Path) -> None:
    """``../other/checkout.py`` escapes the workspace, so it is not evidence."""
    found = detect_autonomous_decisions(
        tmp_path,
        "write",
        {"path": "../other/checkout.py", "content": "x"},
        "Written 1 bytes to ../other/checkout.py",
        user_input="investigate the bug",
    )
    assert found == []
