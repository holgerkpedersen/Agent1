"""Phase 5 of the autonomous-decision plan: the decision CLI surfaces.

The plan's last phase wires the decision system (Phases 1-4) into the REPL.
These tests drive the REAL :class:`DecideCommand` against a real temp
workspace - no mocks, no LLM, no network - and pin the promises the help text
makes:

- ``decide autonomous`` shows what the agent decided on its own.
- ``decide discrepancies`` shows recorded divergences, and filters them.
- ``decide improve`` is READ-ONLY: it reports, it never rewrites the ledger.
- ``decide apply`` keeps both records (revision appended, original marked
  superseded) and writes nothing unless the user confirms.
- ``--threshold`` is validated instead of silently defaulting.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_core.commands.base import set_autonomous
from agent_core.commands.decide_cmd import DecideCommand
from agent_core.decision_improvement import format_improvement_report
from agent_core.decisions import (
    STATUS_ACTIVE,
    STATUS_SUPERSEDED,
    add_decision,
    load_decisions,
    record_autonomous_decisions,
    record_discrepancy,
)


@pytest.fixture(autouse=True)
def _reset_autonomous():
    """No test may leak autonomous mode into another one."""
    set_autonomous(None)
    yield
    set_autonomous(None)


class _Agent:
    """Minimal stand-in for Agent: the command only needs ``workspace``."""

    def __init__(self, workspace):
        self.workspace = str(workspace)


def _run(cmd: DecideCommand, argv: list[str], agent: _Agent) -> bool:
    """Execute *argv* through the real dispatch path."""
    return asyncio.run(cmd.execute(argv, agent))


def _decision(ws, title, files, decision_text="", tags=None):
    """Add a real active decision and return its id."""
    return add_decision(
        ws,
        title=title,
        decision=decision_text,
        affected_files=files,
        tags=tags or [],
    )["id"]


def _file(ws, name):
    """Create a real file so the decision's surface resolves on disk."""
    path = ws / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x = 1\n", encoding="utf-8")
    return name


def _disc(decision_id, disc_type="blocked_action", severity="error", path=None):
    """One Phase 2 discrepancy record for *decision_id*."""
    return {
        "decision_id": decision_id,
        "decision_title": f"decision {decision_id}",
        "action_tool": "write_file",
        "action_args": {"path": path or f"agent_core/mod_{decision_id}.py"},
        "discrepancy_type": disc_type,
        "severity": severity,
        "suggestion": "do not touch this path",
    }


def _drifting_workspace(ws):
    """A decision violated once on its only file -> violation rate 1.0.

    ``total_actions`` is ``max(surface, violations)`` = max(1, 1) = 1, so the
    measured rate is 1.0 and clears every sane threshold.
    """
    _file(ws, "agent_core/a.py")
    decision_id = _decision(
        ws,
        "Keep writes in agent_core",
        ["agent_core/a.py"],
        decision_text="Never write outside agent_core",
    )
    record_discrepancy(ws, _disc(decision_id))
    return decision_id


def _snapshot(ws):
    """The ledger as a comparable value, for read-only assertions."""
    return [
        (d["id"], d["status"], d["title"], d.get("decision", ""))
        for d in load_decisions(ws)
    ]


# ── help / dispatch ────────────────────────────────────────────────────────


def test_help_text_documents_every_new_subcommand():
    help_text = DecideCommand().help_text
    for sub in ("autonomous", "discrepancies", "improve", "apply"):
        assert sub in help_text, f"{sub} missing from decide help"


def test_unknown_subcommand_still_falls_back_to_add(tmp_path, capsys):
    """A bare title must keep recording a decision (no regression)."""
    agent = _Agent(tmp_path)
    assert _run(DecideCommand(), ["A brand new decision"], agent) is True
    assert [d["title"] for d in load_decisions(tmp_path)] == ["A brand new decision"]


# ── decide improve: read-only ──────────────────────────────────────────────


def test_improve_reports_a_drifting_decision(tmp_path, capsys):
    decision_id = _drifting_workspace(tmp_path)
    before = _snapshot(tmp_path)

    assert _run(DecideCommand(), ["improve"], _Agent(tmp_path)) is True

    out = capsys.readouterr().out
    assert decision_id in out
    assert "Nothing was changed" in out
    assert "decide apply" in out
    # Read-only promise: the ledger is byte-for-byte equivalent.
    assert _snapshot(tmp_path) == before


def test_improve_threshold_excludes_a_partly_violated_decision(tmp_path, capsys):
    """A decision violated on 1 of 4 files has rate 0.25 — under a 0.5 default.

    Building the surface with four real files makes ``total_actions`` 4, so the
    threshold has something real to filter on instead of a degenerate 1.0.
    """
    files = [_file(tmp_path, f"agent_core/sub/f{i}.py") for i in range(4)]
    decision_id = _decision(tmp_path, "Partly violated", files)
    record_discrepancy(tmp_path, _disc(decision_id))
    before = _snapshot(tmp_path)

    # Default threshold 0.5 > 0.25: it must NOT be proposed.
    assert _run(DecideCommand(), ["improve"], _Agent(tmp_path)) is True
    assert "No decision needs rewording" in capsys.readouterr().out

    # Lowering the bar to 0.2 < 0.25 brings it back.
    assert _run(DecideCommand(), ["improve", "--threshold", "0.2"], _Agent(tmp_path))
    assert decision_id in capsys.readouterr().out
    assert _snapshot(tmp_path) == before


def test_improve_rejects_out_of_range_threshold(tmp_path, capsys):
    _drifting_workspace(tmp_path)
    before = _snapshot(tmp_path)

    assert _run(DecideCommand(), ["improve", "--threshold", "2"], _Agent(tmp_path))

    captured = capsys.readouterr()
    assert "between 0 and 1" in captured.out + captured.err
    assert _snapshot(tmp_path) == before


def test_improve_rejects_non_numeric_threshold(tmp_path, capsys):
    assert _run(DecideCommand(), ["improve", "--threshold", "loose"], _Agent(tmp_path))
    captured = capsys.readouterr()
    assert "needs a number" in captured.out + captured.err


def test_improve_on_a_clean_workspace_says_so(tmp_path, capsys):
    _decision(tmp_path, "No violations here", [_file(tmp_path, "agent_core/b.py")])

    assert _run(DecideCommand(), ["improve"], _Agent(tmp_path)) is True

    out = capsys.readouterr().out
    assert "No decision needs rewording" in out


# ── decide apply ───────────────────────────────────────────────────────────


def test_apply_dry_run_writes_nothing(tmp_path, capsys):
    decision_id = _drifting_workspace(tmp_path)
    before = _snapshot(tmp_path)

    assert _run(DecideCommand(), ["apply", "--dry-run"], _Agent(tmp_path)) is True

    out = capsys.readouterr().out
    assert "--dry-run" in out
    assert decision_id in out
    assert _snapshot(tmp_path) == before


def test_apply_declined_writes_nothing(tmp_path, capsys):
    _drifting_workspace(tmp_path)
    before = _snapshot(tmp_path)

    # No --yes: the confirmation is asked.  Autonomous mode answers the safe
    # default ("n") instead of blocking on stdin, which is exactly the
    # behaviour a headless run must have.
    set_autonomous(True)
    assert _run(DecideCommand(), ["apply"], _Agent(tmp_path)) is True

    out = capsys.readouterr().out
    assert _snapshot(tmp_path) == before
    assert "Cancelled" in out
    assert "superseded by a new active record" not in out


def test_apply_keeps_both_records(tmp_path, capsys):
    decision_id = _drifting_workspace(tmp_path)

    assert _run(DecideCommand(), ["apply", "--yes"], _Agent(tmp_path)) is True

    ledger = load_decisions(tmp_path)
    old = next(d for d in ledger if d["id"] == decision_id)
    # The audit trail survives: the original is superseded, not deleted, and
    # the revision is a NEW active record.
    assert old["status"] == STATUS_SUPERSEDED
    assert old.get("resolved_by")
    revision = next(d for d in ledger if d["id"] == old["resolved_by"])
    assert revision["status"] == STATUS_ACTIVE
    assert len(ledger) == 2

    out = capsys.readouterr().out
    assert "Applied 1 revision(s)" in out


def test_apply_with_nothing_to_do_is_a_no_op(tmp_path, capsys):
    _decision(tmp_path, "No violations here", [_file(tmp_path, "agent_core/c.py")])
    before = _snapshot(tmp_path)

    assert _run(DecideCommand(), ["apply", "--yes"], _Agent(tmp_path)) is True

    assert "No decision needs rewording" in capsys.readouterr().out
    assert _snapshot(tmp_path) == before


# ── decide discrepancies ───────────────────────────────────────────────────


def test_discrepancies_lists_recorded_records(tmp_path, capsys):
    decision_id = _drifting_workspace(tmp_path)

    assert _run(DecideCommand(), ["discrepancies"], _Agent(tmp_path)) is True

    out = capsys.readouterr().out
    assert f"#{decision_id}" in out
    assert "blocked_action" in out


def test_discrepancies_severity_filter_excludes(tmp_path, capsys):
    _drifting_workspace(tmp_path)  # severity "error"

    assert _run(
        DecideCommand(), ["discrepancies", "--severity", "critical"], _Agent(tmp_path)
    )

    assert "No discrepancies recorded" in capsys.readouterr().out


def test_discrepancies_on_empty_ledger_says_so(tmp_path, capsys):
    assert _run(DecideCommand(), ["discrepancies"], _Agent(tmp_path)) is True
    assert "No discrepancies recorded" in capsys.readouterr().out


# ── decide autonomous ──────────────────────────────────────────────────────


def test_autonomous_shows_recorded_entries(tmp_path, capsys):
    # The observation log is written by the plural recorder (Phase 1): it logs
    # every detection and promotes the new ones into the ledger.
    record_autonomous_decisions(
        tmp_path, [{"title": "Picked SQLite for storage", "tool_name": "write"}]
    )

    assert _run(DecideCommand(), ["autonomous"], _Agent(tmp_path)) is True

    out = capsys.readouterr().out
    assert "Picked SQLite for storage" in out
    assert "autonomous decision(s) recorded" in out


def test_autonomous_respects_limit(tmp_path, capsys):
    record_autonomous_decisions(
        tmp_path, [{"title": f"Decision {i}", "tool_name": "write"} for i in range(3)]
    )

    assert _run(DecideCommand(), ["autonomous", "--limit", "1"], _Agent(tmp_path))

    out = capsys.readouterr().out
    assert "Decision 2" in out          # newest entry is kept
    assert "Decision 0" not in out      # oldest is dropped


def test_autonomous_rejects_bad_limit(tmp_path, capsys):
    assert _run(DecideCommand(), ["autonomous", "--limit", "many"], _Agent(tmp_path))
    captured = capsys.readouterr()
    assert "needs a number" in captured.out + captured.err


def test_autonomous_on_empty_workspace_says_so(tmp_path, capsys):
    assert _run(DecideCommand(), ["autonomous"], _Agent(tmp_path)) is True
    assert "No autonomous decisions recorded" in capsys.readouterr().out


# ── the shared formatter ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "proposal_kwargs",
    [
        {},
        {"suggested_title": "New title"},
        {"suggested_decision": "Do it differently"},
        {"suggested_rationale": "Because it is violated"},
    ],
)
def test_format_report_handles_every_optional_field(proposal_kwargs):
    """Any subset of the optional fields must render without raising."""
    from agent_core.decision_improvement import DecisionUpdateProposal

    proposal = DecisionUpdateProposal(
        decision_id="001",
        confidence=0.75,
        evidence=[{"action_tool": "write", "discrepancy_type": "blocked_action"}],
        **proposal_kwargs,
    )
    report = format_improvement_report([proposal])
    assert "#001" in report
    assert "75%" in report
    assert "Run `decide apply`" in report


def test_format_report_of_nothing_is_stable_text():
    """The CLI and the REPL share this string, so it must not be empty."""
    assert format_improvement_report([]).startswith("No decision needs rewording")
