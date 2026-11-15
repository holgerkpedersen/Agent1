"""Phase 3 of the autonomous-decision plan: decision collection.

Card ``9bc4327f`` ("Improve Autonomous Decisions in Agent1"), plan
``.decisions_plan.md`` — Phase 3 aggregates what Phase 1 detected and Phase 2
flagged into one review-ready picture:

- :meth:`DecisionCollector.collect_turn_decisions` extracts the autonomous
  decisions a conversation turn made, from the same detector Phase 1 uses.
  It is a *reader*: it must never promote anything into the ledger, because
  the tool hook already owns promotion.
- :meth:`DecisionCollector.aggregate_periodic` folds the observation log, the
  discrepancy ledger and anything passed to ``collect_turn_decisions`` into
  counts grouped by tag and by discrepancy type/severity — the input the
  Phase 4 improvement pipeline consumes.

These tests drive the REAL modules against a temp workspace, and the real
detector (no LLM, no network, no mocks).
"""

from __future__ import annotations

import json

from agent_core.decisions import (
    AUTONOMOUS_LOG_FILE,
    DISCREPANCIES_FILE,
    load_decisions,
    record_discrepancy,
)
from agent_core.decision_collector import DecisionCollector
from agent_core.discrepancy import Discrepancy


def _append_log(workspace, entry: dict) -> None:
    """Append one raw record to the Phase 1 observation log."""
    with (workspace / AUTONOMOUS_LOG_FILE).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _obs(title: str, tags: list[str], decision_id=None, detected_at=None) -> dict:
    """One Phase 1 observation-log record."""
    entry: dict = {"title": title, "tags": tags, "decision_id": decision_id}
    if detected_at is not None:
        entry["detected_at"] = detected_at
    return entry


def _user_turn(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


# ── collect_turn_decisions ────────────────────────────────────────────────


def test_collect_turn_decisions_extracts_decisions_from_tool_calls(
    tmp_path,
) -> None:
    """A write to a path the user never named is an autonomous decision."""
    collector = DecisionCollector(tmp_path)

    found = collector.collect_turn_decisions(
        turn_messages=_user_turn("Add a helper for dates."),
        tools_called=[("write", {"path": "agent_core/dates.py"})],
        discrepancies=[],
    )

    assert [c["title"] for c in found] == ["Change agent_core/dates.py"]


def test_collect_turn_decisions_skips_read_only_calls(tmp_path) -> None:
    """Inspection is a judgement of fact, not a policy decision."""
    collector = DecisionCollector(tmp_path)

    found = collector.collect_turn_decisions(
        turn_messages=_user_turn("What does this file do?"),
        tools_called=[("read", {"path": "agent_core/agent.py"})],
        discrepancies=[],
    )

    assert found == []


def test_collect_turn_decisions_excludes_paths_the_user_named(
    tmp_path,
) -> None:
    """A path in the request is the user's instruction, not the agent's call."""
    collector = DecisionCollector(tmp_path)

    found = collector.collect_turn_decisions(
        turn_messages=_user_turn("Please edit agent_core/dates.py."),
        tools_called=[("write", {"path": "agent_core/dates.py"})],
        discrepancies=[],
    )

    assert found == []


def test_collect_turn_decisions_never_promotes_to_the_ledger(tmp_path) -> None:
    """The collector reads; the Phase 1 tool hook owns ledger promotion."""
    collector = DecisionCollector(tmp_path)

    collector.collect_turn_decisions(
        turn_messages=_user_turn("Add a helper for dates."),
        tools_called=[("write", {"path": "agent_core/dates.py"})],
        discrepancies=[],
    )

    assert load_decisions(tmp_path) == []


def test_collect_turn_decisions_ignores_refused_tool_calls(tmp_path) -> None:
    """A call the guard refused decided nothing."""
    collector = DecisionCollector(tmp_path)

    refused = ("write", {"path": "agent_core/dates.py"}, "blocked: not allowed")
    found = collector.collect_turn_decisions(
        turn_messages=_user_turn("Add a helper for dates."),
        tools_called=[refused],
        discrepancies=[],
    )

    assert found == []


# ── aggregate_periodic ───────────────────────────────────────────────────


def test_aggregate_periodic_counts_observed_decisions_by_tag(tmp_path) -> None:
    """The log is grouped by tag, with counts that add up."""
    _append_log(tmp_path, _obs("Change a.py", ["autonomous", "write"], 1))
    _append_log(tmp_path, _obs("Change b.py", ["autonomous", "write"], 2))
    _append_log(tmp_path, _obs("Ran git commit", ["autonomous", "git"], 3))

    summary = DecisionCollector(tmp_path).aggregate_periodic()

    assert summary["total"] == 3
    assert summary["by_tag"]["autonomous"] == 3
    assert summary["by_tag"]["write"] == 2
    assert summary["by_tag"]["git"] == 1


def test_aggregate_periodic_separates_promoted_from_unpromoted(tmp_path) -> None:
    """A detected-but-never-promoted decision is the Phase 4 review target."""
    _append_log(tmp_path, _obs("Change a.py", ["autonomous"], 1))
    _append_log(tmp_path, _obs("Change b.py", ["autonomous"], None))

    summary = DecisionCollector(tmp_path).aggregate_periodic()

    assert summary["promoted"] == 1
    assert summary["unpromoted"] == 1


def test_aggregate_periodic_counts_discrepancies_by_type_and_severity(
    tmp_path,
) -> None:
    """Both axes of a discrepancy are counted, not just the total."""
    record_discrepancy(
        tmp_path,
        Discrepancy(
            decision_id="1",
            decision_title="Guard",
            action_tool="write",
            action_args={},
            discrepancy_type="blocked_action",
            severity="error",
        ),
    )
    record_discrepancy(
        tmp_path,
        Discrepancy(
            decision_id="2",
            decision_title="Guard",
            action_tool="write",
            action_args={},
            discrepancy_type="blocked_action",
            severity="warning",
        ),
    )

    summary = DecisionCollector(tmp_path).aggregate_periodic()

    assert summary["discrepancies"]["total"] == 2
    assert summary["discrepancies"]["by_type"]["blocked_action"] == 2
    assert summary["discrepancies"]["by_severity"]["error"] == 1
    assert summary["discrepancies"]["by_severity"]["warning"] == 1


def test_aggregate_periodic_includes_discrepancies_collected_in_turn(
    tmp_path,
) -> None:
    """Discrepancies handed to collect_turn_decisions reach the summary."""
    collector = DecisionCollector(tmp_path)
    collector.collect_turn_decisions(
        turn_messages=_user_turn("Add a helper for dates."),
        tools_called=[],
        discrepancies=[
            Discrepancy(
                decision_id="7",
                decision_title="No deletions",
                action_tool="run",
                action_args={"command": "del a.py"},
                discrepancy_type="contradiction",
                severity="critical",
            )
        ],
    )

    summary = collector.aggregate_periodic()
    assert summary["discrepancies"]["total"] == 1
    assert summary["discrepancies"]["by_type"]["contradiction"] == 1
    assert summary["discrepancies"]["by_severity"]["critical"] == 1


def test_aggregate_periodic_filters_by_since(tmp_path) -> None:
    """``since`` selects the review window; older observations are excluded."""
    old = "2020-01-01T00:00:00+00:00"
    new = "2030-01-01T00:00:00+00:00"
    _append_log(tmp_path, _obs("Old", ["autonomous"], None, old))
    _append_log(tmp_path, _obs("New", ["autonomous"], None, new))

    summary = DecisionCollector(tmp_path).aggregate_periodic(
        since="2025-01-01T00:00:00+00:00"
    )

    assert summary["total"] == 1


def test_aggregate_periodic_on_an_empty_workspace_is_empty(tmp_path) -> None:
    """No log and no ledger is a valid answer, not an error."""
    summary = DecisionCollector(tmp_path).aggregate_periodic()

    assert summary["total"] == 0
    assert summary["by_tag"] == {}
    assert summary["discrepancies"]["total"] == 0


def test_aggregate_periodic_skips_a_corrupt_log_line(tmp_path) -> None:
    """A truncated append must not fail the review."""
    (tmp_path / AUTONOMOUS_LOG_FILE).write_text(
        '{"title": "Change a.py", "tags": ["autonomous"]}\nnot json\n',
        encoding="utf-8",
    )

    summary = DecisionCollector(tmp_path).aggregate_periodic()

    assert summary["total"] == 1


def test_collector_workspace_is_resolved(tmp_path) -> None:
    """The collector resolves its workspace, so relative paths behave."""
    collector = DecisionCollector(str(tmp_path))

    assert collector.workspace == tmp_path.resolve()


def test_aggregate_periodic_reads_the_discrepancy_ledger_file(tmp_path) -> None:
    """The file the Phase 2 hook writes is the file the collector reads."""
    record_discrepancy(
        tmp_path,
        {
            "decision_id": "3",
            "discrepancy_type": "unresolved_context",
            "severity": "warning",
        },
    )

    assert (tmp_path / DISCREPANCIES_FILE).exists()
    summary = DecisionCollector(tmp_path).aggregate_periodic()
    assert summary["discrepancies"]["by_type"]["unresolved_context"] == 1
