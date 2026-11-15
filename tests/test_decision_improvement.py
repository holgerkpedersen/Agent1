"""Phase 4.1 of the autonomous-decision plan: the improvement pipeline.

Card ``9bc4327f`` ("Improve Autonomous Decisions in Agent1"), plan
``.decisions_plan.md`` — Phase 4 turns what Phase 3 collected into proposals
that can update ``.decisions.json``:

- :func:`detect_drift` answers "is this decision being violated repeatedly?"
  from the Phase 2 discrepancy ledger.
- :func:`propose_decision_updates` turns a drifting decision into a
  :class:`DecisionUpdateProposal` carrying evidence and a confidence.

The pipeline only ever *proposes*: nothing here writes to the ledger, so a
wrong suggestion can never silently rewrite an audit trail.

``total_actions`` is the size of the decision's own matching surface (its
``affected_files`` that still exist), not a fabricated action count: the
denominator must be something a reader can verify from the decision record.

These tests drive the REAL modules against a temp workspace (no LLM, no
network, no mocks).
"""

from __future__ import annotations

import pytest

from agent_core.decision_collector import DecisionCollector
from agent_core.decision_improvement import (
    DecisionUpdateProposal,
    DriftAnalysis,
    detect_drift,
    propose_decision_updates,
)
from agent_core.decisions import (
    STATUS_ACTIVE,
    STATUS_SUPERSEDED,
    add_decision,
    load_decisions,
    record_discrepancy,
    save_decisions,
)


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


def _decision(ws, title, files, decision_text="", tags=None):
    """Add a real active decision and return its id."""
    return add_decision(
        ws,
        title=title,
        decision=decision_text,
        affected_files=files,
        tags=tags or [],
    )["id"]


def _files(ws, names):
    """Create real files so the decision's surface resolves on disk."""
    for name in names:
        path = ws / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x = 1\n", encoding="utf-8")
    return names


def _close(ws, decision_id, status=STATUS_SUPERSEDED):
    """Flip one decision's status in the real ledger."""
    decisions = load_decisions(ws)
    for record in decisions:
        if record["id"] == decision_id:
            record["status"] = status
    save_decisions(ws, decisions)


# ── detect_drift: counting ──────────────────────────────────────────────────


def test_detect_drift_counts_only_the_decisions_own_violations(tmp_path):
    ws = tmp_path
    decision_id = _decision(ws, "Keep writes in agent_core", ["agent_core/a.py"])
    for _ in range(2):
        record_discrepancy(ws, _disc(decision_id))
    for _ in range(3):
        record_discrepancy(ws, _disc("999"))

    from agent_core.decisions import load_discrepancies

    analysis = detect_drift(
        ws, decision_id, load_discrepancies(ws, decision_id=decision_id)
    )

    assert isinstance(analysis, DriftAnalysis)
    assert analysis.decision_id == decision_id
    assert analysis.violation_count == 2


def test_detect_drift_matches_an_int_id_against_a_string_id(tmp_path):
    ws = tmp_path
    record_discrepancy(ws, _disc(12))

    analysis = detect_drift(ws, "12", [_disc(12)])

    assert analysis.violation_count == 1


def test_detect_drift_with_no_violations_is_quiet(tmp_path):
    ws = tmp_path
    decision_id = _decision(ws, "Keep writes in agent_core", ["agent_core/a.py"])

    analysis = detect_drift(ws, decision_id, [])

    assert analysis.violation_count == 0
    assert analysis.violation_rate == 0.0
    assert analysis.common_patterns == []
    assert analysis.recommendation == "none"


def test_detect_drift_rate_never_exceeds_one(tmp_path):
    ws = tmp_path
    decision_id = _decision(ws, "Keep writes in agent_core", ["agent_core/a.py"])
    records = [_disc(decision_id) for _ in range(5)]
    for record in records:
        record_discrepancy(ws, record)

    analysis = detect_drift(ws, decision_id, records)

    assert analysis.violation_count == 5
    assert analysis.total_actions >= analysis.violation_count
    assert 0.0 < analysis.violation_rate <= 1.0


def test_detect_drift_survives_malformed_records(tmp_path):
    ws = tmp_path
    decision_id = _decision(ws, "Keep writes in agent_core", ["agent_core/a.py"])
    malformed = [
        {"decision_id": decision_id},
        {"decision_id": decision_id, "action_args": "not-a-dict"},
        {"decision_id": decision_id, "discrepancy_type": None},
        {},
        None,
    ]

    analysis = detect_drift(ws, decision_id, malformed)

    assert analysis.violation_count == 3
    assert isinstance(analysis.common_patterns, list)


# ── detect_drift: the denominator is the real surface ──────────────────────


def test_total_actions_counts_decision_files_that_exist(tmp_path):
    ws = tmp_path
    _files(ws, ["agent_core/real_mod.py"])
    decision_id = _decision(ws, "Keep writes in agent_core", ["agent_core/real_mod.py"])

    analysis = detect_drift(ws, decision_id, [])

    assert analysis.total_actions >= 1


def test_total_actions_skips_a_deleted_decision_file(tmp_path):
    ws = tmp_path
    decision_id = _decision(ws, "Keep writes in agent_core", ["agent_core/gone.py"])

    analysis = detect_drift(ws, decision_id, [])

    assert analysis.total_actions == 0


# ── detect_drift: recommendations ──────────────────────────────────────────


def test_repeated_blocked_action_recommends_update(tmp_path):
    ws = tmp_path
    decision_id = _decision(ws, "Keep writes in agent_core", ["agent_core/a.py"])
    records = [_disc(decision_id) for _ in range(4)]

    analysis = detect_drift(ws, decision_id, records)

    assert analysis.recommendation == "update"
    assert "blocked_action" in analysis.common_patterns


def test_mixed_violation_types_recommend_investigate(tmp_path):
    ws = tmp_path
    decision_id = _decision(ws, "Keep writes in agent_core", ["agent_core/a.py"])
    records = [
        _disc(decision_id, "blocked_action"),
        _disc(decision_id, "blocked_action"),
        _disc(decision_id, "contradiction"),
        _disc(decision_id, "contradiction"),
        _disc(decision_id, "unresolved_context"),
        _disc(decision_id, "unresolved_context"),
    ]

    analysis = detect_drift(ws, decision_id, records)

    assert analysis.recommendation == "investigate"
    assert len(analysis.common_patterns) >= 2


def test_a_closed_decision_recommends_resolve(tmp_path):
    ws = tmp_path
    decision_id = _decision(ws, "Keep writes in agent_core", ["agent_core/a.py"])
    record_discrepancy(ws, _disc(decision_id))
    _close(ws, decision_id)

    analysis = detect_drift(ws, decision_id, [_disc(decision_id)])

    assert analysis.recommendation == "resolve"


# ── propose_decision_updates: gating ───────────────────────────────────────


def test_empty_workspace_proposes_nothing(tmp_path):
    assert propose_decision_updates(tmp_path, DecisionCollector(tmp_path)) == []


def test_a_single_violation_below_threshold_proposes_nothing(tmp_path):
    ws = tmp_path
    _files(ws, [f"agent_core/m{i}.py" for i in range(10)])
    decision_id = _decision(
        ws, "Keep writes in agent_core", [f"agent_core/m{i}.py" for i in range(10)]
    )
    record_discrepancy(ws, _disc(decision_id))

    proposals = propose_decision_updates(ws, DecisionCollector(ws), threshold=0.5)

    assert proposals == []


def test_a_drifting_decision_is_proposed_with_evidence(tmp_path):
    ws = tmp_path
    files = _files(ws, [f"agent_core/m{i}.py" for i in range(4)])
    decision_id = _decision(ws, "Keep writes in agent_core", files)
    for _ in range(4):
        record_discrepancy(ws, _disc(decision_id))

    proposals = propose_decision_updates(ws, DecisionCollector(ws), threshold=0.5)

    assert len(proposals) == 1
    proposal = proposals[0]
    assert isinstance(proposal, DecisionUpdateProposal)
    assert proposal.decision_id == decision_id
    assert proposal.evidence
    assert 0.0 <= proposal.confidence <= 1.0
    assert proposal.confidence >= 0.5


def test_evidence_is_json_serialisable(tmp_path):
    import json

    ws = tmp_path
    decision_id = _decision(ws, "Keep writes in agent_core", ["agent_core/a.py"])
    for _ in range(4):
        record_discrepancy(ws, _disc(decision_id))

    proposals = propose_decision_updates(ws, DecisionCollector(ws))

    json.dumps(proposals[0].evidence)


def test_the_collector_argument_is_optional(tmp_path):
    ws = tmp_path
    decision_id = _decision(ws, "Keep writes in agent_core", ["agent_core/a.py"])
    for _ in range(4):
        record_discrepancy(ws, _disc(decision_id))

    assert len(propose_decision_updates(ws)) == 1


def test_a_discrepancy_for_an_unknown_decision_is_not_proposed(tmp_path):
    ws = tmp_path
    _decision(ws, "Keep writes in agent_core", ["agent_core/a.py"])
    for _ in range(4):
        record_discrepancy(ws, _disc("999"))

    assert propose_decision_updates(ws, DecisionCollector(ws)) == []


def test_a_superseded_decision_is_reported_but_not_edited(tmp_path):
    ws = tmp_path
    decision_id = _decision(ws, "Keep writes in agent_core", ["agent_core/a.py"])
    for _ in range(4):
        record_discrepancy(ws, _disc(decision_id))
    _close(ws, decision_id, STATUS_SUPERSEDED)

    proposals = propose_decision_updates(ws, DecisionCollector(ws))

    assert proposals == []


# ── propose_decision_updates: what a proposal suggests ─────────────────────


def test_an_update_proposal_suggests_new_prose(tmp_path):
    ws = tmp_path
    decision_id = _decision(
        ws,
        "Keep writes in agent_core",
        ["agent_core/a.py"],
        decision_text="Only write under agent_core/.",
    )
    for _ in range(4):
        record_discrepancy(ws, _disc(decision_id))

    proposal = propose_decision_updates(ws, DecisionCollector(ws))[0]

    assert proposal.suggested_title is None or proposal.suggested_title.strip()
    assert isinstance(proposal.suggested_decision, str)
    assert proposal.suggested_decision.strip()
    assert "discrepanc" in (proposal.suggested_rationale or "").lower()


def test_an_investigate_proposal_suggests_no_new_wording(tmp_path):
    ws = tmp_path
    decision_id = _decision(
        ws,
        "Keep writes in agent_core",
        ["agent_core/a.py"],
        decision_text="Only write under agent_core/.",
    )
    for disc_type in ("blocked_action", "contradiction", "unresolved_context"):
        record_discrepancy(ws, _disc(decision_id, disc_type))
        record_discrepancy(ws, _disc(decision_id, disc_type))

    proposal = propose_decision_updates(ws, DecisionCollector(ws))[0]

    assert proposal.suggested_decision is None
    assert "investigat" in (proposal.suggested_rationale or "").lower()


def test_proposals_are_ordered_by_confidence(tmp_path):
    ws = tmp_path
    names = [f"agent_core/s{i}.py" for i in range(5)]
    _files(ws, names)
    mostly = _decision(ws, "Mostly violated", names[:5])
    partly = _decision(ws, "Partly violated", names[:5])
    for _ in range(5):
        record_discrepancy(ws, _disc(mostly))
    for _ in range(3):
        record_discrepancy(ws, _disc(partly))

    proposals = propose_decision_updates(ws, DecisionCollector(ws), threshold=0.5)

    assert [p.decision_id for p in proposals] == [mostly, partly]
    assert proposals[0].confidence >= proposals[1].confidence


def test_an_active_decision_is_left_active(tmp_path):
    ws = tmp_path
    decision_id = _decision(ws, "Keep writes in agent_core", ["agent_core/a.py"])
    for _ in range(4):
        record_discrepancy(ws, _disc(decision_id))

    propose_decision_updates(ws, DecisionCollector(ws))

    statuses = {d["id"]: d["status"] for d in load_decisions(ws)}
    assert statuses[decision_id] == STATUS_ACTIVE


def test_detect_drift_is_pure(tmp_path):
    ws = tmp_path
    decision_id = _decision(ws, "Keep writes in agent_core", ["agent_core/a.py"])
    record_discrepancy(ws, _disc(decision_id))
    before = (ws / ".decisions.json").read_text(encoding="utf-8")

    detect_drift(ws, decision_id, [_disc(decision_id)])
    detect_drift(ws, decision_id, [_disc(decision_id)])

    assert (ws / ".decisions.json").read_text(encoding="utf-8") == before


@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
def test_threshold_is_honoured(tmp_path, threshold):
    ws = tmp_path
    names = [f"agent_core/t{i}.py" for i in range(4)]
    _files(ws, names)
    decision_id = _decision(ws, "Partly violated", names)
    for _ in range(2):  # rate 2/4 == 0.5
        record_discrepancy(ws, _disc(decision_id))

    proposals = propose_decision_updates(ws, DecisionCollector(ws), threshold=threshold)

    assert len(proposals) == (1 if threshold <= 0.5 else 0)
