"""Phase 4.2: applying accepted proposals to the decision ledger.

These tests pin the four properties that make the write path safe:

* a REJECTED proposal leaves ``.decisions.json`` byte-identical;
* an ACCEPTED one appends a superseding record and closes the old one, so the
  original wording survives in the audit trail;
* applying twice creates no duplicate;
* a proposal naming a decision that does not exist is refused, not created.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_core.decision_improvement import (
    LABEL_APPLIED,
    LABEL_REJECTED,
    DecisionUpdateProposal,
    apply_decision_updates,
    review_proposals,
)
from agent_core.decisions import (
    STATUS_ACTIVE,
    STATUS_SUPERSEDED,
    load_decisions,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A workspace holding one active decision (#001) about ``src/auth.py``."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "auth.py").write_text("def login(): ...\n", encoding="utf-8")
    ledger = [
        {
            "id": "001",
            "date": "2026-01-01T00:00:00+00:00",
            "title": "Auth validates session tokens",
            "context": "Session handling",
            "decision": "All requests validate the session token in src/auth.py.",
            "rationale": "Unvalidated requests must not reach handlers.",
            "affected_files": ["src/auth.py"],
            "tags": ["security"],
            "contradictions": [],
            "resolved_by": None,
            "status": STATUS_ACTIVE,
            "category": "security",
        }
    ]
    (tmp_path / ".decisions.json").write_text(
        json.dumps(ledger, indent=2), encoding="utf-8"
    )
    return tmp_path


def _proposal(**overrides) -> DecisionUpdateProposal:
    base = dict(
        decision_id="001",
        suggested_title=None,
        suggested_decision="Token validation lives in src/auth.py::validate_session.",
        suggested_rationale="3 of 3 observed actions diverged from it (file_moved).",
        evidence=[{"discrepancy_type": "file_moved", "decision_id": "001"}],
        confidence=1.0,
    )
    base.update(overrides)
    return DecisionUpdateProposal(**base)


def _ledger_text(workspace: Path) -> str:
    return (workspace / ".decisions.json").read_text(encoding="utf-8")


# ── review_proposals: the accept/reject decision, kept separate from writing ─


def test_review_accepts_only_at_or_above_the_threshold(workspace: Path):
    accepted, rejected = review_proposals(
        [_proposal(confidence=0.9), _proposal(decision_id="002", confidence=0.3)]
    )
    assert [p.decision_id for p in accepted] == ["001"]
    assert [p.decision_id for p in rejected] == ["002"]


def test_review_threshold_is_inclusive(workspace: Path):
    accepted, rejected = review_proposals([_proposal(confidence=0.5)], threshold=0.5)
    assert len(accepted) == 1
    assert rejected == []


# ── apply_decision_updates ──────────────────────────────────────────────────


def test_rejected_proposal_leaves_the_ledger_byte_identical(workspace: Path):
    """The safety property: declining a suggestion must not touch the ledger."""
    before = _ledger_text(workspace)
    result = apply_decision_updates(
        workspace, [_proposal(confidence=0.2)], threshold=0.5
    )

    assert _ledger_text(workspace) == before
    assert result["applied"] == []
    assert [p.decision_id for p in result["rejected"]] == ["001"]


def test_accepted_proposal_appends_and_supersedes(workspace: Path):
    """The old wording survives as a superseded record; the new one is #002."""
    result = apply_decision_updates(workspace, [_proposal()], threshold=0.5)

    records = load_decisions(workspace)
    assert len(records) == 2
    old = next(r for r in records if r["id"] == "001")
    new = next(r for r in records if r["id"] == "002")

    assert old["status"] == STATUS_SUPERSEDED
    assert old["resolved_by"] == new["id"]
    assert new["status"] == STATUS_ACTIVE
    # The original wording is preserved verbatim — an audit trail, not a rewrite.
    assert new["decision"] == "Token validation lives in src/auth.py::validate_session."
    assert result["applied"] == ["001"]


def test_applied_proposal_is_labelled_with_its_evidence(workspace: Path):
    apply_decision_updates(workspace, [_proposal()], threshold=0.5)

    new = next(r for r in load_decisions(workspace) if r["id"] == "002")
    assert new["source"] == "improvement"
    assert new["review_label"] == LABEL_APPLIED
    assert len(new["evidence"]) == 1
    assert new["evidence"][0]["discrepancy_type"] == "file_moved"


def test_accepted_proposal_is_labelled_only_once_written(workspace: Path):
    """The label records what happened to the ledger, not that a gate was passed."""
    accepted, _ = review_proposals([_proposal()])
    assert accepted[0].review_label == LABEL_APPLIED

    apply_decision_updates(workspace, accepted, threshold=0.5)
    assert accepted[0].review_label == LABEL_APPLIED


def test_rejected_proposal_is_labelled(workspace: Path):
    _, rejected = review_proposals([_proposal(confidence=0.1)])

    assert [p.review_label for p in rejected] == [LABEL_REJECTED]


def test_applying_twice_creates_no_duplicate(workspace: Path):
    first = apply_decision_updates(workspace, [_proposal()], threshold=0.5)
    second = apply_decision_updates(workspace, [_proposal()], threshold=0.5)

    assert len(first["applied"]) == 1
    assert second["applied"] == []
    assert len(load_decisions(workspace)) == 2


def test_unknown_decision_id_is_refused_and_leaves_ledger_untouched(workspace: Path):
    before = _ledger_text(workspace)
    result = apply_decision_updates(
        workspace, [_proposal(decision_id="999")], threshold=0.5
    )

    assert result["applied"] == []
    assert [p.decision_id for p in result["rejected"]] == ["999"]
    assert _ledger_text(workspace) == before


def test_suggestions_are_applied_verbatim(workspace: Path):
    """A suggestion is stored as written — silently rewriting it would defeat review."""
    apply_decision_updates(
        workspace,
        [_proposal(suggested_title="Narrowed scope", suggested_rationale="Because X.")],
        threshold=0.5,
    )

    new = next(r for r in load_decisions(workspace) if r["id"] == "002")
    assert new["title"] == "Narrowed scope"
    assert new["rationale"] == "Because X."


def test_unsuggested_fields_inherit_from_the_original(workspace: Path):
    """Only the suggested fields change; context and tags carry over."""
    apply_decision_updates(
        workspace,
        [_proposal(suggested_title=None, suggested_rationale="Restated.")],
        threshold=0.5,
    )

    new = next(r for r in load_decisions(workspace) if r["id"] == "002")
    assert new["title"] == "Auth validates session tokens"
    assert new["context"] == "Session handling"
    assert new["tags"] == ["security"]
    assert new["affected_files"] == ["src/auth.py"]


def test_a_repeated_suggestion_is_declined_not_duplicated(workspace: Path):
    """The same suggestion applied twice must not produce two records."""
    apply_decision_updates(workspace, [_proposal()], threshold=0.5)

    second = apply_decision_updates(workspace, [_proposal()], threshold=0.5)

    assert second["applied"] == []
    assert [p.decision_id for p in second["rejected"]] == ["001"]
    assert len(load_decisions(workspace)) == 2


def test_a_second_revision_targets_the_record_the_first_created(workspace: Path):
    """A revision is written against the live record, not the closed original.

    Once #001 is superseded, drift on it is a closure problem (RECOMMEND_RESOLVE),
    so a proposal naming #001 is declined.  The next wording change targets #002 —
    the active record the first application produced.
    """
    apply_decision_updates(workspace, [_proposal()], threshold=0.5)

    second = apply_decision_updates(
        workspace,
        [_proposal(decision_id="002", suggested_rationale="Narrowed to login only.")],
        threshold=0.5,
    )

    assert second["applied"] == ["002"]
    records = load_decisions(workspace)
    assert len(records) == 3
    assert next(r for r in records if r["id"] == "002")["status"] == STATUS_SUPERSEDED
    assert next(r for r in records if r["id"] == "003")["status"] == STATUS_ACTIVE


def test_a_proposal_naming_a_superseded_decision_is_declined(workspace: Path):
    """The closed original is not reworded — that is what makes re-apply a no-op."""
    apply_decision_updates(workspace, [_proposal()], threshold=0.5)

    second = apply_decision_updates(
        workspace,
        [_proposal(suggested_rationale="Narrowed to login only.")],
        threshold=0.5,
    )

    assert second["applied"] == []
    assert [p.decision_id for p in second["rejected"]] == ["001"]
    assert len(load_decisions(workspace)) == 2


def test_empty_proposal_list_is_a_no_op(workspace: Path):
    before = _ledger_text(workspace)
    result = apply_decision_updates(workspace, [], threshold=0.5)

    assert result == {"applied": [], "rejected": []}
    assert _ledger_text(workspace) == before


def test_missing_ledger_yields_no_proposals_and_writes_nothing(tmp_path: Path):
    """No ledger means nothing to improve; the call must not create one."""
    result = apply_decision_updates(tmp_path, [_proposal()], threshold=0.5)

    assert result["applied"] == []
    assert [p.decision_id for p in result["rejected"]] == ["001"]
    assert not (tmp_path / ".decisions.json").exists()
