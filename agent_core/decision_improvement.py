"""Phase 4 of the autonomous-decision plan: decide when a decision has drifted.

Card ``9bc4327f`` ("Improve Autonomous Decisions in Agent1"), plan
``.decisions_plan.md`` — Phase 3 collects what Phase 1 detected and Phase 2
flagged.  This module is the only consumer of that picture: it answers "is this
decision being violated repeatedly?" and, when the answer is yes, turns it into
a :class:`DecisionUpdateProposal`.

Two rules shape the design:

* **Nothing here writes.**  A proposal is a suggestion, never an edit, so a
  wrong guess cannot rewrite an audit trail.  Applying proposals is Phase 4.2.
* **The denominator is real.**  ``total_actions`` is the size of the decision's
  own matching surface — its ``affected_files`` that still exist — raised to at
  least the violations recorded against it, because a violation is itself an
  action against that decision.  Nothing is estimated: a reader can verify the
  number from the decision record and the discrepancy ledger.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_core.decision_collector import DecisionCollector
from agent_core.decisions import (
    STATUS_ARCHIVED,
    STATUS_ACTIVE,
    STATUS_SUPERSEDED,
    STATUS_VOID,
    _next_id,
    load_decisions,
    load_discrepancies,
    normalize_affected_files,
    save_decisions,
)

logger = logging.getLogger(__name__)

#: Recommendations :func:`detect_drift` can return.
RECOMMEND_NONE = "none"
RECOMMEND_UPDATE = "update"
RECOMMEND_RESOLVE = "resolve"
RECOMMEND_INVESTIGATE = "investigate"

#: A decision in one of these states is already closed; drift against it means
#: the closure did not stick, not that the wording needs a new revision.
_CLOSED_STATUSES = {STATUS_SUPERSEDED, STATUS_ARCHIVED, STATUS_VOID}

#: Review outcomes for a proposal (Phase 4.2).  A label is recorded on the
#: proposal so the accept/reject decision is auditable next to its evidence.
LABEL_APPLIED = "applied"
LABEL_REJECTED = "rejected"

#: ``source`` of a ledger record created by :func:`apply_decision_updates`.
#: Distinguishes an improvement revision from a hand-written decision and from
#: one the agent took autonomously (``source="agent"``).
SOURCE_IMPROVEMENT = "improvement"

#: Most disagreement a single proposal carries into its evidence list.
_MAX_EVIDENCE = 20


@dataclass
class DriftAnalysis:
    """How one decision is being violated, and what to do about it."""

    decision_id: str
    violation_count: int
    total_actions: int
    violation_rate: float
    common_patterns: list[str] = field(default_factory=list)
    recommendation: str = RECOMMEND_NONE


@dataclass
class DecisionUpdateProposal:
    """A suggested change to one decision, with the evidence behind it.

    ``confidence`` is the violation rate the proposal rests on, so it always
    equals the ``DriftAnalysis.violation_rate`` that produced it — it is a
    measured proportion, not a judgement.

    ``review_label`` is ``None`` until the proposal has been through
    :func:`review_proposals`: applying sets :data:`LABEL_APPLIED`, declining
    sets :data:`LABEL_REJECTED`.
    """

    decision_id: str
    suggested_title: str | None = None
    suggested_decision: str | None = None
    suggested_rationale: str | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    review_label: str | None = None


# ── Reading the ledgers ─────────────────────────────────────────────────────


def _decision_index(workspace: str | Path) -> dict[str, dict[str, Any]]:
    """Map decision id (as ``str``) to its ledger record."""
    index: dict[str, dict[str, Any]] = {}
    for record in load_decisions(workspace):
        key = str(record.get("id") or "").strip()
        if key:
            index[key] = record
    return index


def _own_violations(
    decision_id: str, discrepancies: list[Any] | None
) -> list[dict[str, Any]]:
    """Discrepancies recorded against *decision_id*, newest last.

    Ids are compared as text: the ledger stores ``"001"`` while a
    :class:`agent_core.discrepancy.Discrepancy` may carry ``1``.  A record
    without a ``decision_id`` belongs to no decision and is skipped; a record
    whose remaining fields are malformed is still kept, because the fact that a
    violation happened is the signal, and its detail is only decoration.
    """
    key = str(decision_id).strip()
    return [
        entry
        for entry in discrepancies or []
        if isinstance(entry, dict)
        and str(entry.get("decision_id") or "").strip() == key
    ]


def _surface_size(workspace: str | Path, decision: dict[str, Any]) -> int:
    """How many of the decision's files still exist on disk."""
    root = Path(workspace)
    count = 0
    for rel in decision.get("affected_files") or []:
        try:
            if (root / str(rel)).exists():
                count += 1
        except OSError:  # pragma: no cover — unreadable path component
            logger.debug("could not stat decision surface %r", rel, exc_info=True)
    return count


def _patterns(violations: list[dict[str, Any]]) -> list[str]:
    """Distinct discrepancy types, most frequent first."""
    counts: dict[str, int] = {}
    for entry in violations:
        kind = str(entry.get("discrepancy_type") or "unknown").strip() or "unknown"
        counts[kind] = counts.get(kind, 0) + 1
    return [kind for kind, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


# ── Drift analysis ──────────────────────────────────────────────────────────


def detect_drift(
    workspace: str | Path,
    decision_id: str,
    discrepancies: list[dict[str, Any]],
) -> DriftAnalysis:
    """Analyze whether a decision is being violated repeatedly.

    Returns the violation frequency, the common violation patterns and a
    recommendation:

    - ``"none"`` — nothing recorded against this decision.
    - ``"resolve"`` — the decision is already closed, so the violations mean the
      closure did not take effect.
    - ``"investigate"`` — several different kinds of violation, so the cause is
      not the wording and rewriting it would guess.
    - ``"update"`` — one recurring violation kind: the wording is the problem.

    Read-only: the ledgers are not touched.
    """
    violations = _own_violations(decision_id, discrepancies)
    decision = _decision_index(workspace).get(str(decision_id).strip(), {})
    surface = _surface_size(workspace, decision)
    # A violation is itself an action against the decision, so the observed
    # count can never be smaller than the violations we hold evidence for.
    total_actions = max(surface, len(violations))
    violation_rate = len(violations) / total_actions if total_actions else 0.0
    patterns = _patterns(violations)

    if not violations:
        recommendation = RECOMMEND_NONE
    elif str(decision.get("status") or STATUS_ACTIVE) in _CLOSED_STATUSES:
        recommendation = RECOMMEND_RESOLVE
    elif len(patterns) > 1:
        recommendation = RECOMMEND_INVESTIGATE
    else:
        recommendation = RECOMMEND_UPDATE

    return DriftAnalysis(
        decision_id=str(decision_id).strip(),
        violation_count=len(violations),
        total_actions=total_actions,
        violation_rate=round(violation_rate, 4),
        common_patterns=patterns,
        recommendation=recommendation,
    )


# ── Proposals ───────────────────────────────────────────────────────────────


def _update_proposal(
    decision: dict[str, Any], analysis: DriftAnalysis
) -> DecisionUpdateProposal:
    """Proposal for a decision whose wording is repeatedly violated."""
    patterns = ", ".join(analysis.common_patterns) or "unknown"
    surface = [str(p) for p in decision.get("affected_files") or []]
    scope = (
        f"It names {len(surface)} file(s): {', '.join(surface[:5])}"
        if surface
        else "It names no file, so its scope is ambiguous"
    )
    return DecisionUpdateProposal(
        decision_id=analysis.decision_id,
        suggested_title=None,
        suggested_decision=(
            f"Re-affirm or narrow this decision before relying on it: "
            f"{analysis.violation_count} of {analysis.total_actions} observed "
            f"action(s) diverged from it ({patterns}). {scope}."
        ),
        suggested_rationale=(
            f"{analysis.violation_count} discrepancies of type {patterns} were "
            f"recorded against this decision "
            f"(violation rate {analysis.violation_rate:.0%}). A recurring single "
            f"pattern usually means the decision no longer matches the code it "
            f"describes, so restating it is safer than leaving it to be ignored."
        ),
        evidence=[],
        confidence=analysis.violation_rate,
    )


def _investigate_proposal(
    decision: dict[str, Any], analysis: DriftAnalysis
) -> DecisionUpdateProposal:
    """Proposal that asks for investigation instead of new wording."""
    patterns = ", ".join(analysis.common_patterns)
    return DecisionUpdateProposal(
        decision_id=analysis.decision_id,
        suggested_title=None,
        suggested_decision=None,
        suggested_rationale=(
            f"{analysis.violation_count} discrepancies of "
            f"{len(analysis.common_patterns)} different kinds were recorded "
            f"against this decision ({patterns}). "
            f"Divergence on several fronts is not fixed by rewording it: investigate "
            f"the root cause first, then update or resolve the decision."
        ),
        evidence=[],
        confidence=analysis.violation_rate,
    )


def propose_decision_updates(
    workspace: str | Path,
    collector: DecisionCollector | None = None,
    threshold: float = 0.5,
) -> list[DecisionUpdateProposal]:
    """Propose updates to decisions based on collected data.

    A proposal is emitted for an **active** decision whose violation rate
    reaches *threshold*; a closed decision is never proposed, because its drift
    is a closure problem rather than a wording problem (see
    :func:`detect_drift`).

    *collector* is the Phase 3 object the caller already holds — passing it
    reuses its workspace instead of building a second one.  Proposals come back
    strongest first, ordered by the violation rate they rest on.  The ledger is
    only read.
    """
    base = (
        Path(collector.workspace)
        if collector is not None
        else Path(workspace).resolve()
    )
    index = _decision_index(base)
    if not index:
        return []
    discrepancies = load_discrepancies(base)
    if not discrepancies:
        return []

    proposals: list[DecisionUpdateProposal] = []
    for decision_id, decision in index.items():
        analysis = detect_drift(base, decision_id, discrepancies)
        if analysis.recommendation not in (RECOMMEND_UPDATE, RECOMMEND_INVESTIGATE):
            continue
        if analysis.violation_rate < threshold:
            continue
        if analysis.recommendation == RECOMMEND_UPDATE:
            proposal = _update_proposal(decision, analysis)
        else:
            proposal = _investigate_proposal(decision, analysis)
        proposal.evidence = _own_violations(decision_id, discrepancies)[-_MAX_EVIDENCE:]
        proposals.append(proposal)

    proposals.sort(key=lambda p: p.confidence, reverse=True)
    return proposals


# ── Review and application (Phase 4.2) ──────────────────────────────────────

#: Title of a record written for a proposal that suggests no new title.
_REWRITE_TITLE = "Decision restated after repeated divergence"


def review_proposals(
    proposals: list[DecisionUpdateProposal],
    threshold: float = 0.5,
) -> tuple[list[DecisionUpdateProposal], list[DecisionUpdateProposal]]:
    """Split *proposals* into accepted and rejected by their confidence.

    Returns ``(accepted, rejected)`` and labels each side —
    :data:`LABEL_APPLIED` is set later, only once a record has really been
    written (see :func:`apply_decision_updates`), so a label never claims more
    than happened.  The split is by the measured violation rate alone; nothing
    is read from or written to the ledger here.
    """
    accepted: list[DecisionUpdateProposal] = []
    rejected: list[DecisionUpdateProposal] = []
    for proposal in proposals:
        if proposal.confidence >= threshold:
            proposal.review_label = LABEL_APPLIED
            accepted.append(proposal)
        else:
            proposal.review_label = LABEL_REJECTED
            rejected.append(proposal)
    return accepted, rejected


def _supersede(
    workspace: Path, old: dict[str, Any], proposal: DecisionUpdateProposal
) -> dict[str, Any]:
    """Append a revision of *old* and close *old* as superseded by it.

    Written by hand rather than with :func:`agent_core.decisions.add_decision`
    because the two writes are one operation: ``resolved_by`` on the old record
    must point at the new id, and that id is only known while building it.  A
    single :func:`agent_core.decisions.save_decisions` call commits both, so the
    ledger never holds a superseded decision pointing nowhere.

    Fields the proposal does not suggest are inherited from *old*, so a
    revision narrows a decision without losing its context.
    """
    decisions = load_decisions(workspace)
    new_id = _next_id(decisions)
    title = proposal.suggested_title or str(old.get("title") or "")

    record: dict[str, Any] = {
        "id": new_id,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "title": title,
        "context": str(old.get("context") or ""),
        "decision": (
            proposal.suggested_decision
            if proposal.suggested_decision is not None
            else str(old.get("decision") or "")
        ),
        "rationale": (
            proposal.suggested_rationale
            if proposal.suggested_rationale is not None
            else str(old.get("rationale") or "")
        ),
        "affected_files": normalize_affected_files(
            workspace, list(old.get("affected_files") or [])
        ),
        "tags": list(old.get("tags") or []),
        "contradictions": [],
        "resolved_by": None,
        "status": STATUS_ACTIVE,
        "category": str(old.get("category") or ""),
        "source": SOURCE_IMPROVEMENT,
    }
    # The evidence is kept on the new record: it is why this revision exists.
    record["evidence"] = list(proposal.evidence)
    record["review_label"] = LABEL_APPLIED

    updated = [
        {**d, "status": STATUS_SUPERSEDED, "resolved_by": new_id}
        if str(d.get("id")) == str(old.get("id"))
        else d
        for d in decisions
    ]
    updated.append(record)
    save_decisions(workspace, updated)
    return record


def _suggestion_of(proposal: DecisionUpdateProposal) -> tuple[str, str]:
    """The (decision, rationale) text a proposal would write, before inheritance.

    Used as the dedup key by :func:`apply_decision_updates`.  Text is compared
    because it is what a reviewer sees as "the same suggestion": two proposals
    that would write the same wording are one suggestion, applied once.
    """
    return (
        str(proposal.suggested_decision or ""),
        str(proposal.suggested_rationale or ""),
    )


def apply_decision_updates(
    workspace: str | Path,
    proposals: list[DecisionUpdateProposal],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Apply accepted proposals to the ledger; leave rejected ones untouched.

    Each accepted proposal appends a new active record and marks the decision it
    revises ``superseded`` with ``resolved_by`` pointing at it, so the original
    wording survives in the audit trail.  A proposal is declined — writing
    nothing — when its confidence is below *threshold*, when it names a decision
    that is not in the ledger, when the decision is already closed, or when its
    exact wording is already recorded as an improvement, so applying the same
    proposal twice cannot duplicate a revision.

    Returns ``{"applied": [decision_id, ...], "rejected": [...]}``.  A proposal
    that fails the existence check is reported in ``rejected``: it was not
    declined on its merits, and the caller needs to tell those cases apart.
    """
    base = Path(workspace).resolve()
    result: dict[str, Any] = {"applied": [], "rejected": []}
    if not proposals:
        return result

    index = _decision_index(base)
    accepted, low_confidence = review_proposals(proposals, threshold=threshold)
    result["rejected"].extend(low_confidence)

    for proposal in accepted:
        old = index.get(str(proposal.decision_id).strip())
        if old is None:
            # Not a judgement about the suggestion — it names nothing.  Unlabel
            # it so the caller can tell "no such decision" from "declined".
            proposal.review_label = None
            result["rejected"].append(proposal)
            logger.warning(
                "improvement proposal for unknown decision %s ignored",
                proposal.decision_id,
            )
            continue
        suggestion = _suggestion_of(proposal)
        if any(
            str(r.get("source") or "") == SOURCE_IMPROVEMENT
            and (str(r.get("decision") or ""), str(r.get("rationale") or ""))
            == suggestion
            for r in index.values()
        ):
            # The same wording is already recorded as an improvement — applying
            # it again would add a second record of one suggestion.  Checked
            # before the status check: re-applying to a decision this very call
            # superseded lands on a closed record, and dedup catches it first.
            proposal.review_label = LABEL_REJECTED
            result["rejected"].append(proposal)
            continue

        status = str(old.get("status") or STATUS_ACTIVE)
        if status in _CLOSED_STATUSES:
            # Drift on a closed decision is a closure problem (RECOMMEND_RESOLVE),
            # not a wording problem, so rewording it is out of scope here.  A
            # proposal naming a superseded record lands in the same branch, which
            # is what makes applying one proposal twice a no-op.
            proposal.review_label = LABEL_REJECTED
            result["rejected"].append(proposal)
            continue

        _supersede(base, old, proposal)
        index = _decision_index(base)  # the new id must be visible to the next proposal
        result["applied"].append(str(proposal.decision_id))

    return result
