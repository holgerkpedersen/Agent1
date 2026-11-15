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
from pathlib import Path
from typing import Any

from agent_core.decision_collector import DecisionCollector
from agent_core.decisions import (
    STATUS_ARCHIVED,
    STATUS_ACTIVE,
    STATUS_SUPERSEDED,
    STATUS_VOID,
    load_decisions,
    load_discrepancies,
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
    """

    decision_id: str
    suggested_title: str | None = None
    suggested_decision: str | None = None
    suggested_rationale: str | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0


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
