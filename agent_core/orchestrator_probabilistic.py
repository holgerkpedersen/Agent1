"""Phase 3: Probabilistic integration.

Adds a threshold-based commitment layer on top of the base orchestrator.
Speculative branches produce scored candidates; this module decides whether
to COMMIT to the best candidate or REFUSE (and broaden/re-evaluate).
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional

from agent_core.scoring import CandidateSelector, ScoredCandidate


class Decision(Enum):
    """Outcome of a commitment check."""

    COMMIT = "commit"
    REFUSE = "refuse"


@dataclass
class CommitmentResult:
    """A decision plus the candidate it refers to (None when refusing)."""

    kind: Decision
    best: Optional[ScoredCandidate]


class ProbabilisticOrchestrator:
    """Wraps a base orchestrator and adds probabilistic commitment logic."""

    def __init__(self, base_orchestrator: Any = None) -> None:
        self.base_orchestrator = base_orchestrator
        self._selector = CandidateSelector()

    def decide_commitment(
        self, candidates: List[ScoredCandidate], threshold: float
    ) -> CommitmentResult:
        """Pick the best candidate and commit only if it meets the threshold.

        Raises:
            ValueError: if candidates is empty, or threshold is outside [0, 1].
        """
        if not candidates:
            raise ValueError("candidates must be a non-empty list")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be within [0.0, 1.0]")

        best = self._selector.select_best(candidates)

        if best.score >= threshold:  # boundary is inclusive
            return CommitmentResult(kind=Decision.COMMIT, best=best)
        return CommitmentResult(kind=Decision.REFUSE, best=None)
