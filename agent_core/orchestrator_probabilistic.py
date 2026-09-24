"""Phase 3: Probabilistic integration.

Adds a threshold-based commitment layer on top of the base orchestrator.
Speculative branches produce scored candidates; this module decides whether
to COMMIT to the best candidate or REFUSE (and broaden/re-evaluate).
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional

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

    def run_speculative(
        self,
        reasoning_func: Callable[[int, Any], Any],
        context: Any,
        threshold: float,
        scorer: Callable[[Any], float],
        num_branches: int = 3,
        timeout: Optional[float] = 5.0,
    ) -> CommitmentResult:
        """Run the full deliberation loop against the base orchestrator.

        Dispatches speculative branches, waits for them, scores each result,
        and commits to the best candidate only if it meets the threshold.

        Returns:
            COMMIT with the best candidate, or REFUSE when nothing qualifies
            (no branches, all branches failed, or all scores below threshold).

        Raises:
            ValueError: if no base orchestrator is set, or threshold is
                outside [0, 1] (validated before any branch is dispatched).
            RuntimeError: if the branches do not finish within ``timeout``.
        """
        if self.base_orchestrator is None:
            raise ValueError("base_orchestrator is required to run speculative deliberation")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be within [0.0, 1.0]")

        task_ids = self.base_orchestrator.dispatch_speculative(
            reasoning_func, context, num_branches=num_branches
        )
        if not self.base_orchestrator.wait_for_completion(timeout=timeout):
            raise RuntimeError("speculative deliberation timed out after %ss" % timeout)

        candidates: List[ScoredCandidate] = []
        for task_id in task_ids:
            result = self.base_orchestrator.get_result(task_id)
            # Failed branches are reported as {"error": ...} by get_result;
            # they are not committable candidates.
            if result is None or "error" in result:
                continue
            candidates.append(ScoredCandidate(result=result, score=float(scorer(result))))

        if not candidates:
            return CommitmentResult(kind=Decision.REFUSE, best=None)
        return self.decide_commitment(candidates, threshold)
