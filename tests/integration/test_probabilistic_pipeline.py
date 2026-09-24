"""Phase 4: integration tests for the speculative deliberation pipeline.

Wires ``ProbabilisticOrchestrator`` into the real execution loop:
speculative branch generation (``Orchestrator.dispatch_speculative``) ->
result collection -> scoring -> ``decide_commitment`` -> COMMIT/REFUSE.

Uses the real ``agent_core.swarm_orchestrator.Orchestrator`` thread pool --
not a mock -- so dispatch, result collection, branch failure and timeout
paths are exercised end-to-end.
"""

import time

import pytest

from agent_core.orchestrator_probabilistic import Decision, ProbabilisticOrchestrator
from agent_core.swarm_orchestrator import Orchestrator


def _quality_scorer(result: dict) -> float:
    """Score a branch result by its declared quality field."""
    return float(result["quality"])


def test_pipeline_commits_best_branch():
    """The highest-scoring branch above threshold is committed to."""
    with Orchestrator(agents=[], max_workers=4) as base:
        def reasoning_func(branch_id: int, context: dict) -> dict:
            return {"branch": branch_id, "quality": [0.2, 0.9, 0.5][branch_id]}

        orch = ProbabilisticOrchestrator(base_orchestrator=base)
        decision = orch.run_speculative(
            reasoning_func=reasoning_func,
            context={},
            threshold=0.7,
            scorer=_quality_scorer,
            num_branches=3,
            timeout=5.0,
        )

    assert decision.kind == Decision.COMMIT
    assert decision.best is not None
    assert decision.best.result["branch"] == 1
    assert decision.best.score == 0.9


def test_pipeline_refuses_when_all_below_threshold():
    """No branch reaches the threshold -> REFUSE with no committed candidate."""
    with Orchestrator(agents=[], max_workers=4) as base:
        def reasoning_func(branch_id: int, context: dict) -> dict:
            return {"branch": branch_id, "quality": [0.1, 0.3, 0.2][branch_id]}

        orch = ProbabilisticOrchestrator(base_orchestrator=base)
        decision = orch.run_speculative(
            reasoning_func=reasoning_func,
            context={},
            threshold=0.7,
            scorer=_quality_scorer,
            num_branches=3,
            timeout=5.0,
        )

    assert decision.kind == Decision.REFUSE
    assert decision.best is None


def test_pipeline_skips_failed_branches():
    """A branch that raises is excluded from candidates; others still commit."""
    with Orchestrator(agents=[], max_workers=4) as base:
        def reasoning_func(branch_id: int, context: dict) -> dict:
            if branch_id == 1:
                raise ValueError("branch blew up")
            return {"branch": branch_id, "quality": 0.8}

        orch = ProbabilisticOrchestrator(base_orchestrator=base)
        decision = orch.run_speculative(
            reasoning_func=reasoning_func,
            context={},
            threshold=0.7,
            scorer=_quality_scorer,
            num_branches=3,
            timeout=5.0,
        )

    assert decision.kind == Decision.COMMIT
    assert decision.best is not None
    assert decision.best.result["branch"] in (0, 2)
    assert "error" not in decision.best.result


def test_pipeline_refuses_when_all_branches_fail():
    """Every branch raising leaves no committable candidate -> REFUSE."""
    with Orchestrator(agents=[], max_workers=4) as base:
        def reasoning_func(branch_id: int, context: dict) -> dict:
            raise RuntimeError("branch %d failed" % branch_id)

        orch = ProbabilisticOrchestrator(base_orchestrator=base)
        decision = orch.run_speculative(
            reasoning_func=reasoning_func,
            context={},
            threshold=0.1,
            scorer=_quality_scorer,
            num_branches=3,
            timeout=5.0,
        )

    assert decision.kind == Decision.REFUSE
    assert decision.best is None


def test_pipeline_requires_base_orchestrator():
    """Running the pipeline without a base orchestrator is a caller error."""
    orch = ProbabilisticOrchestrator()
    with pytest.raises(ValueError, match="base_orchestrator"):
        orch.run_speculative(
            reasoning_func=lambda b, c: {"quality": 1.0},
            context={},
            threshold=0.5,
            scorer=_quality_scorer,
        )


def test_pipeline_rejects_out_of_range_threshold():
    """An invalid threshold fails fast with a threshold-specific error."""
    with Orchestrator(agents=[], max_workers=4) as base:
        def reasoning_func(branch_id: int, context: dict) -> dict:
            return {"branch": branch_id, "quality": 0.5}

        orch = ProbabilisticOrchestrator(base_orchestrator=base)
        with pytest.raises(ValueError, match="threshold"):
            orch.run_speculative(
                reasoning_func=reasoning_func,
                context={},
                threshold=-0.1,
                scorer=_quality_scorer,
                num_branches=1,
                timeout=5.0,
            )


def test_pipeline_times_out_when_branches_stall():
    """Branches that outlive the timeout abort the pipeline loudly."""
    with Orchestrator(agents=[], max_workers=4) as base:
        def reasoning_func(branch_id: int, context: dict) -> dict:
            time.sleep(0.5)
            return {"branch": branch_id, "quality": 0.9}

        orch = ProbabilisticOrchestrator(base_orchestrator=base)
        with pytest.raises(RuntimeError, match="timed out"):
            orch.run_speculative(
                reasoning_func=reasoning_func,
                context={},
                threshold=0.7,
                scorer=_quality_scorer,
                num_branches=1,
                timeout=0.01,
            )


def test_pipeline_with_zero_branches_refuses():
    """Nothing to deliberate on -> REFUSE rather than an exception."""
    with Orchestrator(agents=[], max_workers=4) as base:
        orch = ProbabilisticOrchestrator(base_orchestrator=base)
        decision = orch.run_speculative(
            reasoning_func=lambda b, c: {"quality": 1.0},
            context={},
            threshold=0.5,
            scorer=_quality_scorer,
            num_branches=0,
            timeout=5.0,
        )

    assert decision.kind == Decision.REFUSE
    assert decision.best is None
