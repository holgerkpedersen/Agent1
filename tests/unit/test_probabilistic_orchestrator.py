"""Tests for the Probabilistic Orchestrator (Phase 3: Probabilistic Integration)."""
import pytest
from unittest.mock import MagicMock

from agent_core.scoring import ScoredCandidate, CandidateSelector
from agent_core.orchestrator_probabilistic import ProbabilisticOrchestrator, Decision


@pytest.fixture
def base_orchestrator():
    """A mocked base orchestrator so unit tests avoid heavy execution logic."""
    return MagicMock()


def test_commits_when_candidate_exceeds_threshold(base_orchestrator):
    orch = ProbabilisticOrchestrator(base_orchestrator=base_orchestrator)
    candidates = [
        ScoredCandidate(result="low", score=0.4),
        ScoredCandidate(result="high", score=0.9),
    ]

    decision = orch.decide_commitment(candidates, threshold=0.7)

    assert decision.kind == Decision.COMMIT
    assert decision.best is not None
    assert decision.best.result == "high"


def test_refuses_when_all_below_threshold(base_orchestrator):
    orch = ProbabilisticOrchestrator(base_orchestrator=base_orchestrator)
    candidates = [
        ScoredCandidate(result="a", score=0.3),
        ScoredCandidate(result="b", score=0.5),
    ]

    decision = orch.decide_commitment(candidates, threshold=0.7)

    assert decision.kind == Decision.REFUSE
    assert decision.best is None


def test_empty_candidate_list_raises(base_orchestrator):
    orch = ProbabilisticOrchestrator(base_orchestrator=base_orchestrator)
    with pytest.raises(ValueError):
        orch.decide_commitment([], threshold=0.7)


def test_negative_threshold_raises(base_orchestrator):
    orch = ProbabilisticOrchestrator(base_orchestrator=base_orchestrator)
    candidates = [ScoredCandidate(result="x", score=0.5)]
    with pytest.raises(ValueError):
        orch.decide_commitment(candidates, threshold=-0.1)


def test_commit_uses_candidate_selector_for_best(base_orchestrator):
    """The committed candidate must be the max-scored one, via CandidateSelector."""
    orch = ProbabilisticOrchestrator(base_orchestrator=base_orchestrator)
    candidates = [
        ScoredCandidate(result="mid", score=0.75),
        ScoredCandidate(result="best", score=0.95),
        ScoredCandidate(result="low", score=0.2),
    ]

    decision = orch.decide_commitment(candidates, threshold=0.7)
    expected = CandidateSelector().select_best(candidates)

    assert decision.kind == Decision.COMMIT
    assert decision.best is expected


def test_threshold_boundary_is_inclusive(base_orchestrator):
    """A score exactly equal to the threshold should commit."""
    orch = ProbabilisticOrchestrator(base_orchestrator=base_orchestrator)
    candidates = [ScoredCandidate(result="exact", score=0.7)]

    decision = orch.decide_commitment(candidates, threshold=0.7)

    assert decision.kind == Decision.COMMIT
    assert decision.best.result == "exact"
