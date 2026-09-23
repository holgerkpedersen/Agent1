import pytest
from agent_core.scoring import ScoredCandidate, CandidateSelector

def test_candidate_selector_picks_highest_score():
    candidates = [
        ScoredCandidate(result={"path": "A"}, score=0.5),
        ScoredCandidate(result={"path": "B"}, score=0.9),
        ScoredCandidate(result={"path": "C"}, score=0.7),
    ]
    selector = CandidateSelector()
    best = selector.select_best(candidates)
    assert best.result == {"path": "B"}
    assert best.score == 0.9

def test_candidate_selector_empty_list():
    selector = CandidateSelector()
    with pytest.raises(ValueError, match="No candidates provided"):
        selector.select_best([])

def test_candidate_selector_tie_breaks_first():
    candidates = [
        ScoredCandidate(result={"path": "A"}, score=0.8),
        ScoredCandidate(result={"path": "B"}, score=0.8),
    ]
    selector = CandidateSelector()
    best = selector.select_best(candidates)
    assert best.result == {"path": "A"}
