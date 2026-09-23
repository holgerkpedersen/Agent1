from dataclasses import dataclass
from typing import Any, List

@dataclass
class ScoredCandidate:
    result: Any
    score: float

class CandidateSelector:
    def select_best(self, candidates: List[ScoredCandidate]) -> ScoredCandidate:
        if not candidates:
            raise ValueError("No candidates provided")
        
        # Simple implementation: pick the one with the highest score.
        # In case of ties, max() returns the first occurrence, which matches my test expectation.
        return max(candidates, key=lambda c: c.score)
