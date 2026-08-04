from collections import defaultdict
from typing import Optional

CONFIDENCE_THRESHOLD = 0.6


class RefinementVoter:
    """Collects votes on prompt refinements and decides via majority rule
    with a confidence threshold, delegating the final quorum check to an
    external consensus mechanism."""

    def __init__(self, quorum_threshold: float = 0.5) -> None:
        self._votes: dict[str, list[bool]] = defaultdict(list)
        from agent_core.llm.orchestrator import ConsensusVoter
        self._consensus = ConsensusVoter(quorum_threshold=quorum_threshold)

    def collect_vote(self, refinement_id: str, vote: bool) -> None:
        self._votes[refinement_id].append(vote)

    def decide(self, refinement_id: str) -> Optional[bool]:
        votes_list = self._votes.get(refinement_id, [])
        if not votes_list:
            return None
        yes_count = sum(1 for v in votes_list if v)
        confidence = yes_count / len(votes_list)
        majority = yes_count > (len(votes_list) - yes_count)
        if confidence < CONFIDENCE_THRESHOLD or not majority:
            return None
        # integrate external consensus mechanism as the quorum gate
        return self._consensus.tally_votes(refinement_id)

    def vote_status(self, refinement_id: str) -> dict[str, int]:
        votes_list = self._votes.get(refinement_id, [])
        yes_count = sum(1 for v in votes_list if v)
        no_count = len(votes_list) - yes_count
        return {"yes": yes_count, "no": no_count, "total": len(votes_list)}

    def reset(self, refinement_id: Optional[str] = None) -> None:
        if refinement_id is None:
            self._votes.clear()
        else:
            self._votes.pop(refinement_id, None)