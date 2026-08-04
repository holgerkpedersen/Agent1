"""LLM orchestrator with consensus voting support."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from .types import VoteResult


class ConsensusVoter:
    """In-memory consensus voter that optionally persists votes to a SQLite table.

    Tracks per-template_id vote counts across approve/reject/abstain and determines
    whether quorum approval threshold is met.
    """

    def __init__(self, quorum_threshold: float = 0.5) -> None:
        if not (0.0 <= quorum_threshold <= 1.0):
            raise ValueError("quorum_threshold must be between 0.0 and 1.0")
        self.quorum_threshold: float = quorum_threshold
        # In-memory vote store keyed by template_id -> agent_id -> VoteResult
        self._votes: Dict[str, Dict[str, VoteResult]] = {}

    def _ensure_template(self, template_id: str) -> None:
        if template_id not in self._votes:
            self._votes[template_id] = {}

    def vote(
        self,
        template_id: str,
        agent_id: str,
        result: VoteResult,
        *,
        db_path: Optional[str | Path] = None,
    ) -> None:
        """Record a single agent's vote for a given template.

        Optionally persists the vote into a ``prompt_votes`` table when ``db_path`` is provided.
        """
        self._ensure_template(template_id)
        # Overwrite previous vote from same agent (latest wins).
        self._votes[template_id][agent_id] = result

        if db_path is not None:
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS prompt_votes (
                        template_id TEXT NOT NULL,
                        agent_id    TEXT NOT NULL,
                        result      TEXT NOT NULL,
                        PRIMARY KEY (template_id, agent_id)
                    )
                    """
                )
                conn.execute(
                    "INSERT OR REPLACE INTO prompt_votes (template_id, agent_id, result) VALUES (?, ?, ?)",
                    (template_id, agent_id, _vote_result_to_str(result)),
                )
                conn.commit()
            finally:
                conn.close()

    def tally_votes(self, template_id: str) -> bool:
        """Return True if quorum approval threshold is met for the given template."""
        counts = self.get_vote_status(template_id)
        total = counts["approve"] + counts["reject"] + counts["abstain"]
        if total == 0:
            return False
        approval_ratio = counts["approve"] / total
        return approval_ratio >= self.quorum_threshold

    def get_vote_status(self, template_id: str) -> Dict[str, int]:
        """Return vote counts for approve/reject/abstain keyed by ``template_id``.

        Returns zeroed counts if no votes have been recorded yet.
        """
        agent_votes = self._votes.get(template_id, {})
        counts: Dict[str, int] = {"approve": 0, "reject": 0, "abstain": 0}
        for result in agent_votes.values():
            key = _vote_result_to_str(result)
            if key in counts:
                counts[key] += 1
        return counts


def _vote_result_to_str(value: VoteResult) -> str:
    """Convert a ``VoteResult`` enum member to its lowercase string name."""
    return value.name.lower()


__all__: list[str] = ["ConsensusVoter"]

# Re-export commonly used types for convenience within the orchestrator module.
Any  # noqa: F841 - kept for type-checking compatibility with downstream imports