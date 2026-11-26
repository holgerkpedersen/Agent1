"""Cost-aware model router — closes plan item #4 (close the meta-policy loop).

Provides:
  * :class:`SessionBudget` — tracks cumulative cost/tokens for a session,
    with an optional cap that blocks profile switches when exceeded.
  * :func:`compute_cost_per_turn` — estimates the USD cost of one LLM turn
    from prompt/completion token counts and a :class:`~ProfileMetadata`.
  * ``_session_budget`` — module-level singleton shared by
    ``agent_core.llm.learning.recommend_profile`` so budget state is
    consistent across calls.

Contract: **never raises**. A missing profile or bad data falls back to
zero cost, which effectively disables the cost-aware branch until real
metrics arrive.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "SessionBudget",
    "_session_budget",
    "compute_cost_per_turn",
]


class SessionBudget:
    """Tracks cumulative token and cost usage for a single session.

    Parameters
    ----------
    cap :
        Maximum USD cost before the router refuses to switch profiles.
        ``None`` means unlimited (default).
    """

    def __init__(self, *, cap: Optional[float] = None) -> None:
        self.cap = cap
        self.total_cost: float = 0.0
        self.total_tokens: int = 0

    def add_usage(self, *, cost: float, tokens: int) -> None:
        """Record one turn's token and cost usage."""
        self.total_cost += max(cost, 0.0)
        self.total_tokens += max(tokens, 0)

    def exceeded(self) -> bool:
        """True when the budget cap has been reached or passed."""
        if self.cap is None:
            return False
        return self.total_cost >= self.cap


#: Shared singleton — all recommendation calls consult the same budget.
_session_budget = SessionBudget(cap=None)  # default: unlimited, user can set via pref


def compute_cost_per_turn(
    prompt_tokens: int,
    completion_tokens: int,
    profile_metadata,
) -> float:
    """Estimate USD cost of one turn from token counts and profile pricing.

    Returns zero when the profile has no cost metadata (unknown model).
    """
    try:
        pp = getattr(profile_metadata, "prompt_cost_per_1m", 0.0) or 0.0
        cp = getattr(
            profile_metadata, "completion_cost_per_1m", 0.0
        ) or 0.0
        return (prompt_tokens * pp + completion_tokens * cp) / 1_000_000
    except Exception:  # noqa: BLE001 - must never break a turn
        logger.debug("Cost estimation failed (no-op)", exc_info=True)
        return 0.0
