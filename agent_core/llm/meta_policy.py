# agent_core/llm/meta_policy.py
from .metrics_tracker import MetricsTracker
from .types import TaskType, ProfileType

DECAY_FACTOR = 0.95


class MetaPolicyEvolver:
    """Evolves profile selection weights from tracked performance data."""

    def __init__(self) -> None:
        self._weights: dict[ProfileType, float] = {
            ProfileType.FAST_CODEGEN: 1.0,
            ProfileType.DEEP_ANALYSIS: 1.0,
            ProfileType.PRECISE: 1.0,
        }

    def update_weights(self, tracker: MetricsTracker) -> dict[ProfileType, float]:
        """Adjust weights from success rate and failure count; decay prevents overfitting."""
        for profile_type in self._weights:
            success_rate = tracker.get_success_rate(profile_type)
            failure_count = tracker.get_failure_count(profile_type)
            adjustment = (success_rate - 0.5) * 0.2
            # Dampen adjustments when recent failures are frequent to avoid overfitting
            if failure_count > 0:
                adjustment *= DECAY_FACTOR ** min(failure_count, 10)
            if adjustment != 0:
                new_weight = max(0.1, min(self._weights[profile_type] + adjustment, 2.0))
                self._weights[profile_type] = new_weight * DECAY_FACTOR
        return dict(self._weights)

    def get_preference_scores(self) -> dict[ProfileType, float]:
        """Return current profile preference scores."""
        return dict(self._weights)