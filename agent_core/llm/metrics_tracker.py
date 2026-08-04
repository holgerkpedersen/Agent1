from collections import defaultdict

from .types import TaskType, ProfileType


class MetricsTracker:
    """Tracks success/failure metrics per profile+task combination."""

    def __init__(self) -> None:
        self._success_counts: dict[tuple[str, str], int] = defaultdict(int)
        self._failure_counts: dict[tuple[str, str], int] = defaultdict(int)
        self._latencies: dict[tuple[str, str], list[float]] = defaultdict(list)

    def record_success(self, task_type: TaskType, profile_type: ProfileType, latency_seconds: float) -> None:
        key = (task_type.value, profile_type.value)
        self._success_counts[key] += 1
        self._latencies[key].append(latency_seconds)

    def record_failure(self, task_type: TaskType, profile_type: ProfileType) -> None:
        key = (task_type.value, profile_type.value)
        self._failure_counts[key] += 1

    def get_success_rate(self, profile_type: ProfileType) -> float:
        """Return overall success rate for a profile across all task types."""
        successes = sum(v for k, v in self._success_counts.items() if k[1] == profile_type.value)
        failures = sum(v for k, v in self._failure_counts.items() if k[1] == profile_type.value)
        total = successes + failures
        if total == 0:
            return 0.5
        return successes / total

    def get_failure_count(self, profile_type: ProfileType) -> int:
        """Return total failure count for a profile across all task types."""
        return sum(v for k, v in self._failure_counts.items() if k[1] == profile_type.value)

    def get_performance_score(self, task_type: TaskType, profile_type: ProfileType) -> float:
        """Returns a performance score in [0.0, 1.0] for the given combo."""
        key = (task_type.value, profile_type.value)
        successes = self._success_counts[key]
        failures = self._failure_counts[key]
        total = successes + failures
        if total == 0:
            return 0.5
        success_rate = successes / total
        latencies = self._latencies[key]
        avg_latency = sum(latencies) / len(latencies) if latencies else 1.0
        latency_factor = min(1.0, 1.0 / max(avg_latency, 0.001))
        return success_rate * 0.7 + latency_factor * 0.3

    def get_all_scores(self) -> dict[tuple[TaskType, ProfileType], float]:
        """Returns performance scores for all recorded profile+task combos."""
        scores: dict[tuple[TaskType, ProfileType], float] = {}
        seen_keys: set[tuple[str, str]] = set()
        for key in self._success_counts.keys() | self._failure_counts.keys():
            if key not in seen_keys:
                seen_keys.add(key)
                task_type = TaskType(key[0])
                profile_type = ProfileType(key[1])
                scores[(task_type, profile_type)] = self.get_performance_score(task_type, profile_type)
        return scores

    def get_metrics(self, task_type: TaskType, profile_type: ProfileType) -> dict[str, float | int]:
        """Returns raw metrics for meta-policy evolution and cache updates."""
        key = (task_type.value, profile_type.value)
        successes = self._success_counts[key]
        failures = self._failure_counts[key]
        latencies = self._latencies[key]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        return {
            "success_count": successes,
            "failure_count": failures,
            "avg_latency": avg_latency,
            "last_updated": float(successes + failures),
        }

    def reset(self) -> None:
        """Clears all recorded metrics."""
        self._success_counts.clear()
        self._failure_counts.clear()
        self._latencies.clear()