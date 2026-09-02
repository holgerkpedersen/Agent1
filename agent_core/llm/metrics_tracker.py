from __future__ import annotations
from collections import defaultdict
from typing import Any

from .llm_types import TaskType, ProfileType


class MetricsTracker:
    """Tracks success/failure metrics per profile+task combination.

    Also accumulates per-turn token and cost counters (plan ARCH item 16):
    :meth:`record_turn` accepts a :class:`~agent_core.llm.provider.ResponseMetrics`
    (or plain token/cost values) captured from a provider's
    ``last_response_metrics``.
    """

    def __init__(self) -> None:
        self._success_counts: dict[tuple[str, str], int] = defaultdict(int)
        self._failure_counts: dict[tuple[str, str], int] = defaultdict(int)
        self._latencies: dict[tuple[str, str], list[float]] = defaultdict(list)
        self._token_counts: dict[tuple[str, str], int] = defaultdict(int)
        self._token_entries: dict[tuple[str, str], int] = defaultdict(int)
        self._costs: dict[tuple[str, str], float] = defaultdict(float)
        self._cost_entries: dict[tuple[str, str], int] = defaultdict(int)
        # Cache hit/miss tracking per (task_type, profile_type) key.
        # Keys map to {"hits": int, "misses": int} for prompt template cache stats.
        self._cache_stats: dict[tuple[str, str], dict[str, int]] = {}

    def record_cache_hit(self, task_type: str, profile_type: str) -> None:
        """Record a template-cache hit for the given (task_type, profile_type)."""
        key = (task_type, profile_type)
        entry = self._cache_stats.get(key)
        if entry is None:
            entry = {"hits": 0, "misses": 0}
            self._cache_stats[key] = entry
        entry["hits"] += 1

    def record_cache_miss(self, task_type: str, profile_type: str) -> None:
        """Record a template-cache miss for the given (task_type, profile_type)."""
        key = (task_type, profile_type)
        entry = self._cache_stats.get(key)
        if entry is None:
            entry = {"hits": 0, "misses": 0}
            self._cache_stats[key] = entry
        entry["misses"] += 1

    def get_cache_metrics(self, task_type: TaskType, profile_type: ProfileType) -> dict[str, float | int]:
        """Return cache hit/miss stats for a (task_type, profile_type) pair.

        Returns {"hits", "misses", "total_lookups", "hit_rate_pct"} or an empty
        dict when no data has been recorded yet.
        """
        key = (task_type.value, profile_type.value)
        entry = self._cache_stats.get(key)
        if entry is None:
            return {}
        hits = entry["hits"]
        misses = entry["misses"]
        total = hits + misses
        hit_rate = round(hits / max(total, 1) * 100.0, 2)
        return {
            "hits": hits,
            "misses": misses,
            "total_lookups": total,
            "hit_rate_pct": hit_rate,
        }

    def get_all_cache_metrics(self) -> dict[tuple[str, str], dict[str, float | int]]:
        """Return cache stats for every (task_type, profile_type) that has data."""
        result: dict[tuple[str, str], dict[str, float | int]] = {}
        for key, entry in self._cache_stats.items():
            hits = entry["hits"]
            misses = entry["misses"]
            total = hits + misses
            hit_rate = round(hits / max(total, 1) * 100.0, 2)
            result[key] = {
                "hits": hits,
                "misses": misses,
                "total_lookups": total,
                "hit_rate_pct": hit_rate,
            }
        return result

    def record_success(self, task_type: TaskType, profile_type: ProfileType, latency_seconds: float) -> None:
        key = (task_type.value, profile_type.value)
        self._success_counts[key] += 1
        self._latencies[key].append(latency_seconds)

    def record_failure(self, task_type: TaskType, profile_type: ProfileType) -> None:
        key = (task_type.value, profile_type.value)
        self._failure_counts[key] += 1

    def record_turn(
        self,
        task_type: TaskType,
        profile_type: ProfileType,
        latency_seconds: float | None = None,
        tokens: int | None = None,
        cost: float | None = None,
        metrics: Any | None = None,
    ) -> None:
        """Record a successful turn's token/latency/cost accounting.

        *metrics* may be a ``ResponseMetrics`` object (then tokens/cost are
        taken from it); explicit *tokens*/*cost* override it.  *latency_seconds*
        defaults to ``metrics.latency_ms / 1000`` when available.
        """
        key = (task_type.value, profile_type.value)
        if metrics is not None:
            tokens = tokens if tokens is not None else getattr(metrics, "total_tokens", None)
            cost = cost if cost is not None else getattr(metrics, "cost", 0.0)
            if latency_seconds is None:
                latency_seconds = getattr(metrics, "latency_ms", 0.0) / 1000.0
        self._success_counts[key] += 1
        if latency_seconds is not None:
            self._latencies[key].append(latency_seconds)
        if tokens is not None:
            self._token_counts[key] += tokens
            self._token_entries[key] += 1
        if cost is not None:
            self._costs[key] += cost
            self._cost_entries[key] += 1

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
        tokens = self._token_counts[key]
        token_entries = self._token_entries[key]
        costs = self._costs[key]
        cost_entries = self._cost_entries[key]
        return {
            "success_count": successes,
            "failure_count": failures,
            "avg_latency": avg_latency,
            "total_tokens": tokens,
            "avg_tokens": round(tokens / token_entries, 1) if token_entries else 0,
            "total_cost": round(costs, 6),
            "avg_cost": round(costs / cost_entries, 6) if cost_entries else 0.0,
            "last_updated": float(successes + failures),
        }

    def reset(self) -> None:
        """Clears all recorded metrics."""
        self._success_counts.clear()
        self._failure_counts.clear()
        self._latencies.clear()
        self._token_counts.clear()
        self._token_entries.clear()
        self._costs.clear()
        self._cost_entries.clear()
        self._cache_stats.clear()