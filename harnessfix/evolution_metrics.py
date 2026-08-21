"""Per-run quality scoring from HarnessFix trace events (read-only).

This module computes a per-run quality score from the JSONL traces produced by
``harnessfix/tracing.py`` and indexed by ``harnessfix/history.py``.  It reuses
the sliding-window design from ``agent1/evolution/metrics.py`` but binds it to
real trace data instead of caller-supplied metrics.

VERIFIED TRACE SCHEMA (do not assume fields that do not exist):
- ``loop_end`` event carries ``outcome`` (observed: completed, no_progress,
  budget_exhausted, stuck, error).  Success == ``outcome == "completed"``.
  There is NO ``success`` field.
- ``tool_result`` events carry ``duration_s`` (float seconds, per tool).  There
  is NO per-run ``latency`` field; run latency is the SUM of tool durations.
- There is NO explicit ``score`` field in trace data; the per-run score is
  DERIVED from success + a latency penalty.

Read-only contract: this module never calls the LLM and never opens any file
for writing outside its own unit tests.  Trace strings are never eval/exec'd —
only the numeric ``duration_s`` and the string ``outcome`` are read.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Sequence

from agent1.evolution.metrics import EvolutionMetrics, ExecutionMetric

from .corpus import collect_traces
from .reader import TraceValidationError, read_trace
from .tracing import KIND_LOOP_END, KIND_TOOL_RESULT

#: Default sliding-window size (number of recent runs kept).
DEFAULT_WINDOW_SIZE = 10
#: Default quality threshold below which evolution is triggered.
DEFAULT_THRESHOLD = 0.7
#: Run latency (seconds) that produces a full penalty.
DEFAULT_LATENCY_BUDGET_S = 30.0
#: Maximum score reduction contributed by latency.
DEFAULT_PENALTY_WEIGHT = 0.3


def _run_outcome(events: Sequence[dict[str, Any]]) -> str | None:
    """Return the ``outcome`` of the single ``loop_end`` event, if present."""
    for ev in events:
        if ev.get("kind") == KIND_LOOP_END:
            return str(ev.get("outcome", "completed"))
    return None


def _run_latency_s(events: Sequence[dict[str, Any]]) -> float:
    """Sum the ``duration_s`` of every ``tool_result`` event (run latency)."""
    total = 0.0
    for ev in events:
        if ev.get("kind") == KIND_TOOL_RESULT:
            dur = ev.get("duration_s")
            if isinstance(dur, (int, float)):
                total += float(dur)
    return total


def score_run(
    events: Sequence[dict[str, Any]],
    latency_budget_s: float = DEFAULT_LATENCY_BUDGET_S,
    penalty_weight: float = DEFAULT_PENALTY_WEIGHT,
) -> float:
    """Compute a per-run quality score in [0, 1] from trace events.

    score = 1.0 if the run succeeded (outcome == "completed") else 0.0,
    minus a latency penalty = min(latency_s / budget, 1) * weight,
    clamped to [0, 1].  A run with no ``loop_end`` scores 0.0 (incomplete).
    """
    outcome = _run_outcome(events)
    if outcome is None or outcome != "completed":
        base = 0.0
    else:
        base = 1.0
    latency = _run_latency_s(events)
    if latency_budget_s <= 0:
        penalty = 0.0
    else:
        penalty = min(latency / latency_budget_s, 1.0) * penalty_weight
    return max(0.0, min(1.0, base - penalty))


def iter_run_scores(
    trace_dir: str | os.PathLike[str],
    latency_budget_s: float = DEFAULT_LATENCY_BUDGET_S,
    penalty_weight: float = DEFAULT_PENALTY_WEIGHT,
) -> Iterable[tuple[str, float]]:
    """Yield ``(task_id, score)`` for every readable trace in *trace_dir*.

    Unreadable/corrupt traces are skipped (read_trace raises
    TraceValidationError).  Traces without a ``loop_end`` score 0.0.
    """
    for path in collect_traces(Path(trace_dir)):
        try:
            events = read_trace(path)
        except TraceValidationError:
            continue
        if not events:
            continue
        task_id = str(events[0].get("task_id", Path(path).stem))
        yield task_id, score_run(events, latency_budget_s, penalty_weight)


class EvolutionMetricsScorer:
    """Sliding-window quality scorer over HarnessFix run traces.

    Wraps :class:`agent1.evolution.metrics.EvolutionMetrics`, feeding it a
    derived :class:`ExecutionMetric` per run.  Exposes ``windowed_average()``
    and ``should_evolve()`` (default threshold 0.7) per the project spec.
    """

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        threshold: float = DEFAULT_THRESHOLD,
        latency_budget_s: float = DEFAULT_LATENCY_BUDGET_S,
        penalty_weight: float = DEFAULT_PENALTY_WEIGHT,
    ) -> None:
        self.window_size = window_size
        self.threshold = threshold
        self.latency_budget_s = latency_budget_s
        self.penalty_weight = penalty_weight
        self._metrics = EvolutionMetrics(window_size=window_size, threshold=threshold)

    def record_trace(self, events: Sequence[dict[str, Any]]) -> float:
        """Score one run's events and record it in the sliding window.

        Returns the derived per-run score.
        """
        score = score_run(events, self.latency_budget_s, self.penalty_weight)
        outcome = _run_outcome(events)
        success = outcome == "completed"
        latency = _run_latency_s(events)
        self._metrics.record(ExecutionMetric(score=score, success=success, latency=latency))
        return score

    def record_trace_file(self, path: str | os.PathLike[str]) -> float | None:
        """Score and record a single trace file; None if unreadable."""
        try:
            events = read_trace(Path(path))
        except TraceValidationError:
            return None
        if not events:
            return None
        return self.record_trace(events)

    def load_corpus(self, trace_dir: str | os.PathLike[str]) -> int:
        """Record every readable trace in *trace_dir*. Returns run count."""
        count = 0
        for task_id, score in iter_run_scores(
            trace_dir, self.latency_budget_s, self.penalty_weight
        ):
            # Re-derive success/latency from the file for the ExecutionMetric.
            try:
                events = read_trace(Path(trace_dir) / f"{task_id}.jsonl")
            except TraceValidationError:
                continue
            outcome = _run_outcome(events)
            self._metrics.record(
                ExecutionMetric(
                    score=score,
                    success=outcome == "completed",
                    latency=_run_latency_s(events),
                )
            )
            count += 1
        return count

    def windowed_average(self) -> float:
        """Mean score across the current sliding window (0.0 when empty)."""
        return self._metrics.average_score()

    def should_evolve(self) -> bool:
        """True when the windowed average drops below the threshold."""
        return self._metrics.should_evolve()

    def summary(self) -> dict[str, Any]:
        """Snapshot of the current windowed quality state."""
        s = self._metrics.summary()
        s["windowed_average"] = self.windowed_average()
        s["should_evolve"] = self.should_evolve()
        return s
