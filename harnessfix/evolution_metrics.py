"""Per-run quality scoring from HarnessFix trace events (read-only).

This module computes a per-run quality score from the JSONL traces produced by
``harnessfix/tracing.py`` and indexed by ``harnessfix/history.py``.  It reuses
the sliding-window design from ``agent_core/evolution_metrics.py`` but binds it to
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

from agent_core.evolution_metrics import EvolutionMetrics, ExecutionMetric

from .corpus import collect_traces, is_countable_events
from .reader import TraceValidationError, read_trace
from .tracing import KIND_LLM_RESPONSE, KIND_LOOP_END, KIND_TOOL_RESULT

#: Guard-terminated loop_end outcomes (decision #052): the loop guards stopped
#: the LOOP, not the answer — such a run that still delivered a substantive
#: final answer DID the task and must not score 0.0.  ``error`` is deliberately
#: NOT here: a provider/transport failure is a real failure even if some text
#: was produced, so an error run keeps scoring 0.0.
DELIVERED_OUTCOMES = frozenset({"stuck", "budget_exhausted", "no_progress"})

#: A guard-terminated run that delivered an answer is worth slightly less than
#: a cleanly completed one (the guards firing is itself a signal to evolve).
DEFAULT_DELIVERED_BASE = 0.9

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


def _has_substantive_final_answer(events: Sequence[dict[str, Any]]) -> bool:
    """True when the LAST ``llm_response`` text is a substantive answer.

    Event-level twin of :meth:`harnessfix.htir.TraceGraph.has_final_answer`
    (same :data:`harnessfix.htir.MIN_FINAL_ANSWER_CHARS` threshold, imported
    lazily to avoid a module cycle).  This is what makes decision #052's
    refinement apply to SCORING as well as diagnosis: a guard-terminated run
    that delivered its final answer did the task.
    """
    from .htir import MIN_FINAL_ANSWER_CHARS

    answer = ""
    for ev in events:
        if ev.get("kind") == KIND_LLM_RESPONSE:
            answer = str(ev.get("text", "")).strip()
    return len(answer) >= MIN_FINAL_ANSWER_CHARS


def score_run(
    events: Sequence[dict[str, Any]],
    latency_budget_s: float = DEFAULT_LATENCY_BUDGET_S,
    penalty_weight: float = DEFAULT_PENALTY_WEIGHT,
    delivered_base: float = DEFAULT_DELIVERED_BASE,
) -> float:
    """Compute a per-run quality score in [0, 1] from trace events.

    Base = 1.0 when the run succeeded (outcome == "completed"); when the loop
    guards terminated the run (stuck / budget_exhausted / no_progress) but the
    run still DELIVERED a substantive final answer, base = *delivered_base*
    (decision #052: the guard stops the loop, not the answer).  Everything else
    — a genuine ``error``, an interrupted run with no ``loop_end``, a guard
    stop with no answer — bases at 0.0.  Then a latency penalty
    ``min(latency_s / budget, 1) * weight`` is subtracted and the result is
    clamped to [0, 1].
    """
    outcome = _run_outcome(events)
    if outcome == "completed":
        base = 1.0
    elif outcome in DELIVERED_OUTCOMES and _has_substantive_final_answer(events):
        base = delivered_base
    else:
        base = 0.0
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
    TraceValidationError).  Aborted noise stubs (no ``loop_end``, fewer than
    ``MIN_ACTIVITY_EVENTS`` events) record no outcome and are skipped too, so
    they cannot depress the aggregate quality signal.  A real interrupted run
    (no ``loop_end`` but enough activity) still scores 0.0.
    """
    for path in collect_traces(Path(trace_dir)):
        try:
            events = read_trace(path)
        except TraceValidationError:
            continue
        if not is_countable_events(events):
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
        # The flag must agree with what the score means (decision #052): a run
        # that scored above zero DID the task, including a guard-terminated run
        # that still delivered its answer.  A ``score > 0`` test is exactly the
        # base>0 condition of ``score_run`` (the latency penalty never reaches
        # the full base), so the two can never contradict each other.
        success = score > 0.0
        latency = _run_latency_s(events)
        self._metrics.record(ExecutionMetric(score=score, success=success, latency=latency))
        return score

    def record_trace_file(self, path: str | os.PathLike[str]) -> float | None:
        """Score and record a single trace file; None if unreadable or a stub."""
        try:
            events = read_trace(Path(path))
        except TraceValidationError:
            return None
        if not is_countable_events(events):
            return None
        return self.record_trace(events)

    def load_corpus(self, trace_dir: str | os.PathLike[str]) -> int:
        """Record every countable trace in *trace_dir*. Returns run count.

        Iterates the trace files directly (not via a reconstructed
        ``task_id``-derived path, which silently skipped any trace whose
        recorded ``task_id`` differs from its filename stem).  Aborted noise
        stubs are excluded, matching :func:`iter_run_scores`.
        """
        count = 0
        for path in collect_traces(Path(trace_dir)):
            try:
                events = read_trace(path)
            except TraceValidationError:
                continue
            if not is_countable_events(events):
                continue
            self.record_trace(events)
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
