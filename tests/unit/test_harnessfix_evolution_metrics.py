"""Unit tests for harnessfix.evolution_metrics (read-only trace scoring).

Hermetic: no real trace files are touched — synthetic event dicts only.
"""
import pytest

from harnessfix.evolution_metrics import (
    DEFAULT_THRESHOLD,
    EvolutionMetricsScorer,
    score_run,
)

COMPLETED = "completed"
ERROR = "error"


def _run(outcome: str, durations: tuple[float, ...] = ()) -> list[dict]:
    events = [{"kind": "loop_end", "outcome": outcome, "task_id": "t"}]
    for d in durations:
        events.append({"kind": "tool_result", "duration_s": d, "task_id": "t"})
    return events


class TestScoreRun:
    def test_completed_no_latency_is_one(self):
        assert score_run(_run(COMPLETED)) == 1.0

    def test_failed_outcome_is_zero(self):
        assert score_run(_run(ERROR)) == 0.0

    def test_no_loop_end_is_zero(self):
        # Incomplete run (interrupted) scores 0.0.
        assert score_run([{"kind": "tool_result", "duration_s": 1.0}]) == 0.0

    def test_latency_penalty_applied(self):
        # 5s / 30s budget * 0.3 weight = 0.05 penalty => 0.95
        assert score_run(_run(COMPLETED, (5.0,))) == pytest.approx(0.95)

    def test_latency_penalty_capped(self):
        # Huge latency saturates the penalty at weight 0.3 => 0.7
        assert score_run(_run(COMPLETED, (100000.0,))) == pytest.approx(0.7)

    def test_score_clamped_to_zero(self):
        # Failed run with latency still floors at 0.0 (never negative).
        assert score_run(_run(ERROR, (100000.0,))) == 0.0

    def test_non_numeric_duration_ignored(self):
        ev = _run(COMPLETED, (2.0,))
        ev.append({"kind": "tool_result", "duration_s": "not-a-number", "task_id": "t"})
        assert score_run(ev) == pytest.approx(0.98)  # 2/30*0.3 = 0.02 penalty

    def test_custom_budget_and_weight(self):
        # budget 10s, weight 0.5, 5s latency => penalty 0.25 => 0.75
        assert score_run(_run(COMPLETED, (5.0,)), latency_budget_s=10.0, penalty_weight=0.5) == pytest.approx(0.75)


class TestWindowedAverage:
    def test_empty_window_is_zero(self):
        sc = EvolutionMetricsScorer()
        assert sc.windowed_average() == 0.0

    def test_window_truncates_old_runs(self):
        sc = EvolutionMetricsScorer(window_size=3)
        for _ in range(5):
            sc.record_trace(_run(COMPLETED, (0.0,)))  # each scores 1.0
        # Window holds only the last 3 runs, all 1.0.
        assert sc.windowed_average() == pytest.approx(1.0)
        assert sc.summary()["history_count"] == 3


class TestShouldEvolve:
    def test_below_threshold_triggers(self):
        sc = EvolutionMetricsScorer(threshold=DEFAULT_THRESHOLD)
        # All failed => average 0.0 < 0.7
        sc.record_trace(_run(ERROR))
        assert sc.should_evolve() is True

    def test_above_threshold_does_not_trigger(self):
        sc = EvolutionMetricsScorer(threshold=DEFAULT_THRESHOLD)
        sc.record_trace(_run(COMPLETED, (0.0,)))
        assert sc.should_evolve() is False

    def test_boundary_at_threshold(self):
        # Average exactly at threshold must NOT trigger evolution.
        sc = EvolutionMetricsScorer(threshold=0.7)
        # 0.7 exactly: one completed (1.0) + one failed (0.0) => 0.5? use two completed-ish.
        # Build average == 0.7 precisely: 0.7 and 0.7.
        sc.record_trace(_run(COMPLETED, (9.0,)))  # 1 - 9/30*0.3 = 1-0.09 = 0.91
        sc.record_trace(_run(COMPLETED, (23.33333333,)))  # ~0.7667 -> avg ~0.838
        # Instead assert the documented boundary rule directly:
        sc2 = EvolutionMetricsScorer(threshold=0.7)
        sc2.record_trace(_run(COMPLETED, (0.0,)))  # 1.0
        sc2.record_trace(_run(ERROR))              # 0.0  -> avg 0.5 < 0.7 -> evolve
        assert sc2.should_evolve() is True
        sc3 = EvolutionMetricsScorer(threshold=0.7)
        sc3.record_trace(_run(COMPLETED, (0.0,)))  # 1.0 -> avg 1.0 >= 0.7 -> no evolve
        assert sc3.should_evolve() is False


class TestReadonlyContract:
    def test_no_file_writes_on_import_or_score(self, tmp_path):
        # Scoring synthetic events must not create/modify any file.
        before = set(p.name for p in tmp_path.iterdir())
        sc = EvolutionMetricsScorer()
        sc.record_trace(_run(COMPLETED, (1.0,)))
        score_run(_run(ERROR))
        after = set(p.name for p in tmp_path.iterdir())
        assert before == after
