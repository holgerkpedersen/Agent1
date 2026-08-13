"""Tests for agent1.evolution.metrics — ExecutionMetric and EvolutionMetrics."""

from __future__ import annotations

import pytest

from agent1.evolution.metrics import ExecutionMetric, EvolutionMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metric(score: float = 0.5, success: bool = True, latency: float = 0.1) -> ExecutionMetric:
    return ExecutionMetric(score=score, success=success, latency=latency)


# ---------------------------------------------------------------------------
# Construction & defaults
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_window_size(self) -> None:
        m = EvolutionMetrics()
        assert m.window_size == 10

    def test_default_threshold(self) -> None:
        m = EvolutionMetrics()
        assert m.threshold == pytest.approx(0.75)

    def test_custom_params(self) -> None:
        m = EvolutionMetrics(window_size=5, threshold=0.6)
        assert m.window_size == 5
        assert m.threshold == pytest.approx(0.6)

    def test_history_starts_empty(self) -> None:
        m = EvolutionMetrics()
        assert m.recent_metrics() == []


# ---------------------------------------------------------------------------
# ExecutionMetric
# ---------------------------------------------------------------------------

class TestExecutionMetric:
    def test_attributes_stored(self) -> None:
        metric = _metric(score=0.8, success=False, latency=1.2)
        assert metric.score == pytest.approx(0.8)
        assert metric.success is False
        assert metric.latency == pytest.approx(1.2)

    def test_defaults_via_helper(self) -> None:
        metric = _metric()
        assert metric.score == pytest.approx(0.5)
        assert metric.success is True
        assert metric.latency == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# record — sliding window behaviour
# ---------------------------------------------------------------------------

class TestRecord:
    def test_appends_metric(self) -> None:
        m = EvolutionMetrics()
        em = _metric(score=0.9)
        m.record(em)
        assert m.recent_metrics() == [em]

    def test_multiple_records_preserve_order(self) -> None:
        m = EvolutionMetrics()
        a, b, c = _metric(0.1), _metric(0.2), _metric(0.3)
        m.record(a); m.record(b); m.record(c)
        assert m.recent_metrics() == [a, b, c]

    def test_window_trims_oldest(self) -> None:
        """When window_size exceeded, oldest metric is dropped."""
        m = EvolutionMetrics(window_size=3)
        a, b, c, d = _metric(0.1), _metric(0.2), _metric(0.3), _metric(0.4)
        for em in (a, b, c, d):
            m.record(em)
        assert m.recent_metrics() == [b, c, d]

    def test_history_count_within_window(self) -> None:
        m = EvolutionMetrics()
        for _ in range(10):
            m.record(_metric())
        assert len(m.recent_metrics()) == 10

    def test_exactly_at_window_boundary_kept(self) -> None:
        """Recording exactly window_size entries keeps all of them."""
        m = EvolutionMetrics(window_size=3)
        for _ in range(3):
            m.record(_metric())
        assert len(m.recent_metrics()) == 3

    def test_empty_after_no_records(self) -> None:
        m = EvolutionMetrics()
        assert history_count_in_summary_is_zero(m)


# ---------------------------------------------------------------------------
# average_score / score
# ---------------------------------------------------------------------------

class TestScores:
    def test_average_of_single_metric(self) -> None:
        m = EvolutionMetrics()
        m.record(_metric(score=0.8))
        assert m.average_score() == pytest.approx(0.8)

    def test_average_multiple_metrics(self) -> None:
        m = EvolutionMetrics()
        for s in (0.5, 0.7, 0.9):
            m.record(_metric(score=s))
        expected = sum((0.5, 0.7, 0.9)) / 3
        assert m.average_score() == pytest.approx(expected)

    def test_average_empty_returns_zero(self) -> None:
        m = EvolutionMetrics()
        assert m.average_score() == pytest.approx(0.0)

    def test_score_returns_most_recent(self) -> None:
        m = EvolutionMetrics()
        m.record(_metric(score=0.3))
        m.record(_metric(score=0.9))
        assert m.score() == pytest.approx(0.9)

    def test_score_empty_returns_zero(self) -> None:
        m = EvolutionMetrics()
        assert m.score() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# should_evolve — threshold logic
# ---------------------------------------------------------------------------

class TestShouldEvolve:
    def test_below_threshold_triggers(self) -> None:
        """Average below threshold → evolution required."""
        m = EvolutionMetrics(window_size=3, threshold=0.75)
        for s in (0.4, 0.5, 0.6):
            m.record(_metric(score=s))
        assert m.should_evolve() is True

    def test_above_threshold_no_evolution(self) -> None:
        """Average above threshold → no evolution."""
        m = EvolutionMetrics(window_size=3, threshold=0.75)
        for s in (0.8, 0.9, 0.9):
            m.record(_metric(score=s))
        assert m.should_evolve() is False

    def test_equal_to_threshold_no_evolution(self) -> None:
        """Average exactly at threshold → no evolution (strict <)."""
        m = EvolutionMetrics(window_size=2, threshold=0.5)
        for s in (0.5, 0.5):
            m.record(_metric(score=s))
        assert m.should_evolve() is False

    def test_empty_history_no_evolution(self) -> None:
        """No history → average 0 < default threshold 0.75 → evolve."""
        m = EvolutionMetrics()
        # average_score == 0.0 which is below the default threshold of 0.75
        assert m.should_evolve() is True

    def test_custom_threshold_boundary(self) -> None:
        """With a low custom threshold, poor scores trigger evolution."""
        m = EvolutionMetrics(window_size=2, threshold=0.1)
        for s in (0.05, 0.05):
            m.record(_metric(score=s))
        assert m.should_evolve() is True


# ---------------------------------------------------------------------------
# summary — snapshot structure & values
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_keys_present(self) -> None:
        m = EvolutionMetrics()
        s = m.summary()
        for key in ("window_size", "threshold", "average_score",
                    "history_count", "should_evolve"):
            assert key in s

    def test_summary_empty_state(self) -> None:
        m = EvolutionMetrics()
        s = m.summary()
        assert s["window_size"] == 10
        assert s["threshold"] == pytest.approx(0.75)
        assert s["average_score"] == pytest.approx(0.0)
        assert s["history_count"] == 0
        # empty → average 0 < threshold → evolve True
        assert s["should_evolve"] is True

    def test_summary_with_records(self) -> None:
        m = EvolutionMetrics(window_size=3, threshold=0.8)
        for s in (0.5, 0.6, 0.7):
            m.record(_metric(score=s))
        snap = m.summary()
        assert snap["window_size"] == 3
        assert snap["threshold"] == pytest.approx(0.8)
        assert snap["average_score"] == pytest.approx((0.5 + 0.6 + 0.7) / 3)
        assert snap["history_count"] == 3
        assert snap["should_evolve"] is True

    def test_summary_history_count_reflects_window(self) -> None:
        m = EvolutionMetrics(window_size=4, threshold=0.5)
        for _ in range(6):
            m.record(_metric(score=0.9))
        snap = m.summary()
        assert snap["history_count"] == 4

    def test_summary_is_independent_dict(self) -> None:
        """Mutating the returned dict must not alter internal state."""
        m = EvolutionMetrics()
        m.record(_metric(score=0.7))
        snap = m.summary()
        snap.clear()
        assert len(m.recent_metrics()) == 1


# ---------------------------------------------------------------------------
# recent_metrics — returns window contents
# ---------------------------------------------------------------------------

class TestRecentMetrics:
    def test_returns_history_list(self) -> None:
        m = EvolutionMetrics()
        a, b = _metric(0.1), _metric(0.2)
        m.record(a); m.record(b)
        assert m.recent_metrics() == [a, b]

    def test_after_trim_only_window_entries(self) -> None:
        m = EvolutionMetrics(window_size=3)
        a, b, c, d = _metric(0.1), _metric(0.2), _metric(0.3), _metric(0.4)
        for em in (a, b, c, d):
            m.record(em)
        assert m.recent_metrics() == [b, c, d]

    def test_empty_returns_empty_list(self) -> None:
        m = EvolutionMetrics()
        assert m.recent_metrics() == []


# ---------------------------------------------------------------------------
# integration — record + should_evolve interplay across window
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_recovery_after_low_scores(self) -> None:
        """Low scores trigger evolution; high scores recover."""
        m = EvolutionMetrics(window_size=3, threshold=0.75)
        # First three poor → evolve True
        for s in (0.4, 0.5, 0.6):
            m.record(_metric(score=s))
        assert m.should_evolve() is True
        # Replace window with high scores → no longer evolving
        for s in (0.9, 0.9, 0.8):
            m.record(_metric(score=s))
        assert m.should_evolve() is False

    def test_latency_and_success_not_affecting_score(self) -> None:
        """score/should_evolve depend only on score attribute."""
        m = EvolutionMetrics(window_size=2, threshold=0.5)
        # High scores but success=False → still above threshold
        for s in (0.9, 0.8):
            m.record(_metric(score=s, success=False))
        assert m.should_evolve() is False

    def test_summary_consistent_with_methods(self) -> None:
        """summary values match the individual accessor methods."""
        m = EvolutionMetrics(window_size=3, threshold=0.75)
        for s in (0.6, 0.7, 0.8):
            m.record(_metric(score=s))
        snap = m.summary()
        assert snap["average_score"] == pytest.approx(m.average_score())
        assert snap["should_evolve"] == m.should_evolve()
        assert snap["history_count"] == len(m.recent_metrics())


# ---------------------------------------------------------------------------
# tiny helper used by TestRecord.test_empty_after_no_records above
# ---------------------------------------------------------------------------

def history_count_in_summary_is_zero(metrics: EvolutionMetrics) -> bool:  # noqa: D401
    """Assert summary reports zero history count when nothing recorded."""
    assert metrics.summary()["history_count"] == 0
    return True
