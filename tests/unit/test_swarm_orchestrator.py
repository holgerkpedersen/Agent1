"""Swarm Orchestrator tests (plan task 39): task dispatch, result retrieval,
completion waiting, and EvolutionMetrics integration."""
import time

import pytest

from agent1.evolution.metrics import EvolutionMetrics, ExecutionMetric
from agent1.swarm.orchestrator import Orchestrator


class TestOrchestrator:
    def test_dispatch_and_get_result(self):
        with Orchestrator(agents=[], max_workers=4) as orch:
            task_id = orch.dispatch(lambda x: {"value": x * 2}, 21)
            assert orch.wait_for_completion(timeout=5) is True
            result = orch.get_result(task_id)
            assert result == {"value": 42}

    def test_get_result_wraps_non_dict(self):
        with Orchestrator(agents=[], max_workers=2) as orch:
            task_id = orch.dispatch(lambda: "plain")
            assert orch.wait_for_completion(timeout=5) is True
            assert orch.get_result(task_id) == {"data": "plain"}

    def test_task_exception_captured_as_error(self):
        with Orchestrator(agents=[], max_workers=2) as orch:
            def boom():
                raise RuntimeError("task exploded")

            task_id = orch.dispatch(boom)
            assert orch.wait_for_completion(timeout=5) is True
            result = orch.get_result(task_id)
            assert result["error"] == "task exploded"

    def test_get_result_before_done_returns_none(self):
        with Orchestrator(agents=[], max_workers=2) as orch:
            task_id = orch.dispatch(lambda: time.sleep(0.05) or {"ok": 1})
            assert orch.get_result(task_id) is None
            assert orch.wait_for_completion(timeout=5) is True

    def test_wait_for_completion_empty(self):
        with Orchestrator(agents=[], max_workers=2) as orch:
            assert orch.wait_for_completion() is True

    def test_unique_task_ids(self):
        with Orchestrator(agents=[], max_workers=4) as orch:
            ids = {orch.dispatch(lambda: 1) for _ in range(5)}
            assert len(ids) == 5


class TestEvolutionMetricsIntegration:
    def test_record_window_and_evolution_decision(self):
        metrics = EvolutionMetrics(window_size=3, threshold=0.75)
        for score in (1.0, 0.8, 0.7):
            metrics.record(ExecutionMetric(score=score, success=True, latency=1.0))
        assert metrics.average_score() == pytest.approx(0.8333, abs=0.01)
        assert metrics.should_evolve() is False

    def test_low_performance_triggers_evolution(self):
        metrics = EvolutionMetrics(window_size=3, threshold=0.75)
        metrics.record(ExecutionMetric(score=0.4, success=False, latency=2.0))
        assert metrics.should_evolve() is True

    def test_window_slides(self):
        metrics = EvolutionMetrics(window_size=2, threshold=0.5)
        for score in (1.0, 1.0, 0.2):
            metrics.record(ExecutionMetric(score=score, success=True, latency=1.0))
        assert len(metrics.recent_metrics()) == 2
        assert metrics.average_score() == 0.6

    def test_empty_metrics(self):
        metrics = EvolutionMetrics()
        assert metrics.average_score() == 0.0
        assert metrics.score() == 0.0
        # Documented behavior: empty history -> average 0 < threshold -> evolve.
        assert metrics.should_evolve() is True

    def test_summary_snapshot(self):
        metrics = EvolutionMetrics()
        metrics.record(ExecutionMetric(score=0.5, success=True, latency=1.0))
        summary = metrics.summary()
        assert summary["average_score"] == 0.5
        assert summary["history_count"] == 1
        assert "threshold" in summary
