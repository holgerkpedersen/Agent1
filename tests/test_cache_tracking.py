"""Regression tests for prompt-cache hit/miss tracking in MetricsTracker and PromptCache."""
from __future__ import annotations

import pytest

from agent_core.llm.metrics_tracker import MetricsTracker
from agent_core.llm.prompt_cache import PromptCache


class TestMetricsTrackerCache:
    """Tests for the new cache-hit/miss counters on MetricsTracker."""

    def test_record_hit_and_miss(self):
        m = MetricsTracker()
        assert m.get_all_cache_metrics() == {}
        m.record_cache_hit("fix", "fast_codegen")
        m.record_cache_hit("fix", "fast_codegen")
        m.record_cache_miss("fix", "fast_codegen")

        metrics = m.get_cache_metrics(
            __import__("agent_core.llm.llm_types", fromlist=["TaskType"]).TaskType.FIX,
            __import__("agent_core.llm.llm_types", fromlist=["ProfileType"]).ProfileType.FAST_CODEGEN,
        )
        assert metrics["hits"] == 2
        assert metrics["misses"] == 1
        assert metrics["total_lookups"] == 3
        assert metrics["hit_rate_pct"] == pytest.approx(66.67)

    def test_get_all_cache_metrics_multiple_keys(self):
        m = MetricsTracker()
        m.record_cache_hit("fix", "fast_codegen")
        m.record_cache_miss("implement", "deep_analysis")
        all_m = m.get_all_cache_metrics()
        assert len(all_m) == 2

    def test_reset_clears_cache_stats(self):
        m = MetricsTracker()
        m.record_cache_hit("fix", "fast_codegen")
        m.reset()
        assert m._cache_stats == {}

    def test_empty_cache_returns_zero_hit_rate(self):
        m = MetricsTracker()
        metrics = m.get_cache_metrics(
            __import__("agent_core.llm.llm_types", fromlist=["TaskType"]).TaskType.FIX,
            __import__("agent_core.llm.llm_types", fromlist=["ProfileType"]).ProfileType.FAST_CODEGEN,
        )
        assert metrics == {}


class TestPromptCacheStats:
    """Tests for the new get_stats() method on PromptCache."""

    def test_initial_stats_empty(self):
        pc = PromptCache()
        stats = pc.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["total_lookups"] == 0
        assert stats["hit_rate_pct"] == 0.0

    def test_hits_and_misses_increment(self):
        pc = PromptCache()
        # Trigger a miss (no templates saved)
        pc.get_template("fix", "fast_codegen")
        assert pc._misses == 1
        # Same key again -> hit
        pc.get_template("fix", "fast_codegen")
        assert pc._hits == 1

    def test_stats_after_hits_and_misses(self):
        pc = PromptCache()
        pc.get_template("fix", "a")   # miss
        pc.get_template("fix", "b")   # miss
        pc.get_template("fix", "a")   # hit
        stats = pc.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["total_lookups"] == 3
