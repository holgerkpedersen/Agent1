(
    """Regression tests for prompt-cache hit/miss """
    """tracking in MetricsTracker and PromptCache."""
)

from __future__ import annotations

import pytest

from agent_core.llm.metrics_tracker import MetricsTracker
from agent_core.llm.prompt_cache import (
    PromptCache,
    get_prompt_cache,
    get_prompt_cache_stats,
    reset_prompt_cache,
    set_current_command,
)


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
            __import__(
                "agent_core.llm.llm_types", fromlist=["ProfileType"]
            ).ProfileType.FAST_CODEGEN,
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
            __import__(
                "agent_core.llm.llm_types", fromlist=["ProfileType"]
            ).ProfileType.FAST_CODEGEN,
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
        pc.get_template("fix", "a")  # miss
        pc.get_template("fix", "b")  # miss
        pc.get_template("fix", "a")  # hit
        stats = pc.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["total_lookups"] == 3

    def test_per_command_stats_attributed_when_context_set(self):
        pc = PromptCache()
        prev = set_current_command("fix")
        try:
            pc.get_template("fix", "a")  # miss, attributed to fix
            pc.get_template("fix", "a")  # hit, attributed to fix
            set_current_command("analyze")
            pc.get_template("implement", "b")  # miss, attributed to analyze
        finally:
            set_current_command(prev)

        by_cmd = pc.get_command_stats()
        assert by_cmd["fix"]["hits"] == 1
        assert by_cmd["fix"]["misses"] == 1
        assert by_cmd["fix"]["hit_rate_pct"] == 50.0
        assert by_cmd["analyze"]["misses"] == 1
        assert by_cmd["analyze"]["hits"] == 0

    def test_no_attribution_without_command_context(self):
        pc = PromptCache()
        prev = set_current_command(None)
        try:
            pc.get_template("fix", "x")
            pc.get_template("fix", "x")
        finally:
            set_current_command(prev)
        assert pc.get_command_stats() == {}
        assert pc.get_stats()["total_lookups"] == 2

    def test_reset_stats_keeps_cached_templates(self):
        pc = PromptCache()
        prev = set_current_command("fix")
        try:
            pc.get_template("fix", "a")  # miss + cache fill
            pc.get_template("fix", "a")  # hit
        finally:
            set_current_command(prev)
        pc.reset_stats()
        assert pc.get_stats() == {
            "hits": 0,
            "misses": 0,
            "total_lookups": 0,
            "hit_rate_pct": 0.0,
        }
        assert pc.get_command_stats() == {}
        # Cached template still serves a hit (new stats count it)
        prev = set_current_command(None)
        try:
            pc.get_template("fix", "a")
        finally:
            set_current_command(prev)
        assert pc.get_stats()["hits"] == 1


class TestPromptCacheSingleton:
    """Process-wide singleton used by the perf view/dashboard."""

    def test_singleton_stats_shape(self):
        reset_prompt_cache()
        try:
            data = get_prompt_cache_stats()
            assert "overall" in data
            assert "by_command" in data
            assert data["overall"]["hit_rate_pct"] == 0.0
        finally:
            reset_prompt_cache()

    def test_singleton_records_command_attribution(self):
        reset_prompt_cache()
        prev = set_current_command("workflow")
        try:
            pc = get_prompt_cache()
            pc.get_template("fix", "z")
            pc.get_template("fix", "z")
        finally:
            set_current_command(prev)
            reset_prompt_cache()
            # Re-check via a fresh singleton to avoid leaking state: the calls
            # above populated the singleton we just reset; verify the API shape.
        reset_prompt_cache()
        prev = set_current_command("workflow")
        try:
            pc = get_prompt_cache()
            pc.get_template("fix", "z")
            pc.get_template("fix", "z")
            data = get_prompt_cache_stats()
            assert data["overall"]["total_lookups"] == 2
            assert data["by_command"]["workflow"]["hit_rate_pct"] == 50.0
        finally:
            set_current_command(prev)
            reset_prompt_cache()
