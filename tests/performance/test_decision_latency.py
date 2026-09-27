"""Benchmarks for decision latency measurements.

Measures:
- Baseline check_contradictions latency
- check_contradictions_fast latency with pre-filtering
- check_contradictions_fast with knowledge graph
"""

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from agent_core.decisions import (
    build_category_index,
    build_decision_graph,
    check_contradictions,
    check_contradictions_cached,
    check_contradictions_fast,
    clear_contradiction_cache,
    find_overlaps,
)

if TYPE_CHECKING:
    pass


def _make_decisions(count: int = 50) -> list[dict]:
    return [
        {
            "id": str(i).zfill(3),
            "title": f"Decision {i}",
            "decision": f"Decision {i}",
            "rationale": f"Rationale {i}",
            "affected_files": [f"agent_core/file{i}.py"],
            "tags": ["tag1", "tag2"],
        }
        for i in range(1, count + 1)
    ]


@pytest.fixture
def mock_agent():
    """Create a mock Agent for testing."""
    from unittest.mock import AsyncMock, MagicMock

    agent = MagicMock(spec=["llm"])
    agent.llm.chat = AsyncMock(return_value="No contradictions found.")
    return agent


@pytest.fixture
def workspace_with_decisions(tmp_path: Path) -> Path:
    """A real workspace containing 50 decisions and their referenced files.

    check_contradictions_fast loads decisions from the workspace and
    load_decisions drops affected_files entries that do not exist on disk,
    so both the decisions file and the referenced files must be created.
    """
    for i in range(1, 51):
        p = tmp_path / "agent_core" / f"file{i}.py"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# sample\n", encoding="utf-8")
    (tmp_path / ".decisions.json").write_text(
        json.dumps(_make_decisions(), indent=2), encoding="utf-8"
    )
    return tmp_path


@pytest.fixture(autouse=True)
def _clear_contradiction_cache():
    """Keep the module-level cache from leaking across benchmarks."""
    clear_contradiction_cache()
    yield
    clear_contradiction_cache()


class TestCheckContradictionsLatency:
    """Benchmark baseline check_contradictions function."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_benchmark_check_contradictions(self, mock_agent):
        """Measure baseline check_contradictions latency."""
        # Create a large list of decisions
        decisions = [
            {
                "id": str(i).zfill(3),
                "title": f"Decision {i}",
                "decision": f"Decision {i}",
                "rationale": f"Rationale {i}",
                "affected_files": [f"agent_core/file{i}.py"],
                "tags": ["tag1", "tag2"],
            }
            for i in range(1, 51)  # 50 decisions
        ]

        new_decision_text = "New decision to check against all 50 decisions."

        # Warm up
        await check_contradictions(mock_agent, decisions, new_decision_text)

        # Measure
        iterations = 5
        latencies = []

        for _ in range(iterations):
            start = time.perf_counter()
            await check_contradictions(mock_agent, decisions, new_decision_text)
            end = time.perf_counter()
            latencies.append(end - start)

        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)

        print("\nBaseline check_contradictions (50 decisions):")
        print(f"  Average: {avg_latency:.3f}s")
        print(f"  Min: {min_latency:.3f}s")
        print(f"  Max: {max_latency:.3f}s")

        # Assert performance is reasonable (< 5s)
        assert avg_latency < 5.0, (
            f"Baseline check_contradictions too slow: {avg_latency:.3f}s"
        )

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_benchmark_check_contradictions_cached(self, mock_agent):
        """Measure check_contradictions_cached latency with caching."""
        # Create a large list of decisions
        decisions = _make_decisions(50)

        new_decision_text = "New decision to check against all 50 decisions."

        # Warm up
        await check_contradictions_cached(mock_agent, decisions, new_decision_text)

        # Measure (first call - cache miss, subsequent - cache hit)
        iterations = 6
        latencies = []

        for _ in range(iterations):
            start = time.perf_counter()
            await check_contradictions_cached(mock_agent, decisions, new_decision_text)
            end = time.perf_counter()
            latencies.append(end - start)

        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)

        print("\ncheck_contradictions_cached (50 decisions):")
        print(f"  Average: {avg_latency:.3f}s")
        print(f"  Min: {min_latency:.3f}s")
        print(f"  Max: {max_latency:.3f}s")

        # Warm-up call above must have populated the cache: only 1 LLM call
        # across all 7 invocations.
        assert mock_agent.llm.chat.await_count == 1

        # Assert performance is reasonable (< 5s)
        assert avg_latency < 5.0, (
            f"check_contradictions_cached too slow: {avg_latency:.3f}s"
        )


class TestCheckContradictionsFastLatency:
    """Benchmark check_contradictions_fast with pre-filtering."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_benchmark_check_contradictions_fast_no_overlap(
        self, mock_agent, workspace_with_decisions
    ):
        """Measure check_contradictions_fast latency when no overlap found."""
        # New decision with no overlap
        new_decision = {
            "tags": ["new", "feature"],
            "affected_files": ["agent_core/new_feature.py"]
        }
        new_decision_text = "New decision with no overlap."

        # Warm up
        await check_contradictions_fast(
            mock_agent, new_decision, new_decision_text, workspace_with_decisions
        )

        # Measure
        iterations = 5
        latencies = []

        for _ in range(iterations):
            start = time.perf_counter()
            await check_contradictions_fast(
                mock_agent, new_decision, new_decision_text, workspace_with_decisions
            )
            end = time.perf_counter()
            latencies.append(end - start)

        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)

        print("\ncheck_contradictions_fast (no overlap, 50 decisions):")
        print(f"  Average: {avg_latency:.3f}s")
        print(f"  Min: {min_latency:.3f}s")
        print(f"  Max: {max_latency:.3f}s")

        # Should be very fast (early return < 0.5s)
        assert avg_latency < 0.5, (
            f"check_contradictions_fast too slow: {avg_latency:.3f}s"
        )

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_benchmark_check_contradictions_fast_with_overlap(
        self, mock_agent, workspace_with_decisions
    ):
        """Measure check_contradictions_fast latency when overlap found."""
        # New decision with overlap
        new_decision = {
            "tags": ["tag1"],  # Overlaps with all decisions
            "affected_files": ["agent_core/file1.py"]  # Overlaps with decision 1
        }
        new_decision_text = "New decision with overlap."

        # Warm up
        await check_contradictions_fast(
            mock_agent, new_decision, new_decision_text, workspace_with_decisions
        )

        # Measure
        iterations = 5
        latencies = []

        for _ in range(iterations):
            start = time.perf_counter()
            await check_contradictions_fast(
                mock_agent, new_decision, new_decision_text, workspace_with_decisions
            )
            end = time.perf_counter()
            latencies.append(end - start)

        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)

        print("\ncheck_contradictions_fast (with overlap, 50 decisions):")
        print(f"  Average: {avg_latency:.3f}s")
        print(f"  Min: {min_latency:.3f}s")
        print(f"  Max: {max_latency:.3f}s")

        # Overlap path must actually reach the LLM (not early-return)
        assert mock_agent.llm.chat.await_count > 0

        # Should be faster than baseline but slower than no overlap
        assert avg_latency < 2.0, (
            f"check_contradictions_fast too slow: {avg_latency:.3f}s"
        )


class TestDecisionGraphLatency:
    """Benchmark decision graph operations."""

    @pytest.mark.slow
    def test_benchmark_build_decision_graph(self):
        """Measure decision graph building performance."""
        # Create 50 decisions
        decisions = [
            {
                "id": str(i).zfill(3),
                "title": f"Decision {i}",
                "decision": f"Decision {i}",
                "rationale": f"Rationale {i}",
                "affected_files": [f"agent_core/file{i}.py"],
                "tags": ["tag1", "tag2"],
            }
            for i in range(1, 51)  # 50 decisions
        ]

        # Warm up
        build_decision_graph(decisions)

        # Measure
        iterations = 10
        times = []

        for _ in range(iterations):
            start = time.perf_counter()
            build_decision_graph(decisions)
            end = time.perf_counter()
            times.append(end - start)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print("\nbuild_decision_graph (50 decisions):")
        print(f"  Average: {avg_time*1000:.3f}ms")
        print(f"  Min: {min_time*1000:.3f}ms")
        print(f"  Max: {max_time*1000:.3f}ms")

        # Should be very fast (< 100ms)
        assert avg_time < 0.1, f"build_decision_graph too slow: {avg_time*1000:.3f}ms"

    @pytest.mark.slow
    def test_benchmark_build_category_index(self):
        """Measure category index building performance."""
        # Create 50 decisions
        decisions = [
            {
                "id": str(i).zfill(3),
                "title": f"Decision {i}",
                "decision": f"Decision {i}",
                "rationale": f"Rationale {i}",
                "affected_files": [f"agent_core/file{i}.py"],
                "tags": ["tag1", "tag2"],
                "category": ["architecture", "testing"][i % 2],
            }
            for i in range(1, 51)  # 50 decisions
        ]

        # Warm up
        build_category_index(decisions)

        # Measure
        iterations = 10
        times = []

        for _ in range(iterations):
            start = time.perf_counter()
            build_category_index(decisions)
            end = time.perf_counter()
            times.append(end - start)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print("\nbuild_category_index (50 decisions):")
        print(f"  Average: {avg_time*1000:.3f}ms")
        print(f"  Min: {min_time*1000:.3f}ms")
        print(f"  Max: {max_time*1000:.3f}ms")

        # Should be very fast (< 100ms)
        assert avg_time < 0.1, f"build_category_index too slow: {avg_time*1000:.3f}ms"


class TestFindOverlapsLatency:
    """Benchmark find_overlaps filtering performance."""

    @pytest.mark.slow
    def test_benchmark_find_overlaps_fast(self):
        """Measure find_overlaps filtering performance."""
        # Create 50 decisions
        decisions = [
            {
                "id": str(i).zfill(3),
                "title": f"Decision {i}",
                "decision": f"Decision {i}",
                "rationale": f"Rationale {i}",
                "affected_files": [f"agent_core/file{i}.py"],
                "tags": ["tag1", "tag2"],
            }
            for i in range(1, 51)  # 50 decisions
        ]

        new_decision = {
            "tags": ["tag1"],
            "affected_files": ["agent_core/file1.py"]
        }

        # Warm up
        find_overlaps(new_decision, decisions, "/workspace")

        # Measure
        iterations = 10
        times = []

        for _ in range(iterations):
            start = time.perf_counter()
            find_overlaps(new_decision, decisions, "/workspace")
            end = time.perf_counter()
            times.append(end - start)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print("\nfind_overlaps (50 decisions, with overlap):")
        print(f"  Average: {avg_time*1000:.3f}ms")
        print(f"  Min: {min_time*1000:.3f}ms")
        print(f"  Max: {max_time*1000:.3f}ms")

        # Should be very fast (< 10ms)
        assert avg_time < 0.01, f"find_overlaps too slow: {avg_time*1000:.3f}ms"
