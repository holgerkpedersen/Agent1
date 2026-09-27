"""Unit tests for decision performance optimizations.

Tests for:
- find_overlaps fast filtering
- check_contradictions_fast with pre-filtering
- Decision graph building
- Category index building
- Cache hit behavior
"""

from pathlib import Path
from typing import TYPE_CHECKING

import json

import pytest

from agent_core.decisions import (
    build_category_index,
    build_decision_graph,
    check_contradictions_cached,
    check_contradictions_fast,
    clear_contradiction_cache,
    find_overlaps,
)

if TYPE_CHECKING:
    pass


@pytest.fixture
def sample_decisions() -> list[dict[str, any]]:
    """Sample decisions with various overlaps."""
    return [
        {
            "id": "001",
            "title": "Use async/await for API calls",
            "decision": "All API calls should use async/await",
            "rationale": "Better performance and concurrency",
            "affected_files": ["agent_core/api_client.py"],
            "tags": ["concurrency", "performance"],
            "category": "architecture",
        },
        {
            "id": "002",
            "title": "Use SQLAlchemy ORM",
            "decision": "Use SQLAlchemy for database access",
            "rationale": "Type safety and query builder",
            "affected_files": ["agent_core/database.py"],
            "tags": ["database", "orm"],
            "category": "database",
        },
        {
            "id": "003",
            "title": "Use Pydantic for validation",
            "decision": "Use Pydantic for all input validation",
            "rationale": "Type safety and validation",
            "affected_files": ["agent_core/api_client.py"],
            "tags": ["validation"],
            "category": "testing",
        },
        {
            "id": "004",
            "title": "Use pytest for testing",
            "decision": "Use pytest as testing framework",
            "rationale": "Better features and community",
            "affected_files": ["tests/test_api.py"],
            "tags": ["testing", "pytest"],
            "category": "testing",
        },
        {
            "id": "005",
            "title": "Use black for formatting",
            "decision": "Use black for code formatting",
            "rationale": "Consistent style",
            "affected_files": ["agent_core/core.py"],
            "tags": ["formatting"],
            "category": "quality",
        },
    ]


class TestFindOverlapsFast:
    """Test the fast pre-filtering using find_overlaps."""

    def test_find_overlaps_no_overlap(self, sample_decisions):
        """Test that no overlaps returns empty list."""
        new_decision = {
            "tags": ["new", "feature"],
            "affected_files": ["agent_core/new_feature.py"]
        }
        overlaps = find_overlaps(new_decision, sample_decisions, "/workspace")
        assert overlaps == []

    def test_find_overlaps_file_overlap(self, sample_decisions):
        """Test that file overlap is detected."""
        new_decision = {
            "tags": [],
            "affected_files": ["agent_core/api_client.py"]
        }
        overlaps = find_overlaps(new_decision, sample_decisions, "/workspace")
        # Should match decision 001 and 003 (both use api_client.py)
        assert len(overlaps) == 2
        overlap_ids = {d["id"] for d in overlaps}
        assert overlap_ids == {"001", "003"}

    def test_find_overlaps_tag_overlap(self, sample_decisions):
        """Test that tag overlap is detected."""
        new_decision = {
            "tags": ["concurrency"],
            "affected_files": []
        }
        overlaps = find_overlaps(new_decision, sample_decisions, "/workspace")
        # Should match decision 001
        assert len(overlaps) == 1
        assert overlaps[0]["id"] == "001"

    def test_find_overlaps_both_overlap(self, sample_decisions):
        """Test that both file and tag overlap is detected."""
        new_decision = {
            "tags": ["testing"],
            "affected_files": ["tests/test_api.py"]
        }
        overlaps = find_overlaps(new_decision, sample_decisions, "/workspace")
        # Should match decision 004
        assert len(overlaps) == 1
        assert overlaps[0]["id"] == "004"

    def test_find_overlaps_multiple_matches(self, sample_decisions):
        """Test that multiple matches are returned."""
        new_decision = {
            "tags": ["testing"],
            "affected_files": []
        }
        overlaps = find_overlaps(new_decision, sample_decisions, "/workspace")
        # Only decision 004 has "testing" tag
        assert len(overlaps) == 1
        assert overlaps[0]["id"] == "004"


class TestCheckContradictionsFast:
    """Test the fast contradiction check function."""

    @pytest.fixture
    def workspace(self, tmp_path: Path, sample_decisions) -> Path:
        """A real workspace containing the sample decisions and their files.

        ``load_decisions`` normalizes ``affected_files`` and drops entries
        that do not exist, so the referenced files must be created for
        file-based overlap detection to work.
        """
        for d in sample_decisions:
            for f in d["affected_files"]:
                p = tmp_path / f
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("# sample\n", encoding="utf-8")
        (tmp_path / ".decisions.json").write_text(
            json.dumps(sample_decisions), encoding="utf-8"
        )
        return tmp_path

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        clear_contradiction_cache()
        yield
        clear_contradiction_cache()

    @pytest.mark.asyncio
    async def test_check_contradictions_fast_no_overlap(
        self, sample_decisions, mock_agent, workspace
    ):
        """Test that fast check returns early when no overlaps found."""
        new_decision = {
            "tags": ["new", "feature"],
            "affected_files": ["agent_core/new_feature.py"]
        }
        result = await check_contradictions_fast(
            mock_agent,
            new_decision,
            "New feature X",
            workspace
        )
        # Should not call LLM when no overlaps
        assert "No similar decisions found" in result
        mock_agent.llm.chat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_check_contradictions_fast_with_overlap(
        self, sample_decisions, mock_agent, workspace
    ):
        """Test that fast check calls the LLM only on overlapping decisions."""
        new_decision = {
            "tags": ["testing"],
            "affected_files": []
        }
        # This should only check decision 004 (tag "testing")
        result = await check_contradictions_fast(
            mock_agent,
            new_decision,
            "New testing approach",
            workspace
        )
        # LLM should have been called (not early returned)
        assert "contradiction" in result.lower() or "no contradiction" in result.lower()
        mock_agent.llm.chat.assert_awaited_once()
        # Pre-filtering: only decision 004 should be in the prompt
        prompt = mock_agent.llm.chat.await_args.args[0][1]["content"]
        assert "#004" in prompt
        assert "#001" not in prompt

    @pytest.mark.asyncio
    async def test_check_contradictions_fast_cache_hit(
        self, sample_decisions, mock_agent, workspace
    ):
        """A second identical fast check must be served from the cache."""
        new_decision = {"tags": ["testing"], "affected_files": []}
        await check_contradictions_fast(
            mock_agent, new_decision, "New testing approach", workspace
        )
        await check_contradictions_fast(
            mock_agent, new_decision, "New testing approach", workspace
        )
        assert mock_agent.llm.chat.await_count == 1


class TestDecisionGraph:
    """Test the decision graph builder."""

    def test_build_decision_graph_basic(self, sample_decisions):
        """Test basic graph structure."""
        graph = build_decision_graph(sample_decisions)

        # Check structure
        assert "decisions" in graph
        assert "files" in graph
        assert "tags" in graph
        # symbols key is not in our implementation, but it's okay if it exists
        # with empty dict
        if "symbols" in graph:
            assert graph["symbols"] == {}

        # Check decisions
        assert len(graph["decisions"]) == 5
        for d in sample_decisions:
            assert d["id"] in graph["decisions"]

        # Check file relationships
        assert "agent_core/api_client.py" in graph["files"]
        assert graph["files"]["agent_core/api_client.py"] == ["001", "003"]

        # Check tag relationships
        assert "concurrency" in graph["tags"]
        assert graph["tags"]["concurrency"] == ["001"]

    def test_build_decision_graph_empty(self):
        """Test graph building with no decisions."""
        graph = build_decision_graph([])
        assert graph["decisions"] == {}
        assert graph["files"] == {}
        assert graph["tags"] == {}


class TestCategoryIndex:
    """Test the category index builder."""

    def test_build_category_index_basic(self, sample_decisions):
        """Test basic index structure."""
        index = build_category_index(sample_decisions)

        # Check structure
        assert "architecture" in index
        assert "database" in index
        assert "testing" in index
        assert "quality" in index

        # Check decision IDs
        assert index["architecture"] == ["001"]
        assert index["database"] == ["002"]
        assert index["testing"] == ["003", "004"]
        assert index["quality"] == ["005"]

    def test_build_category_index_uncategorized(self, sample_decisions):
        """Test that uncategorized decisions are indexed with empty string."""
        new_decision = {
            "id": "006",
            "title": "Uncategorized decision",
            "decision": "Test",
            "rationale": "Test",
            "affected_files": ["test.py"],
            "tags": [],
            "category": "",
        }
        index = build_category_index(sample_decisions + [new_decision])
        # Empty category is stored as empty string
        assert "" in index
        assert "006" in index[""]

    def test_build_category_index_case_insensitive(self, sample_decisions):
        """Test that categories are normalized to lowercase."""
        new_decision = {
            "id": "007",
            "title": "Test",
            "decision": "Test",
            "rationale": "Test",
            "affected_files": ["test.py"],
            "tags": [],
            "category": "Architecture",  # Capitalized
        }
        index = build_category_index(sample_decisions + [new_decision])
        assert "architecture" in index
        assert "007" in index["architecture"]


class TestCacheBehavior:
    """Test the caching behavior of contradiction checks."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        """Isolate each test from the module-level cache."""
        clear_contradiction_cache()
        yield
        clear_contradiction_cache()

    @pytest.mark.asyncio
    async def test_cache_hit_same_input(self, mock_agent, sample_decisions):
        """Test that same input returns cached result (LLM called once)."""
        # First call - should cache
        result1 = await check_contradictions_cached(
            mock_agent,
            sample_decisions[:2],
            "Test decision",
        )

        # Second call - should hit cache, not call the LLM again
        result2 = await check_contradictions_cached(
            mock_agent,
            sample_decisions[:2],
            "Test decision",
        )

        # Both should return the same result and the LLM ran exactly once
        assert result1 == result2
        assert mock_agent.llm.chat.await_count == 1

    @pytest.mark.asyncio
    async def test_cache_miss_different_input(self, mock_agent, sample_decisions):
        """Test that different input doesn't hit cache."""
        result1 = await check_contradictions_cached(
            mock_agent,
            sample_decisions[:2],
            "Test decision 1",
        )

        result2 = await check_contradictions_cached(
            mock_agent,
            sample_decisions[:2],
            "Test decision 2",
        )

        # Results may differ, but both should be computed
        assert result1 is not None
        assert result2 is not None
        assert mock_agent.llm.chat.await_count == 2

    @pytest.mark.asyncio
    async def test_cache_miss_different_decisions(self, mock_agent, sample_decisions):
        """Different decision sets must not share a cache entry."""
        await check_contradictions_cached(
            mock_agent, sample_decisions[:2], "Test decision"
        )
        await check_contradictions_cached(
            mock_agent, sample_decisions[2:4], "Test decision"
        )
        assert mock_agent.llm.chat.await_count == 2


# Mock fixtures
@pytest.fixture
def mock_agent():
    """Create a mock Agent for testing."""
    from unittest.mock import AsyncMock, MagicMock

    agent = MagicMock(spec=["llm", "workspace"])
    agent.llm.chat = AsyncMock(return_value="No contradictions found.")
    agent.workspace = "/test/workspace"
    return agent
