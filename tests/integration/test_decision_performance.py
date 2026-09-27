"""Integration tests for decision performance optimizations.

Tests for:
- decide command with many decisions
- decide command with overlap detection
- Decision graph queries
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from agent_core.decisions import (
    build_decision_graph,
    build_category_index,
    find_overlaps,
    load_decisions,
)

if TYPE_CHECKING:
    pass


@pytest.fixture
def workspace_with_decisions(tmp_path: Path) -> Path:
    """Create a workspace with sample decisions."""
    decisions_file = tmp_path / ".decisions.json"

    decisions = [
        {
            "id": "001",
            "date": "2024-01-01T00:00:00Z",
            "title": "Use async/await for API calls",
            "context": "Performance requirements",
            "decision": "All API calls should use async/await",
            "rationale": "Better performance and concurrency",
            "affected_files": ["agent_core/api_client.py"],
            "tags": ["concurrency", "performance"],
            "contradictions": [],
            "resolved_by": None,
            "status": "active",
            "category": "architecture",
        },
        {
            "id": "002",
            "date": "2024-01-02T00:00:00Z",
            "title": "Use SQLAlchemy ORM",
            "context": "Database requirements",
            "decision": "Use SQLAlchemy for database access",
            "rationale": "Type safety and query builder",
            "affected_files": ["agent_core/database.py"],
            "tags": ["database", "orm"],
            "contradictions": [],
            "resolved_by": None,
            "status": "active",
            "category": "database",
        },
        {
            "id": "003",
            "date": "2024-01-03T00:00:00Z",
            "title": "Use Pydantic for validation",
            "context": "Input validation",
            "decision": "Use Pydantic for all input validation",
            "rationale": "Type safety and validation",
            "affected_files": ["agent_core/api_client.py"],
            "tags": ["validation"],
            "contradictions": [],
            "resolved_by": None,
            "status": "active",
            "category": "testing",
        },
    ]

    # load_decisions() normalizes affected_files and DROPS entries that do
    # not exist on disk — create the referenced files so file-based overlap
    # detection and the decision graph keep their file edges.
    for d in decisions:
        for f in d["affected_files"]:
            p = tmp_path / f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("# sample\n", encoding="utf-8")

    decisions_file.write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    return tmp_path


class TestDecideCommandWithManyDecisions:
    """Test decide command performance with many decisions."""

    @pytest.mark.asyncio
    async def test_decide_command_with_many_decisions(
        self, mock_agent, workspace_with_decisions
    ):
        """Test that decide check works with many decisions efficiently."""

        # Create a workspace with more decisions (simulate larger repo)
        larger_workspace = workspace_with_decisions.parent / "larger_workspace"
        larger_workspace.mkdir()
        # load_decisions drops affected_files entries that don't exist on disk
        (larger_workspace / "agent_core").mkdir()
        for i in range(1, 51):
            (larger_workspace / "agent_core" / f"file{i}.py").write_text(
                "# sample\n", encoding="utf-8"
            )

        # Add more decisions to simulate larger repo
        more_decisions = [
            {
                "id": str(i).zfill(3),
                "date": f"2024-01-{i:02d}T00:00:00Z",
                "title": f"Decision {i}",
                "context": f"Context {i}",
                "decision": f"Decision {i}",
                "rationale": f"Rationale {i}",
                "affected_files": [f"agent_core/file{i}.py"],
                # Only every 5th decision shares tag1, so overlap filtering
                # is observable (an all-shared tag would match everything).
                "tags": ["tag1"] if i % 5 == 0 else [f"tag{i}"],
                "contradictions": [],
                "resolved_by": None,
                "status": "active",
                "category": "testing",
            }
            for i in range(1, 51)  # 50 decisions
        ]

        (larger_workspace / ".decisions.json").write_text(
            json.dumps(more_decisions, indent=2), encoding="utf-8"
        )

        # Test that we can load all decisions
        all_decisions = load_decisions(larger_workspace)
        assert len(all_decisions) == 50

        # Test that find_overlaps filters correctly
        new_decision = {
            "tags": ["tag1"],
            "affected_files": ["agent_core/file10.py"]
        }
        overlaps = find_overlaps(new_decision, all_decisions, larger_workspace)
        # Should find decisions that share tag1 (every 5th) OR file10.py
        assert len(overlaps) > 0
        assert len(overlaps) < len(all_decisions)  # Should be filtered

    @pytest.mark.asyncio
    async def test_decide_command_with_overlap(
        self, mock_agent, workspace_with_decisions
    ):
        """Test that LLM is only called on overlapping decisions."""
        from agent_core.decisions import check_contradictions_fast

        # Create a new decision that overlaps with existing ones
        new_decision = {
            "tags": ["concurrency"],
            "affected_files": ["agent_core/api_client.py"]
        }

        # This should only check decision 001 (concurrency tag)
        result = await check_contradictions_fast(
            mock_agent,
            new_decision,
            "New concurrency approach",
            workspace_with_decisions
        )

        # Should not have early returned "No similar decisions"
        # because there is overlap (decision 001)
        assert (
            "similar decisions" not in result.lower()
            or "contradiction" in result.lower()
        )


class TestDecisionGraphQueries:
    """Test decision graph-based queries."""

    def test_get_relevant_decisions_by_files(self, workspace_with_decisions):
        """Test querying decisions by affected files."""
        # Build graph
        decisions = load_decisions(workspace_with_decisions)
        graph = build_decision_graph(decisions)

        # Query for decisions affecting api_client.py
        d_ids = set()
        for f in ["agent_core/api_client.py"]:
            d_ids.update(graph.get("files", {}).get(f, []))

        relevant = [graph["decisions"][d_id] for d_id in d_ids]
        assert len(relevant) == 2
        relevant_ids = {d["id"] for d in relevant}
        assert relevant_ids == {"001", "003"}

    def test_get_relevant_decisions_by_tags(self, workspace_with_decisions):
        """Test querying decisions by tags."""
        # Build graph
        decisions = load_decisions(workspace_with_decisions)
        graph = build_decision_graph(decisions)

        # Query for decisions with concurrency tag
        d_ids = set()
        for tag in ["concurrency"]:
            d_ids.update(graph.get("tags", {}).get(tag, []))

        relevant = [graph["decisions"][d_id] for d_id in d_ids]
        assert len(relevant) == 1
        assert relevant[0]["id"] == "001"

    def test_get_relevant_decisions_by_files_and_tags(self, workspace_with_decisions):
        """Test querying decisions by both files and tags."""
        # Build graph
        decisions = load_decisions(workspace_with_decisions)
        graph = build_decision_graph(decisions)

        # Query for decisions affecting api_client.py with concurrency tag
        # ("with" = intersection, not union)
        file_ids = set()
        for f in ["agent_core/api_client.py"]:
            file_ids.update(graph.get("files", {}).get(f, []))
        tag_ids = set()
        for tag in ["concurrency"]:
            tag_ids.update(graph.get("tags", {}).get(tag, []))
        d_ids = file_ids & tag_ids

        relevant = [graph["decisions"][d_id] for d_id in d_ids]
        # Should find decision 001 (has both concurrency tag and affects api_client.py)
        assert len(relevant) == 1
        assert relevant[0]["id"] == "001"


class TestCategoryIndexQueries:
    """Test category index-based queries."""

    def test_get_decisions_by_category(self, workspace_with_decisions):
        """Test querying decisions by category."""
        # Build index
        decisions = load_decisions(workspace_with_decisions)
        index = build_category_index(decisions)
        by_id = {d["id"]: d for d in decisions}

        # Query for architecture decisions (index maps category -> decision IDs)
        d_ids = index.get("architecture", [])
        relevant = [by_id[i] for i in d_ids]
        assert len(relevant) == 1
        assert relevant[0]["id"] == "001"

    def test_get_decisions_by_multiple_categories(self, workspace_with_decisions):
        """Test querying decisions by multiple categories."""
        # Build index
        decisions = load_decisions(workspace_with_decisions)
        index = build_category_index(decisions)
        by_id = {d["id"]: d for d in decisions}

        # Query for testing decisions
        d_ids = set(index.get("testing", []))
        # Add another category
        d_ids.update(index.get("architecture", []))

        relevant = [by_id[i] for i in d_ids]
        assert len(relevant) == 2
        relevant_ids = {d["id"] for d in relevant}
        assert relevant_ids == {"001", "003"}


# Mock fixtures
@pytest.fixture
def mock_agent():
    """Create a mock Agent for testing."""
    from unittest.mock import AsyncMock, MagicMock

    agent = MagicMock(spec=["llm"])
    agent.llm.chat = AsyncMock(return_value="No contradictions found.")
    return agent
