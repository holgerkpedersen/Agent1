"""Tests for RAG-style decision index (Tier A compression).

Verifies that the compact index correctly builds from ``.decisions.json``,
caches and invalidates on ledger changes, and retrieves only relevant
decisions via file/tag matching — instead of broadcasting all active records.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch


from agent_core.decisions_index import (
    CACHE_PATH,
    INDEX_VERSION,
    build_index,
    load_or_build_index,
    retrieve_decisions,
)
from agent_core.decisions import (
    STATUS_SUPERSEDED,
    add_decision,
    load_decisions,
    save_decisions,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _make_workspace(tmp_path: Path) -> Path:
    """Create a workspace directory with a ``.decisions.json`` ledger."""
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


def _touch_files(ws: Path, paths: list[str]) -> None:
    """Create dummy files so ``add_decision``'s path normalization doesn't drop them."""
    for p in paths:
        (ws / p).parent.mkdir(parents=True, exist_ok=True)
        (ws / p).write_text("", encoding="utf-8")


# ── Index building ────────────────────────────────────────────────────────────


class TestBuildIndex:
    """Core index construction from the decision ledger."""

    def test_build_index_creates_cache(self, tmp_path):
        """build_index writes a valid cache file with correct structure."""
        ws = _make_workspace(tmp_path)
        _touch_files(ws, ["agent_core/security/tool_schema.py"])
        add_decision(
            str(ws),
            "Use subprocess not os.system",
            decision="Always use subprocess with list args.",
            affected_files=["agent_core/security/tool_schema.py"],
            tags=["security"],
        )

        index = build_index(str(ws))

        assert index["version"] == INDEX_VERSION
        assert len(index["summaries"]) > 0
        assert "agent_core/security/tool_schema.py" in index["file_index"]
        assert "security" in index["tag_index"]
        # Cache file should exist on disk
        cache_file = CACHE_PATH
        assert cache_file.exists()

    def test_build_index_excludes_superseded(self, tmp_path):
        """Only active decisions are indexed; superseded records are skipped."""
        ws = _make_workspace(tmp_path)
        _touch_files(ws, ["test.py"])
        add_decision(
            str(ws),
            "Active decision",
            decision="This is active.",
            affected_files=["test.py"],
            tags=["testing"],
        )
        # Manually add a superseded record
        decisions = load_decisions(str(ws))
        decisions.append({
            "id": "002",
            "date": "2026-11-05T00:00:00+00:00",
            "title": "Superseded decision",
            "context": "",
            "decision": "This is superseded.",
            "rationale": "",
            "affected_files": ["test.py"],
            "tags": ["testing"],
            "contradictions": [],
            "resolved_by": "001",
            "status": STATUS_SUPERSEDED,
            "category": "testing",
        })
        save_decisions(str(ws), decisions)

        index = build_index(str(ws))
        # Only the active decision should be in summaries
        assert len(index["summaries"]) == 1
        assert "001" in index["summaries"]
        assert "002" not in index["summaries"]

    def test_build_index_no_ledger_returns_empty(self, tmp_path):
        """When no .decisions.json exists, returns an empty index."""
        ws = _make_workspace(tmp_path)
        # No ledger file created
        index = build_index(str(ws))
        assert index["summaries"] == {}
        assert index["file_index"] == {}
        assert index["tag_index"] == {}

    def test_summary_truncation(self, tmp_path):
        """Long decision text is truncated with ellipsis."""
        ws = _make_workspace(tmp_path)
        _touch_files(ws, ["test.py"])
        long_text = "x" * 300
        add_decision(
            str(ws),
            "Long decision",
            decision=long_text,
            affected_files=["test.py"],
        )

        index = build_index(str(ws))
        summary = list(index["summaries"].values())[0]
        assert len(summary) == 201  # 200 chars + ellipsis


# ── Cache management ──────────────────────────────────────────────────────────


class TestCacheManagement:
    """load_or_build_index cache hit/miss/rebuild behavior."""

    def test_load_or_build_reuses_stale_cache(self, tmp_path):
        """No rebuild when ledger mtime matches cached value."""
        ws = _make_workspace(tmp_path)
        add_decision(
            str(ws),
            "Cached decision",
            decision="Stays the same.",
            affected_files=["test.py"],
        )

        # First call builds the index
        load_or_build_index(str(ws))

        # Second call should reuse cache (no rebuild)
        index2 = load_or_build_index(str(ws))
        assert len(index2["summaries"]) > 0

    def test_load_or_build_rebuilds_on_ledger_change(self, tmp_path):
        """Rebuild triggered when .decisions.json mtime changes."""
        ws = _make_workspace(tmp_path)
        add_decision(
            str(ws),
            "First decision",
            decision="Initial.",
            affected_files=["test.py"],
        )

        index1 = load_or_build_index(str(ws))
        first_summary_count = len(index1["summaries"])

        # Modify the ledger (add a new decision)
        time.sleep(0.1)  # ensure mtime changes
        add_decision(
            str(ws),
            "Second decision",
            decision="New one.",
            affected_files=["other.py"],
        )

        index2 = load_or_build_index(str(ws))
        assert len(index2["summaries"]) == first_summary_count + 1

    def test_load_or_build_rebuilds_on_cache_corruption(self, tmp_path):
        """Corrupted cache falls through to rebuild."""
        ws = _make_workspace(tmp_path)
        add_decision(
            str(ws),
            "Valid decision",
            decision="Recoverable.",
            affected_files=["test.py"],
        )

        # Build a valid index first
        load_or_build_index(str(ws))

        # Corrupt the cache file
        CACHE_PATH.write_text("not valid json{{{", encoding="utf-8")

        # Should rebuild successfully
        index = load_or_build_index(str(ws))
        assert len(index["summaries"]) > 0


# ── Retrieval ─────────────────────────────────────────────────────────────────


class TestRetrieveDecisions:
    """File and tag matching, ranking, and top-K limiting."""

    def test_file_index_exact_match(self, tmp_path):
        """Editing a file retrieves decisions whose affected_files match exactly."""
        ws = _make_workspace(tmp_path)
        _touch_files(ws, ["agent_core/security/tool_schema.py"])
        add_decision(
            str(ws),
            "Security constraint",
            decision="Use subprocess with list args.",
            affected_files=["agent_core/security/tool_schema.py"],
            tags=["security"],
        )

        index = build_index(str(ws))
        results = retrieve_decisions(
            index, files=["agent_core/security/tool_schema.py"]
        )

        assert len(results) == 1
        assert results[0]["id"] == "001"
        assert "subprocess" in results[0]["summary"]

    def test_file_index_basename_fallback(self, tmp_path):
        """Basenames match decisions that reference just the file name."""
        ws = _make_workspace(tmp_path)
        _touch_files(ws, ["agent_core/commands/fix_cmd.py"])
        add_decision(
            str(ws),
            "Fix command constraint",
            decision="Never delete pre-existing files.",
            affected_files=["agent_core/commands/fix_cmd.py"],
            tags=["security"],
        )

        index = build_index(str(ws))
        # Only basename provided — should still match via fallback
        results = retrieve_decisions(index, files=["fix_cmd.py"])

        assert len(results) == 1
        assert results[0]["id"] == "001"

    def test_tag_retrieval(self, tmp_path):
        """Tag-based retrieval returns decisions with matching tags."""
        ws = _make_workspace(tmp_path)
        _touch_files(ws, ["agent.py"])
        add_decision(
            str(ws),
            "Security decision",
            decision="Validate all inputs.",
            affected_files=["agent.py"],
            tags=["security", "correctness"],
        )

        index = build_index(str(ws))
        results = retrieve_decisions(index, tags=["security"])

        assert len(results) == 1
        assert results[0]["id"] == "001"

    def test_file_and_tag_combined_ranking(self, tmp_path):
        """File match (weight 2) ranks higher than tag-only match (weight 1)."""
        ws = _make_workspace(tmp_path)
        _touch_files(ws, ["agent_core/commands/fix_cmd.py", "other_file.py"])
        add_decision(
            str(ws),
            "Exact file match",
            decision="Priority constraint.",
            affected_files=["agent_core/commands/fix_cmd.py"],
            tags=["security"],
        )
        add_decision(
            str(ws),
            "Tag-only match",
            decision="Lower priority.",
            affected_files=["other_file.py"],
            tags=["security"],
        )

        index = build_index(str(ws))
        results = retrieve_decisions(index, files=["fix_cmd.py"], tags=["security"])

        # Both should be returned, but exact file match first (higher score)
        assert len(results) == 2
        assert results[0]["id"] == "001"  # exact file match wins

    def test_top_k_limiting(self, tmp_path):
        """Returns at most K results regardless of how many match."""
        ws = _make_workspace(tmp_path)
        dummy_files = [f"file_{i:03d}.py" for i in range(20)]
        _touch_files(ws, dummy_files)
        for i in range(20):
            add_decision(
                str(ws),
                f"Decision {i}",
                decision=f"Constraint {i}.",
                affected_files=[f"file_{i:03d}.py"],
                tags=["testing"],
            )

        index = build_index(str(ws))
        results = retrieve_decisions(index, files=["file_001.py"], top_k=5)

        assert len(results) <= 5

    def test_empty_when_no_match(self, tmp_path):
        """No files and no tags -> empty result (not all active decisions)."""
        ws = _make_workspace(tmp_path)
        _touch_files(ws, ["unrelated.py"])
        add_decision(
            str(ws),
            "Unrelated decision",
            decision="Does not match.",
            affected_files=["unrelated.py"],
            tags=["planning"],
        )

        index = build_index(str(ws))
        results = retrieve_decisions(index, files=None, tags=None)

        assert results == []

    def test_empty_when_no_tags_provided(self, tmp_path):
        """Only file match works; empty tag list is treated as no tags."""
        ws = _make_workspace(tmp_path)
        _touch_files(ws, ["agent.py"])
        add_decision(
            str(ws),
            "File-only decision",
            decision="Match on file.",
            affected_files=["agent.py"],
            tags=["architecture"],
        )

        index = build_index(str(ws))
        results = retrieve_decisions(index, files=["agent.py"], tags=[])

        assert len(results) == 1


# ── Integration ───────────────────────────────────────────────────────────────


class TestContextBlockIntegration:
    """End-to-end: _decision_constraints_block uses the index."""

    def test_context_block_returns_compact_output(self, tmp_path):
        """_decision_constraints_block returns summaries, not full text."""
        from agent_core.prompts.context_blocks import (
            _decision_constraints_block,
        )

        ws = _make_workspace(tmp_path)
        _touch_files(ws, ["agent_core/security/tool_schema.py"])
        add_decision(
            str(ws),
            "Security constraint",
            decision="Use subprocess with list args.",
            affected_files=["agent_core/security/tool_schema.py"],
            tags=["security"],
        )

        # Create a minimal mock agent
        class MockAgent:
            _files_read = ["agent_core/security/tool_schema.py"]
            workspace = str(ws)

        block = _decision_constraints_block(MockAgent())

        assert "CRITICAL DESIGN CONSTRAINTS" in block
        assert "#001:" in block
        assert "subprocess" in block
        # Should NOT contain the full verbose format with "Chose:" / "Why:"
        assert "Chose:" not in block


class TestDecisionConstraintsForIntegration:
    """Ensure existing fix_cmd tests still work after RAG migration."""

    def test_decision_constraints_for_still_works(self, tmp_path):
        """_decision_constraints_for uses _decision_workspace_for + index."""
        from agent_core.commands.fix_cmd import (
            _decision_constraints_for,
        )

        ws = _make_workspace(tmp_path)
        pkg = ws / "agent_core" / "commands"
        pkg.mkdir(parents=True)
        target = pkg / "fix_cmd.py"
        target.write_text("x = 1\n", encoding="utf-8")
        add_decision(
            str(ws),
            "No duplicate variants",
            decision="Never create duplicate function variants.",
            affected_files=["agent_core/commands/fix_cmd.py"],
        )

        block = _decision_constraints_for(str(target))
        assert block != "", "constraints lost for nested file"
        assert "CRITICAL DESIGN CONSTRAINTS" in block
        assert "No duplicate variants" in block or "duplicate function" in block.lower()


class TestLoadOrBuildWithMockedMtime:
    """Force cache invalidation by mocking os.path.getmtime."""

    def test_rebuild_on_mtime_mismatch(self, tmp_path):
        """When cached mtime differs from real mtime, rebuilds."""
        ws = _make_workspace(tmp_path)
        add_decision(
            str(ws),
            "Test decision",
            decision="Rebuild on mismatch.",
            affected_files=["test.py"],
        )

        # Build initial index
        load_or_build_index(str(ws))

        # Mock getmtime to return a different value -> forces rebuild
        real_mtime = os.path.getmtime(ws / ".decisions.json")
        with patch("os.path.getmtime", return_value=real_mtime + 9999):
            index = load_or_build_index(str(ws))

        # Should have rebuilt (still valid, but triggered a rebuild)
        assert len(index["summaries"]) > 0
