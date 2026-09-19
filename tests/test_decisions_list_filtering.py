"""Tests for decide list filtering: status, category, --open, --summary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_core.decisions import (
    CATEGORIES,
    STATUS_ACTIVE,
    STATUS_ARCHIVED,
    STATUS_SUPERSEDED,
    _STATUSES,
    add_decision,
    count_by_category,
    count_by_status,
    find_decisions,
    load_decisions,
    save_decisions,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _make_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


# ── Status normalization ───────────────────────────────────────────────────


class TestStatusNormalization:
    """Decisions loaded without a 'status' field default to active."""

    def test_missing_status_defaults_active(self, tmp_path):
        ws = _make_workspace(tmp_path)
        raw = [
            {"id": "001", "title": "No status", "tags": [], "affected_files": []},
            {"id": "002", "title": "Has status", "status": "superseded",
             "tags": [], "affected_files": []},
        ]
        save_decisions(ws, raw)
        loaded = load_decisions(ws)
        assert loaded[0]["status"] == STATUS_ACTIVE
        assert loaded[1]["status"] == STATUS_SUPERSEDED

    def test_invalid_status_normalized_to_active(self, tmp_path):
        ws = _make_workspace(tmp_path)
        raw = [
            {"id": "001", "title": "Bad status", "status": "accepted",
             "tags": [], "affected_files": []},
        ]
        save_decisions(ws, raw)
        loaded = load_decisions(ws)
        assert loaded[0]["status"] == STATUS_ACTIVE

    def test_all_three_statuses_survive_roundtrip(self, tmp_path):
        ws = _make_workspace(tmp_path)
        raw = [
            {"id": "001", "title": "Active", "status": "active",
             "tags": [], "affected_files": []},
            {"id": "002", "title": "Superseded", "status": "superseded",
             "tags": [], "affected_files": []},
            {"id": "003", "title": "Archived", "status": "archived",
             "tags": [], "affected_files": []},
        ]
        save_decisions(ws, raw)
        loaded = load_decisions(ws)
        statuses = [d["status"] for d in loaded]
        assert statuses == ["active", "superseded", "archived"]


# ── Status filtering ────────────────────────────────────────────────────────


class TestStatusFiltering:
    def _setup(self, ws: Path):
        raw = [
            {"id": "001", "title": "Active one", "status": "active",
             "tags": ["a"], "affected_files": []},
            {"id": "002", "title": "Superseded one", "status": "superseded",
             "tags": ["a"], "affected_files": []},
            {"id": "003", "title": "Another active", "status": "active",
             "tags": ["b"], "affected_files": []},
        ]
        save_decisions(ws, raw)

    def test_filter_active(self, tmp_path):
        ws = _make_workspace(tmp_path)
        self._setup(ws)
        results = find_decisions(ws, status="active")
        assert len(results) == 2
        assert all(d["status"] == "active" for d in results)

    def test_filter_superseded(self, tmp_path):
        ws = _make_workspace(tmp_path)
        self._setup(ws)
        results = find_decisions(ws, status="superseded")
        assert len(results) == 1
        assert results[0]["id"] == "002"

    def test_filter_archived_returns_empty(self, tmp_path):
        ws = _make_workspace(tmp_path)
        self._setup(ws)
        results = find_decisions(ws, status="archived")
        assert results == []

    def test_no_status_filter_returns_all(self, tmp_path):
        ws = _make_workspace(tmp_path)
        self._setup(ws)
        results = find_decisions(ws)
        assert len(results) == 3


# ── Category filtering ──────────────────────────────────────────────────────


class TestCategoryFiltering:
    def _setup(self, ws: Path):
        raw = [
            {"id": "001", "title": "Security", "status": "active",
             "tags": [], "affected_files": [], "category": "security"},
            {"id": "002", "title": "Arch", "status": "active",
             "tags": [], "affected_files": [], "category": "architecture"},
            {"id": "003", "title": "More security", "status": "active",
             "tags": [], "affected_files": [], "category": "security"},
            {"id": "004", "title": "No cat", "status": "active",
             "tags": [], "affected_files": [], "category": ""},
        ]
        save_decisions(ws, raw)

    def test_filter_by_category(self, tmp_path):
        ws = _make_workspace(tmp_path)
        self._setup(ws)
        results = find_decisions(ws, category="security")
        assert len(results) == 2
        assert all(d["category"] == "security" for d in results)

    def test_filter_by_architecture(self, tmp_path):
        ws = _make_workspace(tmp_path)
        self._setup(ws)
        results = find_decisions(ws, category="architecture")
        assert len(results) == 1

    def test_filter_by_nonexistent_category(self, tmp_path):
        ws = _make_workspace(tmp_path)
        self._setup(ws)
        results = find_decisions(ws, category="ops")
        assert results == []


# ── Combined filters ────────────────────────────────────────────────────────


class TestCombinedFilters:
    def _setup(self, ws: Path):
        raw = [
            {"id": "001", "title": "Sec active", "status": "active",
             "tags": ["security"], "affected_files": [], "category": "security"},
            {"id": "002", "title": "Sec sup", "status": "superseded",
             "tags": ["security"], "affected_files": [], "category": "security"},
            {"id": "003", "title": "Arch active", "status": "active",
             "tags": ["arch"], "affected_files": [], "category": "architecture"},
        ]
        save_decisions(ws, raw)

    def test_status_plus_category(self, tmp_path):
        ws = _make_workspace(tmp_path)
        self._setup(ws)
        results = find_decisions(ws, status="active", category="security")
        assert len(results) == 1
        assert results[0]["id"] == "001"

    def test_status_plus_tag(self, tmp_path):
        ws = _make_workspace(tmp_path)
        self._setup(ws)
        results = find_decisions(ws, tags=["security"], status="active")
        assert len(results) == 1
        assert results[0]["id"] == "001"

    def test_all_filters_combined(self, tmp_path):
        ws = _make_workspace(tmp_path)
        self._setup(ws)
        results = find_decisions(ws, tags=["security"], status="active",
                                 category="security")
        assert len(results) == 1


# ── Count helpers ───────────────────────────────────────────────────────────


class TestCountHelpers:
    def test_count_by_status(self, tmp_path):
        ws = _make_workspace(tmp_path)
        raw = [
            {"id": "001", "title": "A", "status": "active",
             "tags": [], "affected_files": []},
            {"id": "002", "title": "B", "status": "active",
             "tags": [], "affected_files": []},
            {"id": "003", "title": "C", "status": "superseded",
             "tags": [], "affected_files": []},
        ]
        save_decisions(ws, raw)
        decisions = load_decisions(ws)
        counts = count_by_status(decisions)
        assert counts == {"active": 2, "superseded": 1}

    def test_count_by_category(self, tmp_path):
        ws = _make_workspace(tmp_path)
        raw = [
            {"id": "001", "title": "A", "status": "active",
             "tags": [], "affected_files": [], "category": "security"},
            {"id": "002", "title": "B", "status": "active",
             "tags": [], "affected_files": [], "category": "security"},
            {"id": "003", "title": "C", "status": "active",
             "tags": [], "affected_files": [], "category": "testing"},
            {"id": "004", "title": "D", "status": "active",
             "tags": [], "affected_files": [], "category": ""},
        ]
        save_decisions(ws, raw)
        decisions = load_decisions(ws)
        counts = count_by_category(decisions)
        assert counts["security"] == 2
        assert counts["testing"] == 1
        assert counts["(uncategorized)"] == 1

    def test_count_empty(self):
        assert count_by_status([]) == {}
        assert count_by_category({}) == {}  # type: ignore[arg-type]


# ── add_decision with category ──────────────────────────────────────────────


class TestAddDecisionCategory:
    def test_add_with_category(self, tmp_path):
        ws = _make_workspace(tmp_path)
        record = add_decision(ws, "Test", category="security")
        assert record["category"] == "security"
        assert record["status"] == STATUS_ACTIVE

    def test_add_without_category(self, tmp_path):
        ws = _make_workspace(tmp_path)
        record = add_decision(ws, "Test")
        assert record["category"] == ""

    def test_category_survives_save_load(self, tmp_path):
        ws = _make_workspace(tmp_path)
        add_decision(ws, "Test", category="architecture")
        loaded = load_decisions(ws)
        assert loaded[0]["category"] == "architecture"


# ── set-status roundtrip ────────────────────────────────────────────────────


class TestSetStatusRoundtrip:
    def test_set_status(self, tmp_path):
        ws = _make_workspace(tmp_path)
        add_decision(ws, "Test decision")
        decisions = load_decisions(ws)
        record = decisions[0]
        assert record["status"] == STATUS_ACTIVE

        record["status"] = STATUS_SUPERSEDED
        save_decisions(ws, decisions)

        loaded = load_decisions(ws)
        assert loaded[0]["status"] == STATUS_SUPERSEDED

    def test_set_status_full_lifecycle(self, tmp_path):
        ws = _make_workspace(tmp_path)
        add_decision(ws, "Lifecycle test")
        decisions = load_decisions(ws)

        # active → superseded → archived
        decisions[0]["status"] = STATUS_SUPERSEDED
        save_decisions(ws, decisions)
        assert load_decisions(ws)[0]["status"] == STATUS_SUPERSEDED

        decisions = load_decisions(ws)
        decisions[0]["status"] = STATUS_ARCHIVED
        save_decisions(ws, decisions)
        assert load_decisions(ws)[0]["status"] == STATUS_ARCHIVED


# ── Backward compatibility ──────────────────────────────────────────────────


class TestBackwardCompatibility:
    """Decisions without status/category fields (like the existing 113)
    all load as active with empty category."""

    def test_113_legacy_decisions(self, tmp_path):
        ws = _make_workspace(tmp_path)
        # Simulate 113 legacy decisions without status/category
        raw = [
            {"id": str(i).zfill(3), "title": f"Legacy {i}",
             "tags": ["test"], "affected_files": []}
            for i in range(1, 114)
        ]
        save_decisions(ws, raw)
        loaded = load_decisions(ws)
        assert len(loaded) == 113
        assert all(d["status"] == STATUS_ACTIVE for d in loaded)
        assert all(d["category"] == "" for d in loaded)

    def test_mixed_legacy_and_new(self, tmp_path):
        ws = _make_workspace(tmp_path)
        raw = [
            {"id": "001", "title": "Old", "tags": [], "affected_files": []},
            {"id": "002", "title": "New", "status": "superseded",
             "category": "security", "tags": [], "affected_files": []},
        ]
        save_decisions(ws, raw)
        loaded = load_decisions(ws)
        assert loaded[0]["status"] == STATUS_ACTIVE
        assert loaded[0]["category"] == ""
        assert loaded[1]["status"] == "superseded"
        assert loaded[1]["category"] == "security"


# ── Constants sanity ────────────────────────────────────────────────────────


class TestConstants:
    def test_statuses_cover_all_expected(self):
        assert STATUS_ACTIVE == "active"
        assert STATUS_SUPERSEDED == "superseded"
        assert STATUS_ARCHIVED == "archived"
        assert _STATUSES == {"active", "superseded", "archived"}

    def test_categories_nonempty(self):
        assert len(CATEGORIES) >= 10
        assert "security" in CATEGORIES
        assert "testing" in CATEGORIES
