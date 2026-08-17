"""Decision extraction tests (plan task 35): recording, searching, overlap
detection, persistence, and best-effort contradiction parsing."""
import json
import tempfile
from pathlib import Path

import pytest

from agent_core.decisions import (
    _parse_json_array,
    add_decision,
    find_decisions,
    find_overlaps,
    load_decisions,
    normalize_affected_files,
)


@pytest.fixture
def ws() -> Path:
    return Path(tempfile.mkdtemp(prefix="decisions_"))


class TestAddAndPersist:
    def test_add_decision_persists_and_assigns_id(self, ws):
        record = add_decision(
            ws, "Use exact matching", context="benchmark", tags=["scoring"]
        )
        assert record["id"] == "001"
        assert record["tags"] == ["scoring"]
        loaded = load_decisions(ws)
        assert len(loaded) == 1
        assert loaded[0]["title"] == "Use exact matching"

    def test_ids_increment(self, ws):
        add_decision(ws, "one")
        add_decision(ws, "two")
        assert load_decisions(ws)[1]["id"] == "002"

    def test_add_decision_normalizes_affected_files(self, ws):
        (ws / "agent.py").write_text("x = 1", encoding="utf-8")
        record = add_decision(ws, "t", affected_files=["agent.py"])
        assert record["affected_files"] == ["agent.py"]

    def test_escape_paths_dropped_from_affected_files(self, ws):
        (ws / "agent.py").write_text("x = 1", encoding="utf-8")
        record = add_decision(
            ws, "t", affected_files=["../outside.py", "agent.py"]
        )
        assert record["affected_files"] == ["agent.py"]


class TestFindDecisions:
    def _seed(self, ws):
        (ws / "agent.py").write_text("x = 1", encoding="utf-8")
        (ws / "agent_core").mkdir(exist_ok=True)
        (ws / "agent_core" / "config.py").write_text("y = 1", encoding="utf-8")
        add_decision(ws, "A", tags=["arch"], affected_files=["agent.py"])
        add_decision(ws, "B", tags=["ops"], affected_files=["agent_core/config.py"])

    def test_find_by_tag(self, ws):
        self._seed(ws)
        assert [d["title"] for d in find_decisions(ws, tags=["arch"])] == ["A"]

    def test_find_by_file(self, ws):
        self._seed(ws)
        assert [d["title"] for d in find_decisions(ws, files=["agent.py"])] == ["A"]

    def test_find_by_keyword(self, ws):
        self._seed(ws)
        assert [d["title"] for d in find_decisions(ws, keyword="config")] == ["B"]

    def test_empty_workspace(self, ws):
        assert find_decisions(ws) == []


class TestOverlaps:
    def test_tag_overlap_detected(self, ws):
        add_decision(ws, "old", tags=["security"])
        new = {"tags": ["security"], "affected_files": []}
        assert len(find_overlaps(new, load_decisions(ws), ws)) == 1

    def test_no_overlap(self, ws):
        add_decision(ws, "old", tags=["security"])
        new = {"tags": ["benchmark"], "affected_files": []}
        assert find_overlaps(new, load_decisions(ws), ws) == []


class TestParseJsonArray:
    def test_parses_plain_json_array(self):
        parsed = _parse_json_array('[{"title": "x"}, {"title": "y"}]')
        assert [d["title"] for d in parsed] == ["x", "y"]

    def test_parses_array_embedded_in_text(self):
        parsed = _parse_json_array(
            "Here are the decisions:\n[{\"title\": \"a\"}, {\"title\": \"b\"}]\nRegards"
        )
        assert len(parsed) == 2

    def test_returns_empty_on_garbage(self):
        assert _parse_json_array("no decisions here") == []
        assert _parse_json_array("") == []

    def test_filters_non_dicts(self):
        parsed = _parse_json_array('[{"title": "x"}, 42, "str"]')
        assert len(parsed) == 1
