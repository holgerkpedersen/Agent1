"""Tests for the decision-ledger health check (decision #054)."""

from pathlib import Path

from agent_core.decisions import find_stale_decisions


def _decision(decision_id: str, files: list[str]) -> dict:
    return {"id": decision_id, "title": f"d{decision_id}", "affected_files": files,
            "contradictions": []}


def test_find_stale_decisions_flags_missing_files(tmp_path):
    (tmp_path / "exists.py").write_text("", encoding="utf-8")
    decisions = [
        _decision("001", ["exists.py", "gone.py"]),
        _decision("002", ["also_gone.py"]),
    ]
    stale = find_stale_decisions(tmp_path, decisions)
    assert [d["id"] for d in stale] == ["001", "002"]
    assert stale[0]["_missing_files"] == ["gone.py"]


def test_no_stale_when_all_files_exist(tmp_path):
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "b.py").write_text("", encoding="utf-8")
    decisions = [_decision("001", ["a.py", "b.py"])]
    assert find_stale_decisions(tmp_path, decisions) == []


def test_decisions_without_files_are_never_stale(tmp_path):
    decisions = [_decision("001", [])]
    assert find_stale_decisions(tmp_path, decisions) == []


def test_stale_marks_do_not_mutate_ledger_records(tmp_path):
    (tmp_path / "gone.py").write_text("", encoding="utf-8")
    (tmp_path / "gone.py").unlink()
    record = _decision("001", ["gone.py"])
    find_stale_decisions(tmp_path, [record])
    assert "_missing_files" not in record