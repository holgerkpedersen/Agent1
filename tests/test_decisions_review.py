"""Tests for the decision-ledger health check (decision #054, #080-#087)."""

from pathlib import Path

from agent_core.decisions import (
    find_stale_decisions,
    find_open_contradictions,
    find_meta_warnings,
    ledger_health,
)


def _decision(decision_id: str, files: list[str], **extra) -> dict:
    d = {
        "id": decision_id,
        "title": f"d{decision_id}",
        "affected_files": files,
        "contradictions": [],
    }
    d.update(extra)
    return d


# ── find_stale_decisions (original tests) ──────────────────────────────────


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


def test_transient_paths_are_not_flagged_stale(tmp_path):
    """Regression (CI): a reference under a gitignored/machine-local tree is not
    a stale file.  Decisions #095/#096/#100 referenced ``.pytest_tmp/...`` test
    scratch paths that exist locally but not on a fresh checkout, which failed
    ``scripts/audit_invariants.py`` on every CI run."""
    decisions = [
        _decision("095", [".pytest_tmp/scratch.py"]),
        _decision("096", ["reports/traces/x.jsonl"]),
        _decision("100", [".docs/2026/x/plan.md", "backups/agent.py"]),
    ]
    assert find_stale_decisions(tmp_path, decisions) == []


def test_transient_lookalike_dir_is_still_checked(tmp_path):
    """Only the exact transient directory names are exempt — a real top-level
    file whose name merely starts with one is still verified."""
    decisions = [_decision("001", ["reports_keep.py"])]
    stale = find_stale_decisions(tmp_path, decisions)
    assert [d["id"] for d in stale] == ["001"]


def test_stale_marks_do_not_mutate_ledger_records(tmp_path):
    (tmp_path / "gone.py").write_text("", encoding="utf-8")
    (tmp_path / "gone.py").unlink()
    record = _decision("001", ["gone.py"])
    find_stale_decisions(tmp_path, [record])
    assert "_missing_files" not in record


# ── find_open_contradictions ───────────────────────────────────────────────


def test_open_contradictions_flags_unresolved():
    """Decision with unresolved contradictions is flagged."""
    d = _decision("010", [], contradictions=[
        {"id": "005", "status": "open"},
        {"id": "003", "status": "resolved"},
    ])
    result = find_open_contradictions([d])
    assert len(result) == 1
    assert result[0]["id"] == "010"
    assert result[0]["_open_contradiction_ids"] == ["005"]


def test_open_contradictions_ignores_resolved():
    """Decision with only resolved/superseded contradictions is not flagged."""
    d = _decision("010", [], contradictions=[
        {"id": "005", "status": "resolved"},
        {"id": "003", "status": "superseded"},
    ])
    assert find_open_contradictions([d]) == []


def test_open_contradictions_ignores_empty_list():
    """Decision with no contradictions is not flagged."""
    d = _decision("010", [], contradictions=[])
    assert find_open_contradictions([d]) == []


def test_open_contradictions_ignores_missing_status():
    """Contradiction dict without status field is treated as open."""
    d = _decision("010", [], contradictions=[
        {"id": "005"},  # no status field
    ])
    result = find_open_contradictions([d])
    assert len(result) == 1
    assert result[0]["_open_contradiction_ids"] == ["005"]


def test_open_contradictions_does_not_mutate():
    """Original decision dict is not mutated."""
    d = _decision("010", [], contradictions=[
        {"id": "005", "status": "open"},
    ])
    find_open_contradictions([d])
    assert "_open_contradiction_ids" not in d


# ── find_meta_warnings ─────────────────────────────────────────────────────


def test_meta_warnings_flags_decisions_with_warnings():
    """Decision carrying meta_warnings is flagged."""
    d = _decision(
        "080", ["agent.py"],
        meta_warnings=["Repeats unverified claim reference: .plans.json"],
    )
    result = find_meta_warnings([d])
    assert len(result) == 1
    assert result[0]["id"] == "080"
    assert result[0]["_meta_warnings"] == [
        "Repeats unverified claim reference: .plans.json"
    ]


def test_meta_warnings_ignores_empty_warnings():
    """Decision with no meta_warnings is not flagged."""
    d = _decision("001", ["agent.py"])
    assert find_meta_warnings([d]) == []


def test_meta_warnings_does_not_mutate():
    """Original decision dict is not mutated."""
    d = _decision(
        "082", ["agent.py"],
        meta_warnings=["some warning"],
    )
    find_meta_warnings([d])
    assert "_meta_warnings" not in d


def test_meta_warnings_multiple_decisions():
    """Multiple flagged decisions are all returned."""
    d1 = _decision("080", ["a.py"], meta_warnings=["warn a"])
    d2 = _decision("082", ["b.py"], meta_warnings=["warn b1", "warn b2"])
    d3 = _decision("001", ["c.py"])  # no warnings
    result = find_meta_warnings([d1, d2, d3])
    assert [d["id"] for d in result] == ["080", "082"]
    assert result[1]["_meta_warnings"] == ["warn b1", "warn b2"]


# ── ledger_health (combined report) ────────────────────────────────────────


def test_ledger_health_combines_stale_and_contradictions(tmp_path):
    """ledger_health returns all three buckets."""
    (tmp_path / "exists.py").write_text("", encoding="utf-8")
    decisions = [
        _decision(
            "001", ["exists.py", "gone.py"],
            contradictions=[{"id": "003", "status": "open"}],
        ),
        _decision(
            "080", [],
            meta_warnings=["unverified claim"],
        ),
    ]
    report = ledger_health(tmp_path, decisions)
    assert "stale" in report
    assert "open_contradictions" in report
    assert "meta_warnings" in report
    assert len(report["stale"]) == 1
    assert report["stale"][0]["id"] == "001"
    assert len(report["open_contradictions"]) == 1
    assert report["open_contradictions"][0]["id"] == "001"
    assert len(report["meta_warnings"]) == 1
    assert report["meta_warnings"][0]["id"] == "080"


def test_ledger_health_clean(tmp_path):
    """Clean ledger produces empty buckets."""
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    decisions = [_decision("001", ["a.py"])]
    report = ledger_health(tmp_path, decisions)
    assert report["stale"] == []
    assert report["open_contradictions"] == []
    assert report["meta_warnings"] == []
