"""Tests for harnessfix.review — the human verification gate (decision #053)."""

import json
from pathlib import Path

import pytest

from harnessfix.review import (
    DISPOSITIONS,
    ReviewRecord,
    build_record_for,
    build_reviews,
    export_regression_test,
    label_review,
    load_reviews,
    merge_existing_reviews,
    review_table,
    save_reviews,
)
from harnessfix.tracing import TraceWriter


def _failed_trace(writer: TraceWriter, *, meta: bool = True) -> None:
    begin = {"kind": "task_begin", "layer": "context",
             "user_input": "fix the flaky gate"}
    if meta:
        begin["model"] = "qwen3.8-27b"
        begin["profile"] = "deep-analysis"
    writer.emit(begin)
    writer.emit({"kind": "tool_call", "layer": "tool_interface",
                 "tool": "write", "args_hash": "w"})
    writer.emit({"kind": "tool_result", "layer": "tool_interface",
                 "tool": "write", "affected_files": ["a.py"]})
    writer.emit({"kind": "tool_error", "layer": "tool_interface",
                 "exception": "PermissionError", "message": "denied"})
    writer.emit({"kind": "loop_end", "layer": "lifecycle", "outcome": "error"})


def test_build_reviews_covers_only_failed_metadata_traces(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()

    failed = TraceWriter(task_id="tfail", directory=traces)
    _failed_trace(failed)
    failed.close()

    ok = TraceWriter(task_id="tok", directory=traces)
    ok.emit({"kind": "task_begin", "layer": "context", "user_input": "hi",
             "model": "m", "profile": "p"})
    ok.emit({"kind": "loop_end", "layer": "lifecycle", "outcome": "completed"})
    ok.close()

    old = TraceWriter(task_id="told", directory=traces)
    _failed_trace(old, meta=False)
    old.close()

    reviews = build_reviews(traces)
    assert set(reviews) == {"tfail"}
    rec = reviews["tfail"]
    assert rec.prompt == "fix the flaky gate"
    assert rec.model == "qwen3.8-27b"
    assert rec.profile == "deep-analysis"
    assert rec.affected_files == ["a.py"]
    assert rec.outcome == "error"
    assert rec.root_layer == "execution_environment"
    assert rec.disposition == "unreviewed"


def test_pre050_traces_excluded_from_ledger(tmp_path):
    """Pre-#050 traces carry no model/profile meta and no prompt — they
    cannot be human-judged, so the review ledger must not contain them."""
    traces = tmp_path / "traces"
    traces.mkdir()
    old = TraceWriter(task_id="told", directory=traces)
    _failed_trace(old, meta=False)
    old.close()
    assert build_reviews(traces) == {}


def test_build_record_for_on_demand_auto_review(tmp_path):
    """`review label <task> auto` must work for pre-#050 traces on demand:
    build_record_for creates the record even though build_reviews excludes
    them (the agent's structural review needs no prompt/model)."""
    traces = tmp_path / "traces"
    traces.mkdir()
    old = TraceWriter(task_id="told", directory=traces)
    _failed_trace(old, meta=False)
    old.close()
    assert build_reviews(traces) == {}
    rec = build_record_for(traces / "told.jsonl")
    assert rec is not None
    assert rec.task_id == "told"
    assert rec.outcome == "error"
    assert rec.root_layer == "execution_environment"


def test_build_record_for_rejects_success_trace(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    ok = TraceWriter(task_id="tok", directory=traces)
    ok.emit({"kind": "task_begin", "layer": "context", "user_input": "hi"})
    ok.emit({"kind": "loop_end", "layer": "lifecycle", "outcome": "completed"})
    ok.close()
    assert build_record_for(traces / "tok.jsonl") is None


def test_merge_existing_reviews_preserves_labels(tmp_path):
    """Refresh must never destroy a labeled record whose trace still exists —
    even when the task is outside the default population (pre-#050)."""
    traces = tmp_path / "traces"
    traces.mkdir()
    old = TraceWriter(task_id="told", directory=traces)
    _failed_trace(old, meta=False)
    old.close()
    other = TraceWriter(task_id="tvanish", directory=traces)
    _failed_trace(other, meta=False)
    other.close()
    (traces / "tvanish.jsonl").unlink()

    existing = {
        "told": ReviewRecord(task_id="told", disposition="bug", source="agent"),
        "tvanish": ReviewRecord(task_id="tvanish", disposition="noise", source="human"),
        "tunlabeled": ReviewRecord(task_id="tunlabeled", disposition="unreviewed"),
    }
    merged = merge_existing_reviews({}, existing, traces)
    assert set(merged) == {"told"}
    assert merged["told"].source == "agent"


def test_label_and_persist_roundtrip(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    failed = TraceWriter(task_id="tfail", directory=traces)
    _failed_trace(failed)
    failed.close()

    ledger = tmp_path / "review.json"
    reviews = build_reviews(traces)
    label_review(reviews, "tfail", "bug", note="permission handling missing")
    save_reviews(reviews, ledger)

    loaded = load_reviews(ledger)
    assert loaded["tfail"].disposition == "bug"
    assert loaded["tfail"].note == "permission handling missing"
    assert loaded["tfail"].review_date


def test_label_rejects_unknown_disposition(tmp_path):
    reviews = {"t": ReviewRecord(task_id="t")}
    with pytest.raises(ValueError):
        label_review(reviews, "t", "wontfix")
    assert "bug" in DISPOSITIONS


def test_load_missing_or_corrupt_ledger_is_empty(tmp_path):
    assert load_reviews(tmp_path / "nope.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("not json {", encoding="utf-8")
    assert load_reviews(bad) == {}


def test_review_table_renders(tmp_path):
    reviews = {"t1": ReviewRecord(task_id="t1", model="m", root_layer="lifecycle",
                                  outcome="stuck", disposition="bug")}
    table = review_table(reviews)
    assert "t1" in table and "bug" in table and "lifecycle" in table


def test_review_table_renders_dash_for_missing_fields(tmp_path):
    """Old traces (pre-#050) lack model/root_layer/outcome — the table must
    show '-' instead of blank columns."""
    reviews = {"t1": ReviewRecord(task_id="t1")}
    table = review_table(reviews)
    assert " - " in table.replace("t1", "  ")
    assert "unreviewed" in table


def test_export_regression_test_pins_diagnosis(tmp_path):
    from harnessfix.htir import compile_trace

    traces = tmp_path / "traces"
    traces.mkdir()
    failed = TraceWriter(task_id="tfail", directory=traces)
    _failed_trace(failed)
    failed.close()

    rec = build_reviews(traces)["tfail"]
    out = export_regression_test(rec, traces / "tfail.jsonl", tmp_path / "gen")

    text = out.read_text(encoding="utf-8")
    assert "test_review_pin" in text
    assert rec.root_layer in text
    escaped = str((traces / "tfail.jsonl").resolve()).replace("\\", "\\\\")
    assert escaped in text

    # The generated pin actually passes against the current diagnosis.
    graph = compile_trace(traces / "tfail.jsonl")
    from harnessfix.diagnose import diagnose_graph

    diag = diagnose_graph(graph)
    assert diag.root_layer == rec.root_layer == "execution_environment"
    assert rec.mechanism[:80] in diag.mechanism
