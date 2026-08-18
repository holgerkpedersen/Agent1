"""Tests for harnessfix.review — the human verification gate (decision #053)."""

import json
from pathlib import Path

import pytest

from harnessfix.review import (
    DISPOSITIONS,
    ReviewRecord,
    build_reviews,
    export_regression_test,
    label_review,
    load_reviews,
    review_table,
    save_reviews,
)
from harnessfix.tracing import TraceWriter


def _failed_trace(writer: TraceWriter) -> None:
    writer.emit({"kind": "task_begin", "layer": "context",
                 "user_input": "fix the flaky gate",
                 "model": "qwen3.8-27b", "profile": "deep-analysis"})
    writer.emit({"kind": "tool_call", "layer": "tool_interface",
                 "tool": "write", "args_hash": "w"})
    writer.emit({"kind": "tool_result", "layer": "tool_interface",
                 "tool": "write", "affected_files": ["a.py"]})
    writer.emit({"kind": "tool_error", "layer": "tool_interface",
                 "exception": "PermissionError", "message": "denied"})
    writer.emit({"kind": "loop_end", "layer": "lifecycle", "outcome": "error"})


def test_build_reviews_covers_only_failed_traces(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()

    failed = TraceWriter(task_id="tfail", directory=traces)
    _failed_trace(failed)
    failed.close()

    ok = TraceWriter(task_id="tok", directory=traces)
    ok.emit({"kind": "task_begin", "layer": "context", "user_input": "hi"})
    ok.emit({"kind": "loop_end", "layer": "lifecycle", "outcome": "completed"})
    ok.close()

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
