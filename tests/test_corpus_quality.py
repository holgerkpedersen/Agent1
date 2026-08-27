"""Tests for the offline, harness-centric quality gate (corpus_quality).

The trace corpus is a static record of past runs, so the gate validates
*target alignment*: a repair's layer must actually appear in the corpus's
observed failures.  These tests build synthetic corpora and assert the
baseline/acceptance logic in gates.should_accept_harness plus the snapshot
shape of corpus_quality.corpus_quality.
"""
from __future__ import annotations

from pathlib import Path

from harnessfix import gates
from harnessfix.corpus_quality import CorpusQuality, corpus_quality
from harnessfix.gates import should_accept_harness
from harnessfix.tracing import (
    GUARD_STUCK,
    KIND_GUARD_TRIGGERED,
    KIND_LOOP_END,
    KIND_TOOL_CALL,
    KIND_TOOL_ERROR,
    KIND_TOOL_RESULT,
    TraceWriter,
)


def _write_completed(traces_dir: Path, task_id: str) -> None:
    writer = TraceWriter(task_id=task_id, directory=traces_dir)
    writer.emit({"kind": KIND_LOOP_END, "layer": "lifecycle",
                 "outcome": "completed", "termination_reason": "answer"})
    writer.close()


def _write_tool_interface_failure(traces_dir: Path, task_id: str) -> None:
    writer = TraceWriter(task_id=task_id, directory=traces_dir)
    writer.emit({"kind": KIND_TOOL_ERROR, "layer": "tool_interface",
                 "exception": "ValidationError",
                 "message": "schema validation failed for path"})
    writer.emit({"kind": KIND_LOOP_END, "layer": "lifecycle",
                 "outcome": "completed", "termination_reason": "answer"})
    writer.close()


def _write_stuck_cycle(traces_dir: Path, task_id: str) -> None:
    writer = TraceWriter(task_id=task_id, directory=traces_dir)
    # Three consecutive identical tool calls -> a stuck cycle is diagnosed in
    # the LIFECYCLE layer by diagnose_graph.
    for i in range(3):
        writer.emit({"kind": KIND_TOOL_CALL, "layer": "tool_interface",
                     "tool": "read_file", "args_hash": "abc",
                     "args": {"path": "x"}})
        writer.emit({"kind": KIND_TOOL_RESULT, "layer": "tool_interface",
                     "tool": "read_file", "duration_s": 0.1})
    writer.emit({"kind": KIND_GUARD_TRIGGERED, "layer": "lifecycle",
                 "guard": GUARD_STUCK})
    writer.emit({"kind": KIND_LOOP_END, "layer": "lifecycle",
                 "outcome": "stuck", "termination_reason": "stuck"})
    writer.close()


def test_corpus_quality_counts_completed_and_failed(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    _write_completed(traces, "ok1")
    _write_completed(traces, "ok2")
    _write_tool_interface_failure(traces, "ti1")
    _write_stuck_cycle(traces, "st1")

    q = corpus_quality(traces)
    assert q.total == 4
    # 3 of 4 runs completed (the stuck cycle ended with outcome="stuck").
    assert q.success_rate == 0.75
    # The stuck cycle is diagnosed in the lifecycle layer; the tool-interface
    # failure is diagnosed in the tool_interface layer.
    assert q.layer_counts.get("lifecycle", 0) >= 1
    assert q.layer_counts.get("tool_interface", 0) >= 1


def test_corpus_quality_empty_corpus(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    q = corpus_quality(traces)
    assert q.total == 0
    assert q.success_rate == 0.0
    assert q.layer_counts == {}


def test_corpus_quality_skips_corrupt_traces(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "broken.jsonl").write_text("not valid json\n", encoding="utf-8")
    _write_completed(traces, "ok1")
    q = corpus_quality(traces)
    assert q.total == 1
    assert q.success_rate == 1.0


def test_run_harness_quality_gate_returns_none_on_empty(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    assert gates.run_harness_quality_gate(traces) is None


def test_accept_harness_requires_target_layer_in_corpus(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    _write_tool_interface_failure(traces, "ti1")
    baseline = corpus_quality(traces)

    # Same corpus as post (static) — acceptance depends on target alignment.
    post = corpus_quality(traces)

    # A repair targeting tool_interface IS evidenced -> accepted.
    assert should_accept_harness(baseline, post, target_layer="tool_interface") is True
    # A repair targeting lifecycle is OFF-TARGET for this corpus -> rejected.
    assert should_accept_harness(baseline, post, target_layer="lifecycle") is False
    # No target declared -> fail-open (non-blocking).
    assert should_accept_harness(baseline, post) is True


def test_accept_harness_rejects_dropped_success_rate(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    _write_tool_interface_failure(traces, "ti1")
    baseline = corpus_quality(traces)

    # Simulate a post run whose completion rate fell (e.g. corrupted corpus).
    post = CorpusQuality(success_rate=0.0, mechanism_counts={},
                         layer_counts={"tool_interface": 1}, total=1)
    assert should_accept_harness(baseline, post, target_layer="tool_interface") is False


def test_accept_harness_unavailable_is_non_blocking():
    # Missing evidence degrades to fail-open so it never rejects on its own.
    assert should_accept_harness(None, None, target_layer="lifecycle") is True
