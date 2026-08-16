"""Phase 1 tests: HTIR compilation and link inference.

A synthetic failed trace (stuck cycle with provenance reuse) compiles to the
expected HTIR nodes and links, and round-trips through the reader.
"""
from __future__ import annotations

from harnessfix.htir import TraceGraph, compile_trace
from harnessfix.links import infer_links
from harnessfix.reader import read_trace, task_id_of
from harnessfix.tracing import (
    KIND_LLM_RESPONSE,
    KIND_LOOP_END,
    KIND_STEP_START,
    KIND_TOOL_CALL,
    KIND_TOOL_RESULT,
    TraceWriter,
)


def _write_failed_trace(tmp_path, task_id: str = "htir-1") -> str:
    """Stuck-cycle trace with a provenance-reusing edit call."""
    writer = TraceWriter(task_id=task_id, directory=tmp_path)
    emit = writer.emit
    emit({"kind": KIND_STEP_START, "layer": "lifecycle", "iteration": 0})
    emit({"kind": KIND_LLM_RESPONSE, "layer": "observability", "text": "read the file"})
    emit(
        {"kind": KIND_TOOL_CALL, "layer": "tool_interface", "tool": "read", "args_hash": "h1", "args": '{"path": "a.txt"}'}
    )
    emit(
        {"kind": KIND_TOOL_RESULT, "layer": "tool_interface", "tool": "read", "args_hash": "h1", "result": "important_symbol 12345 found in a.txt"}
    )
    # The edit reuses a token from the read result -> provenance link.
    emit(
        {"kind": KIND_TOOL_CALL, "layer": "tool_interface", "tool": "edit", "args_hash": "h2", "args": '{"path": "a.txt", "old": "important_symbol"}'}
    )
    emit(
        {"kind": KIND_TOOL_RESULT, "layer": "tool_interface", "tool": "edit", "args_hash": "h2", "result": "patched"}
    )
    # Stuck cycle: the same probe three times -> guard fires -> loop ends stuck.
    for i in range(3):
        emit({"kind": KIND_STEP_START, "layer": "lifecycle", "iteration": i + 1})
        emit({"kind": KIND_TOOL_CALL, "layer": "tool_interface", "tool": "search", "args_hash": "h9", "duplicate": i > 0})
        emit({"kind": KIND_TOOL_RESULT, "layer": "tool_interface", "tool": "search", "args_hash": "h9", "duplicate": i > 0, "result": "same"} )
    emit({"kind": "guard_triggered", "layer": "lifecycle", "guard": "stuck", "note": "stop repeating"})
    emit({"kind": KIND_LOOP_END, "layer": "lifecycle", "outcome": "stuck", "termination_reason": "stuck"})
    return str(writer.path)


def test_compile_trace_builds_graph_with_expected_steps(tmp_path):
    path = _write_failed_trace(tmp_path)
    events = read_trace(path)
    assert task_id_of(path) == "htir-1"

    graph: TraceGraph = compile_trace(path)
    assert graph.task_id == "htir-1"
    assert len(graph.steps) == len(events)
    # Steps mirror the event stream.
    assert [s.kind for s in graph.steps] == [e["kind"] for e in events]
    # Failed steps detected (stuck loop_end).
    assert any(s.kind == KIND_LOOP_END and s.is_failed() for s in graph.steps)


def test_provenance_link_from_result_to_reusing_call(tmp_path):
    graph = compile_trace(_write_failed_trace(tmp_path))
    prov = [l for l in graph.links if l.kind == "provenance"]
    assert prov, "expected at least one provenance link"
    target = graph.step(prov[0].target)
    assert target is not None and target.payload.get("tool") == "edit"
    src = graph.step(prov[0].source)
    assert src is not None and "important_symbol" in str(src.payload.get("result", ""))


def test_control_flow_links_for_guard_and_stuck_cycle(tmp_path):
    graph = compile_trace(_write_failed_trace(tmp_path))
    ctrl = [l for l in graph.links if l.kind == "control_flow"]
    # guard->next link plus stuck-cycle links exist.
    assert any(l.detail.startswith("step follows injected guard") for l in ctrl)
    assert any("stuck cycle" in l.detail for l in ctrl)


def test_infer_links_is_deterministic_and_capped(tmp_path):
    graph = compile_trace(_write_failed_trace(tmp_path, "det-a"))
    again = compile_trace(_write_failed_trace(tmp_path, "det-b"))
    assert [l.link_id for l in graph.links] == [l.link_id for l in again.links]
    assert len(graph.links) <= 40
    # Every link refers to real step indices.
    indexes = {s.index for s in graph.steps}
    for link in graph.links:
        assert link.source in indexes and link.target in indexes
