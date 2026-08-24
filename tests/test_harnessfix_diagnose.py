"""Phase 2 tests: heuristic signature classification (spec section 3.3)."""
from __future__ import annotations

from harnessfix.diagnose import Diagnosis, diagnose_graph
from harnessfix.htir import HTIRStep, TraceGraph
from harnessfix.tracing import (
    GUARD_BUDGET,
    KIND_GUARD_TRIGGERED,
    KIND_LLM_RESPONSE,
    KIND_LOOP_END,
    KIND_TOOL_CALL,
    KIND_TOOL_ERROR,
    KIND_TOOL_RESULT,
    LAYER_LIFECYCLE,
    LAYER_TOOL_INTERFACE,
    LAYER_EXECUTION,
    LAYER_GOVERNANCE,
    LAYER_CONTEXT,
    LAYER_VERIFICATION,
)


def _graph(steps: list[tuple[str, str, dict]]) -> TraceGraph:
    """Build a TraceGraph from (kind, layer, payload) triples."""
    return TraceGraph(
        task_id="t",
        steps=[
            HTIRStep(index=i, kind=kind, layer_facet=layer, payload=payload)
            for i, (kind, layer, payload) in enumerate(steps)
        ],
    )


def _loop_end(outcome: str) -> tuple[str, str, dict]:
    return KIND_LOOP_END, LAYER_LIFECYCLE, {"outcome": outcome, "termination_reason": outcome}


def test_validation_error_maps_to_tool_interface():
    g = _graph(
        [
            (KIND_TOOL_ERROR, LAYER_TOOL_INTERFACE, {"exception": "ValidationError", "message": "schema validation failed for path"}),
            _loop_end("completed"),
        ]
    )
    d = diagnose_graph(g)
    assert d.root_layer == "tool_interface"
    assert "validation" in d.mechanism.lower()


def test_shell_error_maps_to_execution_environment():
    g = _graph(
        [
            (KIND_TOOL_ERROR, LAYER_TOOL_INTERFACE, {"exception": "TimeoutExpired", "message": "command timed out"}),
            _loop_end("completed"),
        ]
    )
    d = diagnose_graph(g)
    assert d.root_layer == "execution_environment"


def test_security_rejection_maps_to_governance():
    g = _graph(
        [
            (KIND_TOOL_ERROR, LAYER_TOOL_INTERFACE, {"exception": "SecurityViolationError", "message": "command 'rm' is not allowed"}),
            _loop_end("completed"),
        ]
    )
    d = diagnose_graph(g)
    assert d.root_layer == "governance"
    assert "security policy rejected tool call" == d.mechanism


def test_truncation_warning_in_model_output_is_not_context_pressure():
    """Regression (2026-08-25 corpus): all five "context layer" diagnoses were
    bogus - the signature matched llm_response.text, i.e. the model QUOTING
    the tracer's storage marker "...[truncated N chars]" in its own chat
    output.  Free-text model output is never a system diagnostic."""
    g = _graph(
        [
            (KIND_LLM_RESPONSE, LAYER_CONTEXT, {"text": "WARNING: conversation truncated at 8192 tokens"}),
            _loop_end("completed"),
        ]
    )
    d = diagnose_graph(g)
    assert d.root_layer != "context"


def test_truncation_marker_in_system_note_maps_to_context():
    """A SYSTEM event carrying truncation wording IS context pressure: the
    signature matches guard/loop notes, not free text."""
    g = _graph(
        [
            (
                KIND_GUARD_TRIGGERED,
                LAYER_LIFECYCLE,
                {"guard": GUARD_BUDGET, "note": "history truncated to fit the token budget"},
            ),
            _loop_end("completed"),
        ]
    )
    d = diagnose_graph(g)
    assert d.root_layer == "context"
    assert "history truncation / token limit pressure" == d.mechanism


def test_verification_failure_maps_to_verification():
    g = _graph(
        [
            (KIND_TOOL_RESULT, LAYER_VERIFICATION, {"tool": "tests", "result": "verification failed: expected 3, got 2"}),
            _loop_end("completed"),
        ]
    )
    d = diagnose_graph(g)
    assert d.root_layer == "verification"


def test_repeated_identical_call_maps_to_lifecycle_stuck():
    g = _graph(
        [
            (KIND_TOOL_CALL, LAYER_TOOL_INTERFACE, {"tool": "search", "args_hash": "h"}),
            (KIND_TOOL_CALL, LAYER_TOOL_INTERFACE, {"tool": "search", "args_hash": "h"}),
            (KIND_TOOL_CALL, LAYER_TOOL_INTERFACE, {"tool": "search", "args_hash": "h"}),
            _loop_end("stuck"),
        ]
    )
    d = diagnose_graph(g)
    assert d.root_layer == "lifecycle"
    assert "identical tool call" in d.mechanism


def test_non_consecutive_repeats_are_not_a_stuck_cycle():
    """Regression: the same call repeated 5x *interleaved* with other calls
    (e.g. paging a large file) is not a stuck cycle.  The harness's own
    stuck detector never fired (no duplicate=true, no guard=stuck), so the
    diagnosis must fall through to the guard that actually ended the loop."""
    g = _graph(
        [
            (KIND_TOOL_CALL, LAYER_TOOL_INTERFACE, {"tool": "read", "args_hash": "a", "duplicate": False}),
            (KIND_TOOL_CALL, LAYER_TOOL_INTERFACE, {"tool": "read", "args_hash": "b", "duplicate": False}),
            (KIND_TOOL_CALL, LAYER_TOOL_INTERFACE, {"tool": "read", "args_hash": "a", "duplicate": False}),
            (KIND_TOOL_CALL, LAYER_TOOL_INTERFACE, {"tool": "read", "args_hash": "c", "duplicate": False}),
            (KIND_TOOL_CALL, LAYER_TOOL_INTERFACE, {"tool": "read", "args_hash": "a", "duplicate": False}),
            (KIND_TOOL_CALL, LAYER_TOOL_INTERFACE, {"tool": "read", "args_hash": "b", "duplicate": False}),
            (KIND_TOOL_CALL, LAYER_TOOL_INTERFACE, {"tool": "read", "args_hash": "a", "duplicate": False}),
            (KIND_TOOL_CALL, LAYER_TOOL_INTERFACE, {"tool": "read", "args_hash": "a", "duplicate": False}),
            ("guard_triggered", LAYER_LIFECYCLE, {"guard": "no_mutation", "note": "nudge"}),
            ("guard_triggered", LAYER_LIFECYCLE, {"guard": "no_mutation", "note": "force"}),
            _loop_end("no_progress"),
        ]
    )
    d = diagnose_graph(g)
    assert d.root_layer == "lifecycle"
    assert "stuck cycle" not in d.mechanism
    assert "no_mutation" in d.mechanism


def test_stuck_guard_event_maps_to_lifecycle_stuck():
    g = _graph(
        [
            (KIND_TOOL_CALL, LAYER_TOOL_INTERFACE, {"tool": "search", "args_hash": "h", "duplicate": False}),
            (KIND_TOOL_CALL, LAYER_TOOL_INTERFACE, {"tool": "search", "args_hash": "h", "duplicate": True}),
            (KIND_TOOL_CALL, LAYER_TOOL_INTERFACE, {"tool": "search", "args_hash": "h", "duplicate": True}),
            ("guard_triggered", LAYER_LIFECYCLE, {"guard": "stuck", "note": "stop"}),
            _loop_end("stuck"),
        ]
    )
    d = diagnose_graph(g)
    assert d.root_layer == "lifecycle"
    assert "identical tool call" in d.mechanism
    assert "stuck cycle" in d.mechanism


def test_budget_guard_maps_to_lifecycle():
    g = _graph(
        [
            ("guard_triggered", LAYER_LIFECYCLE, {"guard": "budget_exhausted", "note": "no more tools"}),
            _loop_end("budget_exhausted"),
        ]
    )
    d = diagnose_graph(g)
    assert d.root_layer == "lifecycle"
    assert "budget_exhausted" in d.mechanism


def test_diagnosis_has_evidence_and_confidence():
    g = _graph(
        [
            (KIND_TOOL_ERROR, LAYER_TOOL_INTERFACE, {"exception": "ValidationError", "message": "validation failed"}),
            _loop_end("completed"),
        ]
    )
    d = diagnose_graph(g)
    assert isinstance(d, Diagnosis)
    assert d.confidence > 0.0
    assert d.evidence  # at least one step/link reference
    assert d.repair_proposal
    assert d.root_layer in {"tool_interface", "execution_environment", "governance", "verification", "context", "lifecycle"}


def test_abandonment_detected_via_affected_files():
    """Decision #052: files were mutated (decision #049 affected_files) yet
    the run ended non-completed — the model stopped mid-task."""
    g = _graph(
        [
            (KIND_TOOL_CALL, LAYER_TOOL_INTERFACE, {"tool": "write", "args_hash": "w"}),
            (KIND_TOOL_RESULT, LAYER_TOOL_INTERFACE, {"tool": "write", "affected_files": ["a.py"]}),
            (KIND_TOOL_CALL, LAYER_TOOL_INTERFACE, {"tool": "read", "args_hash": "r"}),
            (KIND_TOOL_RESULT, LAYER_TOOL_INTERFACE, {"tool": "read", "affected_files": ["b.py"]}),
            _loop_end("cap"),
        ]
    )
    d = diagnose_graph(g)
    assert d.root_layer == "lifecycle"
    assert "mutating 2 file(s)" in d.mechanism
    assert "a.py" in d.mechanism
    assert g.affected_files() == ["a.py", "b.py"]


def test_interrupted_run_without_loop_end_is_failed():
    """A trace with >=3 events but no loop_end is an interrupted run (the
    loop always writes loop_end via finally) — diagnosed as lifecycle."""
    g = _graph(
        [
            (KIND_TOOL_CALL, LAYER_TOOL_INTERFACE, {"tool": "read", "args_hash": "r"}),
            (KIND_TOOL_RESULT, LAYER_TOOL_INTERFACE, {"tool": "read"}),
            (KIND_TOOL_CALL, LAYER_TOOL_INTERFACE, {"tool": "search", "args_hash": "s"}),
        ]
    )
    assert not g.has_loop_end()
    d = diagnose_graph(g)
    assert d.root_layer == "lifecycle"
    assert "no loop_end" in d.mechanism


def test_stuck_mechanism_names_the_repeating_tool():
    g = _graph(
        [
            (KIND_TOOL_CALL, LAYER_TOOL_INTERFACE, {"tool": "search", "args_hash": "h", "duplicate": False}),
            (KIND_TOOL_CALL, LAYER_TOOL_INTERFACE, {"tool": "search", "args_hash": "h", "duplicate": True}),
            (KIND_TOOL_CALL, LAYER_TOOL_INTERFACE, {"tool": "search", "args_hash": "h", "duplicate": True}),
            ("guard_triggered", LAYER_LIFECYCLE, {"guard": "stuck", "note": "stop"}),
            _loop_end("stuck"),
        ]
    )
    d = diagnose_graph(g)
    assert "stuck cycle" in d.mechanism
    assert "(search)" in d.mechanism


def test_corpus_flags_interrupted_traces_but_not_stubs(tmp_path):
    """diagnose_corpus: >=3 events without loop_end is failed; 1-2 event
    stubs (aborted demo writes) are not (decision #052)."""
    from harnessfix.corpus import diagnose_corpus
    from harnessfix.tracing import KIND_TOOL_CALL, KIND_TOOL_RESULT, TraceWriter

    real = tmp_path / "traces"
    real.mkdir()
    writer = TraceWriter(task_id="interrupted", directory=real)
    writer.emit({"kind": KIND_TOOL_CALL, "layer": "tool_interface", "tool": "read"})
    writer.emit({"kind": KIND_TOOL_RESULT, "layer": "tool_interface", "tool": "read"})
    writer.emit({"kind": KIND_TOOL_CALL, "layer": "tool_interface", "tool": "read"})
    writer.close()

    stub = TraceWriter(task_id="stub", directory=real)
    stub.emit({"kind": KIND_TOOL_CALL, "layer": "tool_interface", "tool": "read"})
    stub.close()

    ok = TraceWriter(task_id="ok", directory=real)
    ok.emit({"kind": KIND_TOOL_CALL, "layer": "tool_interface", "tool": "read"})
    ok.emit({"kind": KIND_TOOL_RESULT, "layer": "tool_interface", "tool": "read"})
    ok.emit({"kind": "loop_end", "layer": "lifecycle", "outcome": "completed"})
    ok.close()

    diags = diagnose_corpus([p for p in real.glob("*.jsonl")], tmp_path / "diags")
    ids = {d.task_id for d in diags}
    assert ids == {"interrupted"}


def test_tool_result_file_content_does_not_trigger_signatures():
    """Regression: a read result whose FILE CONTENT merely mentions
    "truncat..." produced a bogus context-layer diagnosis (task
    a669a26e...).  Signatures must only match system-diagnostic kinds."""
    g = _graph(
        [
            (KIND_TOOL_RESULT, LAYER_TOOL_INTERFACE,
             {"tool": "read", "text": "the file mentions truncation handling here"}),
            (KIND_TOOL_CALL, LAYER_TOOL_INTERFACE, {"tool": "read", "args_hash": "x", "duplicate": True}),
            (KIND_TOOL_CALL, LAYER_TOOL_INTERFACE, {"tool": "read", "args_hash": "x", "duplicate": True}),
            ("guard_triggered", LAYER_LIFECYCLE, {"guard": "stuck", "note": "stop"}),
            _loop_end("stuck"),
        ]
    )
    d = diagnose_graph(g)
    assert d.root_layer != "context"
    assert "truncation" not in d.mechanism


def test_guard_terminated_run_with_final_answer_is_delivered(tmp_path):
    """A stuck-guard run that still produced a substantive final answer
    delivered the task — it must NOT be diagnosed/failed."""
    from harnessfix.corpus import diagnose_corpus
    from harnessfix.tracing import TraceWriter

    real = tmp_path / "traces"
    real.mkdir()

    delivered = TraceWriter(task_id="delivered", directory=real)
    delivered.emit({"kind": "tool_call", "layer": "tool_interface",
                    "tool": "run", "args_hash": "h", "duplicate": True})
    delivered.emit({"kind": "guard_triggered", "layer": "lifecycle",
                    "guard": "stuck", "note": "stop now"})
    delivered.emit({"kind": "llm_response", "layer": "observability",
                    "text": "## Final Answer\n\nHere is the complete analysis of why the two "
                            "workflow runs differ, with all findings and a recommendation."})
    delivered.emit({"kind": "loop_end", "layer": "lifecycle", "outcome": "stuck"})
    delivered.close()

    empty = TraceWriter(task_id="empty", directory=real)
    empty.emit({"kind": "tool_call", "layer": "tool_interface",
                "tool": "run", "args_hash": "h", "duplicate": True})
    empty.emit({"kind": "guard_triggered", "layer": "lifecycle",
                "guard": "stuck", "note": "stop now"})
    empty.emit({"kind": "loop_end", "layer": "lifecycle", "outcome": "stuck"})
    empty.close()

    diags = diagnose_corpus([p for p in real.glob("*.jsonl")], tmp_path / "diags")
    assert {d.task_id for d in diags} == {"empty"}
