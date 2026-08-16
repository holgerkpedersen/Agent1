"""Phase 2 tests: heuristic signature classification (spec section 3.3)."""
from __future__ import annotations

from harnessfix.diagnose import Diagnosis, diagnose_graph
from harnessfix.htir import HTIRStep, TraceGraph
from harnessfix.tracing import (
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


def test_truncation_warning_maps_to_context():
    g = _graph(
        [
            (KIND_LLM_RESPONSE, LAYER_CONTEXT, {"text": "WARNING: conversation truncated at 8192 tokens"}),
            _loop_end("completed"),
        ]
    )
    d = diagnose_graph(g)
    assert d.root_layer == "context"


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
