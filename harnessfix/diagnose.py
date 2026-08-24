"""Phase 2 - heuristic failure diagnosis (spec section 3.3, tier 1).

Signature-based classification: each failed trace maps to exactly one harness
layer facet plus a mechanism, with evidence (link ids / step indices) and a
repair proposal.  The LLM-agent tier is only activated later if the heuristic
tier's precision falls below ~70% on a labeled set.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .htir import TraceGraph, compile_trace
from .tracing import (
    GUARD_STUCK,
    KIND_LLM_RESPONSE,
    LAYER_CONTEXT,
    LAYER_EXECUTION,
    LAYER_GOVERNANCE,
    LAYER_LIFECYCLE,
    LAYER_TOOL_INTERFACE,
    LAYER_VERIFICATION,
)

#: (facet, mechanism) results with the payload fields that trigger them.
_SIGNATURES: list[tuple[str, str, str, str, bool]] = [
    # (layer, mechanism, field, needle-substring, case_insensitive)
    (LAYER_TOOL_INTERFACE, "schema validation rejected tool args", "message", "validation", True),
    (LAYER_EXECUTION, "subprocess/shell execution failed", "exception", "TimeoutExpired", False),
    (LAYER_EXECUTION, "subprocess/shell execution failed", "exception", "FileNotFoundError", False),
    (LAYER_EXECUTION, "subprocess/shell execution failed", "exception", "PermissionError", False),
    (LAYER_EXECUTION, "subprocess/shell execution failed", "exception", "OSError", False),
    (LAYER_GOVERNANCE, "security policy rejected tool call", "message", "not allowed", True),
    (LAYER_GOVERNANCE, "security policy rejected tool call", "message", "SecurityViolation", False),
    (LAYER_GOVERNANCE, "path escape blocked by sanitizer", "message", "path escape", True),
    (LAYER_VERIFICATION, "benchmark checker failed after completion", "result", "verification failed", True),
    #: Context pressure is a SYSTEM steering note (guard/budget), never the
    #: model's own chat output: llm_response.text routinely QUOTES the
    #: tracer's storage marker "...[truncated N chars]" when the model echoes
    #: file content or its own earlier answer.  All five "context layer"
    #: diagnoses on the 2026-08-25 corpus were exactly that (five hits on
    #: llm_response.text, zero on any system field).
    (LAYER_CONTEXT, "history truncation / token limit pressure", "note", "truncat", True),
]

#: Event kinds whose payloads may carry SYSTEM diagnostics worth signature
#: matching.  tool_call/step_start payloads are excluded entirely (tool args
#: and loop bookkeeping).  tool_result is eligible EXCEPT its "text" field:
#: that field carries FILE CONTENTS — a read of a file that merely mentions
#: "truncation" once produced a bogus context-layer diagnosis (task
#: a669a26e...).  For the same reason llm_response is eligible EXCEPT its
#: free-text "text" field: that is MODEL OUTPUT, not a system diagnostic.
#: Structured result/message/exception/note fields stay eligible.
_SIGNATURE_KINDS = frozenset({"tool_error", "guard_triggered", "llm_response", "loop_end", "tool_result"})


class Diagnosis(BaseModel):
    """Result of diagnosing one failed trace."""

    task_id: str
    root_layer: str
    mechanism: str
    evidence: list[str]
    confidence: float
    repair_proposal: str


def _find_signature(kind: str, payload: dict[str, Any]) -> tuple[str, str] | None:
    """First matching signature for a single event payload, if any."""
    if kind not in _SIGNATURE_KINDS:
        return None
    for layer, mechanism, field, needle, casefold in _SIGNATURES:
        # Free-text fields carry file contents (tool_result) or the model's
        # own output (llm_response); both routinely CONTAIN system marker
        # words without any system event having occurred.
        if kind == "tool_result" and field == "text":
            continue
        if kind == KIND_LLM_RESPONSE and field == "text":
            continue
        value = str(payload.get(field, ""))
        if casefold:
            value, needle = value.lower(), needle.lower()
        if needle in value:
            return layer, mechanism
    return None


def diagnose_graph(graph: TraceGraph) -> Diagnosis:
    """Classify a compiled trace graph against the heuristic signatures."""
    task_id = graph.task_id
    guards: list[str] = []
    guard_indices: list[int] = []
    tool_errors: list[dict[str, Any]] = []

    for step in graph.steps:
        payload = step.payload
        if step.kind == "guard_triggered":
            guards.append(str(payload.get("guard", "")))
            guard_indices.append(step.index)
        if step.kind == "tool_error":
            tool_errors.append(payload)

    # 1. Explicit signatures (tool errors, verification, context pressure).
    for step in graph.steps:
        hit = _find_signature(step.kind, step.payload)
        if hit is not None:
            layer, mechanism = hit
            return _build(graph, layer, mechanism, step.index)

    # 1b. Task abandonment (decision #052): files were actually mutated
    # (decision #049 affected_files) yet the run did not complete — the model
    # wrote/edited then stopped mid-task.  More specific than the stuck/
    # lifecycle heuristics, less specific than an explicit tool error.
    affected = graph.affected_files()
    if affected and not graph.has_loop_end():
        return _build(
            graph,
            LAYER_LIFECYCLE,
            f"task interrupted after mutating {len(affected)} file(s) "
            f"(e.g. {', '.join(affected[:3])})",
            first_index=_first_mutation_index(graph),
        )
    if affected:
        end = next(
            (s for s in reversed(graph.steps) if s.kind == "loop_end"), None
        )
        if end is not None and end.payload.get("outcome") != "completed":
            return _build(
                graph,
                LAYER_LIFECYCLE,
                f"task ended non-completed after mutating {len(affected)} file(s) "
                f"(e.g. {', '.join(affected[:3])})",
                first_index=_first_mutation_index(graph),
            )

    # 2. Stuck cycle -> lifecycle.  Mirrors the harness's own semantics
    #    (tool_loop.py stops the loop on a third *consecutive* identical
    #    call, marks the duplicates and fires the "stuck" guard): the
    #    diagnosis is definitive when that guard fired, and falls back to
    #    three consecutive identical calls in the tool-call sequence.
    #    Non-consecutive repeats are legitimate re-reads (e.g. paging a
    #    large file) and must not be misdiagnosed as a stuck cycle.
    stuck_guard_idx = next(
        (i for g, i in zip(guards, guard_indices) if g == GUARD_STUCK), None
    )
    worst_run, run_start = _max_consecutive_identical_run(graph)
    if stuck_guard_idx is not None or worst_run >= 3:
        count = worst_run if worst_run >= 3 else 3
        tool = _run_tool_at(graph, run_start)
        tool_part = f" ({tool})" if tool else ""
        return _build(
            graph,
            LAYER_LIFECYCLE,
            f"model repeated an identical tool call {count}x{tool_part} (stuck cycle)",
            first_index=stuck_guard_idx if stuck_guard_idx is not None else run_start,
        )

    # 2b. Interrupted run (decision #052): no loop_end — the loop ALWAYS
    #     writes one via finally, so a missing one is a crash/kill/connection
    #     loss, not a normal end.  Stub traces (1-2 events) are filtered by
    #     the corpus before they reach diagnosis.
    if not graph.has_loop_end():
        return _build(
            graph,
            LAYER_LIFECYCLE,
            "run interrupted: no loop_end event (crash / killed / provider loss)",
            first_index=len(graph.steps) - 1,
        )

    # 3. Any guard fired -> lifecycle (deadline / stuck / no-mutation / budget).
    if guards:
        return _build(
            graph,
            LAYER_LIFECYCLE,
            f"lifecycle guard fired: {', '.join(guards)}",
            first_index=guard_indices[-1],
        )

    # 4. Tool errors without a matching signature -> tool interface.
    if tool_errors:
        return _build(graph, LAYER_TOOL_INTERFACE, "unclassified tool error", first_index=0)

    return _build(
        graph,
        LAYER_LIFECYCLE,
        "loop did not complete; no signature matched (fallback)",
        first_index=0,
    )


def _first_mutation_index(graph: TraceGraph) -> int | None:
    """Index of the first tool_result step that reported affected files."""
    for s in graph.steps:
        files = s.payload.get("affected_files")
        if isinstance(files, list) and files:
            return s.index
    return None


def _run_tool_at(graph: TraceGraph, index: int) -> str:
    """Tool name of the nearest tool_call step at/after *index*, if any."""
    for s in graph.steps:
        if s.index >= index and s.kind == "tool_call":
            return str(s.payload.get("tool", ""))
    return ""


def _max_consecutive_identical_run(graph: TraceGraph) -> tuple[int, int]:
    """Longest run of *consecutive* identical tool calls (same tool +
    args_hash) in the tool-call sequence, and the step index where that run
    starts.  Other event kinds between calls (step_start / llm_response /
    tool_result) do not break the run: each loop iteration emits them around
    the call.  Returns (0, 0) when the trace has no tool calls."""
    worst, worst_start = 0, 0
    run, run_start = 0, 0
    prev_key: str | None = None
    for step in graph.steps:
        if step.kind != "tool_call":
            continue
        key = f"{step.payload.get('tool', '')}:{step.payload.get('args_hash', '')}"
        if key == prev_key:
            run += 1
        else:
            run, run_start = 1, step.index
        prev_key = key
        if run > worst:
            worst, worst_start = run, run_start
    return worst, worst_start


def _build(
    graph: TraceGraph,
    layer: str,
    mechanism: str,
    first_index: int | None = None,
) -> Diagnosis:
    """Assemble a Diagnosis with evidence = link ids of failed steps."""
    failed = graph.failed_steps()
    evidence: list[str] = []
    if first_index is not None:
        step = graph.step(first_index)
        if step is not None:
            evidence.extend(step.links)
            if not evidence:
                evidence.append(f"step:{first_index}")
    if not evidence:
        evidence = [ln.link_id for s in failed for ln in graph.links if ln.source == s.index or ln.target == s.index][:6] or [
            f"step:{s.index}" for s in failed
        ][:6]
    confidence = 0.85 if first_index is not None else 0.6
    return Diagnosis(
        task_id=graph.task_id,
        root_layer=layer,
        mechanism=mechanism,
        evidence=evidence,
        confidence=confidence,
        repair_proposal=f"inspect {layer} harness mechanism: {mechanism}",
    )


def diagnose_trace(path: Path | str) -> Diagnosis:
    """Compile a trace file and diagnose it (one-shot convenience API)."""
    return diagnose_graph(compile_trace(path))
