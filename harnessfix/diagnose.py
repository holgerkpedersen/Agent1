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
    (LAYER_CONTEXT, "history truncation / token limit pressure", "text", "truncat", True),
]


class Diagnosis(BaseModel):
    """Result of diagnosing one failed trace."""

    task_id: str
    root_layer: str
    mechanism: str
    evidence: list[str]
    confidence: float
    repair_proposal: str


def _find_signature(payload: dict[str, Any]) -> tuple[str, str] | None:
    """First matching signature for a single event payload, if any."""
    for layer, mechanism, field, needle, casefold in _SIGNATURES:
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
    stuck_counts: dict[str, int] = {}
    tool_errors: list[dict[str, Any]] = []

    for step in graph.steps:
        payload = step.payload
        if step.kind == "guard_triggered":
            guards.append(str(payload.get("guard", "")))
        if step.kind == "tool_error":
            tool_errors.append(payload)
        if step.kind == "tool_call":
            key = f"{payload.get('tool', '')}:{payload.get('args_hash', '')}"
            stuck_counts[key] = stuck_counts.get(key, 0) + 1

    # 1. Explicit signatures (tool errors, verification, context pressure).
    for step in graph.steps:
        hit = _find_signature(step.payload)
        if hit is not None:
            layer, mechanism = hit
            return _build(graph, layer, mechanism, step.index)

    # 2. Repeated identical tool call >= 3 times -> lifecycle (stuck).
    worst = max(stuck_counts.values(), default=0)
    if worst >= 3:
        return _build(
            graph,
            LAYER_LIFECYCLE,
            f"model repeated an identical tool call {worst}x (stuck cycle)",
            first_index=0,
        )

    # 3. Any guard fired -> lifecycle (deadline / stuck / no-mutation / budget).
    if guards:
        return _build(graph, LAYER_LIFECYCLE, f"lifecycle guard fired: {', '.join(guards)}")

    # 4. Tool errors without a matching signature -> tool interface.
    if tool_errors:
        return _build(graph, LAYER_TOOL_INTERFACE, "unclassified tool error", first_index=0)

    return _build(
        graph,
        LAYER_LIFECYCLE,
        "loop did not complete; no signature matched (fallback)",
        first_index=0,
    )


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
