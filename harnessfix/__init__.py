"""harnessfix - trace-grounded harness repair for Agent1.

Implements the HarnessFix loop from docs/HARNESSFIX_SPEC.md: capture
per-task tool-loop traces, compile them into a layer-faceted trace graph
(HTIR), diagnose failures, apply scoped code-level repairs, and re-run the
benchmark/tests gates to accept or reject each repair.
"""

from .tracing import (
    GUARD_BUDGET,
    GUARD_DEADLINE,
    GUARD_NO_MUTATION,
    GUARD_STUCK,
    KIND_LLM_RESPONSE,
    KIND_LOOP_END,
    KIND_STEP_START,
    KIND_TOOL_CALL,
    KIND_TOOL_ERROR,
    KIND_TOOL_RESULT,
    LAYERS,
    TURN_EVENTS_CAP,
    TraceSink,
    TraceWriter,
    begin_turn,
    current_turn_events,
    drain_turn_events,
    take_turn_events,
    trace_enabled,
)
from .reader import TraceValidationError, read_trace, task_id_of
from .htir import HTIRLink, HTIRStep, TraceGraph, compile_trace
from .links import infer_links
from .diagnose import diagnose_trace

__all__ = [
    "GUARD_BUDGET",
    "GUARD_DEADLINE",
    "GUARD_NO_MUTATION",
    "GUARD_STUCK",
    "HTIRLink",
    "HTIRStep",
    "KIND_LLM_RESPONSE",
    "KIND_LOOP_END",
    "KIND_STEP_START",
    "KIND_TOOL_CALL",
    "KIND_TOOL_ERROR",
    "KIND_TOOL_RESULT",
    "LAYERS",
    "TURN_EVENTS_CAP",
    "TraceGraph",
    "TraceSink",
    "TraceValidationError",
    "TraceWriter",
    "begin_turn",
    "compile_trace",
    "current_turn_events",
    "diagnose_trace",
    "drain_turn_events",
    "infer_links",
    "read_trace",
    "take_turn_events",
    "task_id_of",
    "trace_enabled",
]
