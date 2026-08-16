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
    TraceSink,
    TraceWriter,
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
    "TraceGraph",
    "TraceSink",
    "TraceValidationError",
    "TraceWriter",
    "compile_trace",
    "diagnose_trace",
    "infer_links",
    "read_trace",
    "task_id_of",
    "trace_enabled",
]
