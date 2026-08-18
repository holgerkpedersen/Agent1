"""Phase 0 - JSONL trace capture for ToolLoopRunner runs.

One JSONL trace per task at reports/traces/{task_id}.jsonl.  Every event is
tagged with a harness layer facet and the active correlation id (single
canonical source: agent_core.context_management.CORRELATION_ID_CTX).

Tracing is opt-OUT: AGENT_NO_TRACE=1 disables it.  When no TraceWriter is
attached to a run, byte-identical behaviour is preserved (zero change to the
LLM request payloads).
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Protocol

from agent_core.context_management import CORRELATION_ID_CTX

logger = logging.getLogger(__name__)

TRACE_DIR = Path("reports") / "traces"

#: HarnessFix layer facets - every event carries exactly one.
LAYER_EXECUTION = "execution_environment"
LAYER_TOOL_INTERFACE = "tool_interface"
LAYER_CONTEXT = "context"
LAYER_LIFECYCLE = "lifecycle"
LAYER_OBSERVABILITY = "observability"
LAYER_VERIFICATION = "verification"
LAYER_GOVERNANCE = "governance"
LAYERS: frozenset[str] = frozenset(
    {
        LAYER_EXECUTION,
        LAYER_TOOL_INTERFACE,
        LAYER_CONTEXT,
        LAYER_LIFECYCLE,
        LAYER_OBSERVABILITY,
        LAYER_VERIFICATION,
        LAYER_GOVERNANCE,
    }
)

#: Event kinds emitted by the instrumented tool loop (spec section 3.1).
KIND_STEP_START = "step_start"
KIND_LLM_RESPONSE = "llm_response"
KIND_TOOL_CALL = "tool_call"
KIND_TOOL_RESULT = "tool_result"
KIND_TOOL_ERROR = "tool_error"
KIND_GUARD_TRIGGERED = "guard_triggered"
KIND_LOOP_END = "loop_end"
#: Emitted once per task (by the caller, before the loop starts): carries
#: the user prompt, model and profile so a trace is self-describing for the
#: human review gate and cross-model corpus comparison (decision #050).
KIND_TASK_BEGIN = "task_begin"
KINDS: frozenset[str] = frozenset(
    {
        KIND_STEP_START,
        KIND_LLM_RESPONSE,
        KIND_TOOL_CALL,
        KIND_TOOL_RESULT,
        KIND_TOOL_ERROR,
        KIND_GUARD_TRIGGERED,
        KIND_LOOP_END,
        KIND_TASK_BEGIN,
    }
)

#: Guards whose firing must be recorded together with the injected note text.
GUARD_DEADLINE = "deadline"
GUARD_STUCK = "stuck"
GUARD_NO_MUTATION = "no_mutation"
GUARD_BUDGET = "budget_exhausted"

#: Caps on what is persisted.  The model ALWAYS receives the full payload;
#: these bounds only keep trace files small (spec: "result truncated to a cap").
RESULT_CAP = 2000
TEXT_CAP = 1000
#: Cap for the user prompt stored in the task_begin event — prompts can be
#: pasted documents; the trace only needs enough to identify the task.
PROMPT_CAP = 500


class TraceSink(Protocol):
    """Minimal contract the tool loop uses to emit trace events."""

    def emit(self, event: dict[str, Any]) -> None: ...


def trace_enabled() -> bool:
    """Opt-out toggle: AGENT_NO_TRACE=1 (or true/yes/on) disables tracing."""
    raw = os.environ.get("AGENT_NO_TRACE", "").strip().lower()
    return raw not in ("1", "true", "yes", "on")


def truncate(text: str, cap: int) -> str:
    """Bound a payload field for the trace file (full text still reaches LLM)."""
    if len(text) <= cap:
        return text
    return text[:cap] + f"...[truncated {len(text) - cap} chars]"


class TraceWriter:
    """Appends JSONL events for one task to reports/traces/{task_id}.jsonl.

    emit() never raises: trace capture must not change loop behaviour.

    ``meta`` is stamped onto EVERY record (model, profile, ...) so a trace
    is self-describing; the caller emits a task_begin event carrying the
    user prompt via ``emit_task_begin``.
    """

    def __init__(
        self,
        task_id: str | None = None,
        directory: Path | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.task_id = task_id or uuid.uuid4().hex
        self.directory = directory or TRACE_DIR
        self.path = self.directory / f"{self.task_id}.jsonl"
        self._meta = dict(meta or {})
        self._closed = False

    def emit_task_begin(self, user_input: str) -> None:
        """Record the task prompt (model/profile arrive via ``meta``)."""
        self.emit(
            {
                "kind": KIND_TASK_BEGIN,
                "layer": LAYER_CONTEXT,
                "user_input": truncate(user_input, PROMPT_CAP),
            }
        )

    def emit(self, event: dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            record: dict[str, Any] = {
                "task_id": self.task_id,
                "ts": time.time(),
                "correlation_id": CORRELATION_ID_CTX.get(),
                **self._meta,
                **event,
            }
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            logger.warning("trace write failed for %s: %s", self.task_id, exc)

    def close(self) -> None:
        self._closed = True
