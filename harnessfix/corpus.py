"""Phase 4 - trace corpus helpers for the HarnessFix loop.

Collecting, compiling, diagnosing and layer-grouping a corpus of trace files.
Kept separate from loop.py so each module stays small and testable.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .diagnose import Diagnosis, diagnose_graph
from .htir import TraceGraph, compile_trace
from .reader import TraceValidationError
from .repairs import Repair, repairs_for_layer

#: A trace without loop_end is an interrupted run (the loop ALWAYS writes
#: loop_end via finally), but only real sessions count — 1-2-event stubs
#: (aborted demo writes, empty writers) are noise, not failures.
MIN_ACTIVITY_EVENTS = 3


def _is_failed_trace(graph: TraceGraph) -> bool:
    """Failed = tool error, interrupted run (no loop_end, >=3 events), or a
    guard-terminated run (outcome != completed) that did NOT deliver a
    substantive final answer.  A run ending stuck/cap/no_progress but with a
    full answer delivered the task — the guards stop the LOOP, not the
    answer (decision #052 refinement, task a669a26e...)."""
    if any(s.kind == "tool_error" for s in graph.steps):
        return True
    outcome = ""
    for s in graph.steps:
        if s.kind == "loop_end":
            outcome = str(s.payload.get("outcome", "completed"))
    if not outcome:
        return len(graph.steps) >= MIN_ACTIVITY_EVENTS
    if outcome == "completed":
        return False
    return not graph.has_final_answer()


def collect_traces(trace_dir: Path) -> list[Path]:
    """Sorted .jsonl trace files under a directory (empty if none)."""
    if not trace_dir.is_dir():
        return []
    return sorted(trace_dir.glob("*.jsonl"))


def diagnose_corpus(traces: list[Path], output_dir: Path) -> list[Diagnosis]:
    """Compile+diagnose every FAILED trace; persist one JSON per task.

    Successful traces (completed loop, no tool errors) are skipped: diagnosis
    is for failed traces only (spec section 3.5 step 2).  Interrupted runs
    (no loop_end event) count as failed too (decision #052).
    """
    diagnoses: list[Diagnosis] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in traces:
        try:
            graph = compile_trace(path)
        except TraceValidationError:
            continue
        if not _is_failed_trace(graph):
            continue
        diag = diagnose_graph(graph)
        diagnoses.append(diag)
        (output_dir / f"{diag.task_id}.json").write_text(
            diag.model_dump_json(indent=2), encoding="utf-8"
        )
    return diagnoses


def layer_counts(diagnoses: list[Diagnosis]) -> Counter[str]:
    """Frequency of each root harness layer across the corpus."""
    return Counter(d.root_layer for d in diagnoses)


def choose_repair(counts: Counter[str]) -> Repair | None:
    """First catalog repair of the highest-frequency diagnosed layer."""
    for layer, _count in counts.most_common():
        available = repairs_for_layer(layer)
        if available:
            return available[0]
    return None
