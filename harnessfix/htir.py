"""Phase 1 - Harness-aware Trace Intermediate Representation (HTIR).

Compiles validated trace events into a layer-faceted graph of steps with
provenance and control-flow links (spec section 3.2).  A TraceGraph is fully
reproducible from a trace file alone: steps mirror the event stream and links
are inferred deterministically.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .reader import read_trace

LinkKind = Literal["provenance", "control_flow"]

#: loop_end outcomes that count as a failed task (spec: failed traces are
#: the ones we diagnose and repair).
_OK_OUTCOMES = frozenset({"completed"})


class HTIRStep(BaseModel):
    """One trace event with its layer facet and graph links."""

    index: int
    kind: str
    layer_facet: str
    payload: dict[str, Any]
    links: list[str] = Field(default_factory=list)

    def is_failed(self) -> bool:
        if self.kind == "tool_error":
            return True
        if self.kind == "loop_end":
            return self.payload.get("outcome", "completed") not in _OK_OUTCOMES
        return False


class HTIRLink(BaseModel):
    """A directed edge between two steps (provenance or control-flow)."""

    link_id: str
    kind: LinkKind
    source: int
    target: int
    detail: str = ""


class TraceGraph(BaseModel):
    """Layer-faceted trace graph: ordered steps plus inferred links."""

    task_id: str
    steps: list[HTIRStep]
    links: list[HTIRLink] = Field(default_factory=list)

    def steps_of_kind(self, kind: str) -> list[HTIRStep]:
        return [s for s in self.steps if s.kind == kind]

    def failed_steps(self) -> list[HTIRStep]:
        return [s for s in self.steps if s.is_failed()]

    def step(self, index: int) -> HTIRStep | None:
        for s in self.steps:
            if s.index == index:
                return s
        return None


def compile_trace(path: Path | str) -> TraceGraph:
    """Compile a validated trace JSONL file into a TraceGraph.

    Steps mirror the event stream; provenance and control-flow links are
    attached by harnessfix.links.infer_links (imported lazily to keep this
    module free of cycles).
    """
    from .links import infer_links  # deferred: links imports the HTIR models

    events = read_trace(path)
    steps = [
        HTIRStep(
            index=idx,
            kind=str(ev["kind"]),
            layer_facet=str(ev["layer"]),
            payload={k: v for k, v in ev.items() if k not in ("kind", "layer", "task_id", "ts", "correlation_id")},
        )
        for idx, ev in enumerate(events)
    ]
    task_id = str(events[0].get("task_id", "")) if events else ""
    graph = TraceGraph(task_id=task_id, steps=steps)
    for link in infer_links(graph):
        graph.links.append(link)
        src = graph.step(link.source)
        tgt = graph.step(link.target)
        if src is not None:
            src.links.append(link.link_id)
        if tgt is not None:
            tgt.links.append(link.link_id)
    return graph
