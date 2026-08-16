"""Phase 1 - provenance and control-flow link inference (spec section 3.2).

Provenance links: for each tool call, search backward through earlier tool
results for tokens (file paths / identifiers) the call's arguments reuse.
Control-flow links: a step caused by harness logic (e.g. a step that follows
an injected guard note, or the third identical call in a stuck cycle).
"""
from __future__ import annotations

import re
from typing import Any

from .htir import HTIRLink, TraceGraph

_MAX_LINKS = 40
#: Minimum length of a token considered evidence of semantic reuse.
_MIN_TOKEN = 6


def _significant_tokens(text: str) -> set[str]:
    """Lowercased alnum tokens of meaningful length, deduplicated."""
    tokens = {t for t in re.findall(r"[A-Za-z0-9_.\-/\\]+", text) if len(t) >= _MIN_TOKEN}
    return {t.lower() for t in tokens}


def _args_text(step_payload: dict[str, Any]) -> str:
    return str(step_payload.get("args", "")) + str(step_payload.get("args_hash", ""))


def _infer_provenance(graph: TraceGraph) -> list[HTIRLink]:
    """Tool calls that reuse text from an earlier tool result."""
    links: list[HTIRLink] = []
    results: list[tuple[int, set[str]]] = []  # (step index, token set)
    for step in graph.steps:
        if step.kind == "tool_result" and not step.payload.get("duplicate"):
            results.append((step.index, _significant_tokens(str(step.payload.get("result", "")))))
        elif step.kind == "tool_call":
            args_tokens = _significant_tokens(_args_text(step.payload))
            if not args_tokens:
                continue
            for res_idx, res_tokens in reversed(results):
                shared = args_tokens & res_tokens
                if shared and len(links) < _MAX_LINKS:
                    links.append(
                        HTIRLink(
                            link_id=f"prov-{res_idx}-{step.index}",
                            kind="provenance",
                            source=res_idx,
                            target=step.index,
                            detail="args reuse tokens from earlier result: "
                            + ", ".join(sorted(shared)[:3]),
                        )
                    )
                    break  # only the nearest result per call
    return links


def _infer_control_flow(graph: TraceGraph) -> list[HTIRLink]:
    """Steps caused by harness logic: guard notes and stuck cycles."""
    links: list[HTIRLink] = []
    for idx, step in enumerate(graph.steps):
        if step.kind == "guard_triggered" and idx + 1 < len(graph.steps):
            nxt = graph.steps[idx + 1]
            links.append(
                HTIRLink(
                    link_id=f"ctrl-{step.index}-{nxt.index}",
                    kind="control_flow",
                    source=step.index,
                    target=nxt.index,
                    detail=f"step follows injected guard note ({step.payload.get('guard', '')})",
                )
            )
    # Stuck cycle: the third consecutive identical call is forced to stop.
    calls = [s for s in graph.steps if s.kind == "tool_call"]
    for a, b in zip(calls, calls[1:]):
        if (
            a.payload.get("duplicate")
            and b.payload.get("duplicate")
            and a.payload.get("args_hash") == b.payload.get("args_hash")
            and len(links) < _MAX_LINKS
        ):
            links.append(
                HTIRLink(
                    link_id=f"ctrl-stuck-{a.index}-{b.index}",
                    kind="control_flow",
                    source=a.index,
                    target=b.index,
                    detail="repeated identical tool call (stuck cycle)",
                )
            )
    return links[: _MAX_LINKS]


def infer_links(graph: TraceGraph) -> list[HTIRLink]:
    """All links for a trace graph (provenance + control-flow, capped)."""
    return _infer_provenance(graph) + _infer_control_flow(graph)
