"""Chunk plan execution so each LLM call stays in a small memory envelope.

Plan tasks carry free-form text: task descriptions and upstream dependency
results are pasted into the subagent prompt verbatim, and full results are
persisted to disk.  A big model's verbose result therefore inflates every
downstream prompt, and the scheduler keeps every result in memory for the
whole run.  This module adds the missing bounds:

* :func:`summarize_result` — cap a task result before it is persisted or fed
  into downstream prompts (full text stays available via the provider's own
  context trimmer; only what the *next* prompt embeds is bounded).
* :func:`chunk_boundaries` / :func:`plan_chunks` — split a topological order
  into waves of at most ``max_concurrent`` tasks, so at most one wave's worth
  of results is live at a time.

Environment knobs (all optional):

* ``AGENT_PLAN_MAX_SUMMARY_CHARS``   per-task summary cap fed downstream (default 2000)
* ``AGENT_PLAN_MAX_CONCURRENT``      max tasks in flight / retained per wave (default 3)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence

#: Default character cap for the summary a completed task feeds downstream.
DEFAULT_MAX_SUMMARY_CHARS = 2000

#: Default number of tasks executed (and retained) per wave.
DEFAULT_MAX_CONCURRENT = 3

_TRUNCATION_SUFFIX = "\n...[summary truncated to fit the next model's context]"


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, ""))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def max_summary_chars() -> int:
    """Per-task summary cap fed downstream (``AGENT_PLAN_MAX_SUMMARY_CHARS``)."""
    return _env_int("AGENT_PLAN_MAX_SUMMARY_CHARS", DEFAULT_MAX_SUMMARY_CHARS)


def max_concurrent() -> int:
    """Tasks executed per wave (``AGENT_PLAN_MAX_CONCURRENT``)."""
    return _env_int("AGENT_PLAN_MAX_CONCURRENT", DEFAULT_MAX_CONCURRENT)


def summarize_result(
    result: Dict[str, Any], *, max_chars: int | None = None
) -> Dict[str, Any]:
    """Return a copy of *result* with bounded ``summary``/``result`` text.

    Only the string fields downstream prompts embed are truncated; every other
    key (status, counts, file lists) is preserved verbatim.  A summary that
    was cut is marked so consumers can tell it apart from a complete one.
    """
    cap = max_summary_chars() if max_chars is None else max(1, int(max_chars))
    bounded: Dict[str, Any] = {}
    for key, value in result.items():
        if key in ("summary", "result") and isinstance(value, str):
            if len(value) > cap:
                bounded[key] = value[:cap] + _TRUNCATION_SUFFIX
                if key == "summary":
                    bounded["summary_truncated"] = True
            else:
                bounded[key] = value
        else:
            bounded[key] = value
    return bounded


def chunk_boundaries(
    total: int, *, max_concurrent_tasks: int | None = None
) -> List[tuple[int, int]]:
    """Split ``range(total)`` into contiguous ``(start, end)`` waves.

    Each wave holds at most *max_concurrent_tasks* tasks; an empty plan yields
    no waves.
    """
    size = (
        max_concurrent()
        if max_concurrent_tasks is None
        else max(1, int(max_concurrent_tasks))
    )
    return [(i, min(i + size, total)) for i in range(0, total, size)]


def plan_chunks(
    order: Sequence[str], *, max_concurrent_tasks: int | None = None
) -> List[List[str]]:
    """Group a topological *order* into waves of at most N tasks each.

    Chunking a topological order in contiguous groups is safe: a wave's tasks
    depend only on earlier tasks, which live in earlier (already finished)
    waves.
    """
    size = (
        max_concurrent()
        if max_concurrent_tasks is None
        else max(1, int(max_concurrent_tasks))
    )
    return [
        list(order[i:j])
        for i, j in chunk_boundaries(len(order), max_concurrent_tasks=size)
    ]


__all__: list[str] = [
    "DEFAULT_MAX_CONCURRENT",
    "DEFAULT_MAX_SUMMARY_CHARS",
    "chunk_boundaries",
    "max_concurrent",
    "max_summary_chars",
    "plan_chunks",
    "summarize_result",
]
