"""Memory command — inspect LLM decision attribution recorded in memory.

Usage:
    memory                       Attributed vs unattributed experience rows,
                                 plus per-model counts / success rates
    memory --latency             Also show the decision latency histogram
    memory --bucket-ms <ms>      Histogram bucket width (default 100)
    memory --json                Machine-readable output (the same roll-up)

Every ``experiences`` row written by ``Agent._record_llm_experience`` is
stamped with the model that *decided* it, and an attributed write also appends
one ``llm_decisions`` provenance row (see :mod:`agent_core.memory.attribution`).
Those analytics existed but had no caller outside the test suite, so the
measurement the feature was built for — "did the model swap help?" — was
unreachable from the REPL.  This command is that read path.

It is strictly read-only: with no database present it prints a message and
returns, and it never creates the DB file as a side effect.  The DB path is
resolved from ``agent.AGENT_MEMORY_JSON_PATH`` at CALL time — the exact same
expression the writer uses — so the reader can never drift from the DB that
was written.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

from agent_core.memory.attribution import (
    attribution_summary,
    decision_latency_histogram,
)

from .base import Command

if TYPE_CHECKING:
    from agent import Agent

logger = logging.getLogger(__name__)

_DEFAULT_BUCKET_MS = 100.0


def _memory_db_path() -> str:
    """Resolve the memory SQLite path exactly as the writer does.

    ``Agent._record_llm_experience`` derives it from the ``agent`` module's
    ``AGENT_MEMORY_JSON_PATH`` global at call time, so reading it the same way
    keeps the two in lockstep (tests redirect that global to a temp dir).
    """
    import agent as agent_module

    return str(agent_module.AGENT_MEMORY_JSON_PATH).replace(".json", ".db")


def _empty_summary() -> dict[str, Any]:
    """The zero roll-up — used when no DB exists, so nothing is ever created."""
    return {
        "experiences": 0,
        "attributed": 0,
        "unattributed": 0,
        "llm_decisions": 0,
        "by_llm": [],
        "success_by_model": [],
    }


def _parse_args(args: list[str]) -> tuple[bool, bool, float] | None:
    """Return ``(as_json, show_latency, bucket_ms)`` or ``None`` on bad input."""
    as_json = False
    show_latency = False
    bucket_ms = _DEFAULT_BUCKET_MS
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--json":
            as_json = True
        elif arg == "--latency":
            show_latency = True
        elif arg == "--bucket-ms":
            index += 1
            if index >= len(args):
                print("  Usage: memory [--json] [--latency] [--bucket-ms <ms>]")
                return None
            try:
                bucket_ms = float(args[index])
            except ValueError:
                print(f"  Error: --bucket-ms needs a number, got {args[index]!r}")
                return None
            if bucket_ms <= 0:
                print("  Error: --bucket-ms must be a positive number of milliseconds")
                return None
        else:
            print("  Usage: memory [--json] [--latency] [--bucket-ms <ms>]")
            return None
        index += 1
    return as_json, show_latency, bucket_ms


def _render(
    summary: dict[str, Any],
    db_path: str,
    latency: list[dict[str, Any]] | None,
) -> None:
    """Print the human-readable roll-up."""
    print(f"  Memory attribution ({os.path.basename(db_path)})")
    print(f"    experiences: {summary['experiences']}")
    print(f"    attributed: {summary['attributed']}")
    print(f"    unattributed: {summary['unattributed']}")
    print(f"    llm_decisions: {summary['llm_decisions']}")

    by_llm = summary.get("by_llm") or []
    if not by_llm:
        print("  No attributed decisions recorded yet.")
    else:
        rates = {r["decision_llm"]: r for r in summary.get("success_by_model") or []}
        print("  By deciding model")
        for row in by_llm:
            model = row["decision_llm"]
            avg = row.get("avg_outcome")
            rate = rates.get(model, {}).get("success_rate")
            avg_txt = f"{avg:.2f}" if isinstance(avg, (int, float)) else "n/a"
            rate_txt = f"{rate * 100:.1f}%" if isinstance(rate, (int, float)) else "n/a"
            print(
                f"    {model}: {row['experiences']} experience(s), "
                f"avg outcome {avg_txt}, success {rate_txt}"
            )

    if latency is not None:
        print("  Decision latency (ms)")
        if not latency:
            print("    (no latencies logged)")
        else:
            width = _bucket_width(latency)
            for bucket in latency:
                low = bucket["bucket_ms"]
                print(f"    {low:.0f}-{low + width - 1:.0f}: {bucket['count']}")


def _bucket_width(histogram: list[dict[str, Any]]) -> float:
    """Infer the bucket width from consecutive lower edges (default 100)."""
    if len(histogram) >= 2:
        return float(histogram[1]["bucket_ms"]) - float(histogram[0]["bucket_ms"])
    return _DEFAULT_BUCKET_MS


class MemoryCommand(Command):
    """Show which model made which recorded decision, and how it fared."""

    @property
    def name(self) -> str:
        return "memory"

    @property
    def help_text(self) -> str:
        return ("memory [--json] [--latency] [--bucket-ms <ms>] - LLM decision "
                "attribution stats from agent_memory.db")

    async def execute(self, args: list[str], agent: "Agent") -> bool:
        parsed = _parse_args(args)
        if parsed is None:
            return True
        as_json, show_latency, bucket_ms = parsed

        db_path = _memory_db_path()
        exists = os.path.exists(db_path)

        if exists:
            summary = attribution_summary(db_path)
            latency = (
                decision_latency_histogram(db_path, bucket_ms=bucket_ms)
                if show_latency else None
            )
        else:
            # No DB: never touch sqlite (a connect would CREATE the file) and
            # never fabricate a measurement.
            summary = _empty_summary()
            latency = [] if show_latency else None

        if as_json:
            payload: dict[str, Any] = dict(summary)
            if latency is not None:
                payload["latency_histogram"] = latency
            print(json.dumps(payload, indent=2, sort_keys=True))
            return True

        if not exists:
            print(f"  No memory database at {db_path} — nothing recorded yet.")
            return True

        _render(summary, db_path, latency)
        return True
