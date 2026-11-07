"""Process-wide metrics bridge shared by the REPL, the NLP tool loop and the dashboard.

This is the canonical home of the metric-emission plumbing that used to live
in ``agent_dashboard.py`` (the UI layer).  Core code (``agent_core.agent``,
the tool loop) depends on *this* module — a monitoring concern in the core
package — instead of importing the dashboard, which keeps the dependency
direction core -> monitoring and never core -> UI.

``agent_dashboard`` re-exports every symbol here (single shared singleton),
so ``import agent_dashboard; agent_dashboard.get_metrics_collector()`` and
existing monkeypatching tests keep working unchanged.

Design notes:
- The TTTHEME dashboard (upstream @LebToki) reads everything from a
  MetricsCollector instance; the REPL loop feeds the SAME collector the
  dashboard serves, so the web UI shows live session activity.
- Metric names are the single source of truth the TTTHEME UI filters on
  (see loadCommands() regex in static/index.html): exactly ONE
  ``command.<name>.count`` counter per command (the UI's stat card naively
  sums every counter, so an aggregate counter would double the figure),
  an elapsed-seconds histogram sample, and a ``last.*.seconds`` gauge.
- Every write is mirrored into the shared event file (see
  :mod:`agent_core.monitoring.metrics_file`) so a standalone ``--serve``
  dashboard can show activity recorded by other sessions.
"""

from __future__ import annotations

from typing import Optional

from agent_core.monitoring.metrics_collector import MetricsCollector
from agent_core.monitoring.metrics_file import append_event as _append_event

# ---------------------------------------------------------------------------
# Shared metrics plumbing: the process-wide collector singleton.
# ---------------------------------------------------------------------------
_shared_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Return the process-wide collector shared by REPL and dashboard."""
    global _shared_metrics_collector
    if _shared_metrics_collector is None:
        _shared_metrics_collector = MetricsCollector()
    assert _shared_metrics_collector is not None
    return _shared_metrics_collector


def _emit_command_metrics(command: str, elapsed_s: float) -> None:
    """Write the three dashboard metrics for one command execution.

    Single source of truth for the metric NAMES the TTTHEME UI filters on
    (see loadCommands() regex in static/index.html): exactly ONE
    ``command.<name>.count`` counter (the UI's stat card naively sums every
    counter, so an aggregate counter would double the figure), an
    elapsed-seconds histogram sample, and the ``last.command.seconds`` gauge.
    """
    collector = get_metrics_collector()
    collector.increment_counter(f"command.{command}.count")
    collector.record_histogram("command.elapsed.seconds", elapsed_s)
    collector.set_gauge("last.command.seconds", elapsed_s)
    # Mirror into the shared event file so a standalone --serve dashboard can
    # show activity recorded by other sessions (metrics_file replay).
    _append_event("counter", f"command.{command}.count", 1.0)
    _append_event("histogram", "command.elapsed.seconds", elapsed_s)
    _append_event("gauge", "last.command.seconds", elapsed_s)


def record_command_metrics(command: str, elapsed_s: float) -> None:
    """Mirror one real REPL command execution into the shared collector."""
    _emit_command_metrics(command, elapsed_s)


def _emit_tool_metrics(tool_name: str, elapsed_s: float, ok: bool) -> None:
    """Write the dashboard metrics for one TOOL execution (LLM tool loop).

    Companion to :func:`_emit_command_metrics`: same shape, ``tool.`` prefix
    instead of ``command.`` so the TTTHEME command view can show both real
    REPL commands and model-driven tool calls (git, read_file, ...) — its
    loadCommands() regex already matches ``tool``.  One counter per tool name
    keeps the stat-card sum honest (no aggregate counters).
    """
    collector = get_metrics_collector()
    collector.increment_counter(f"tool.{tool_name}.count")
    collector.record_histogram("tool.elapsed.seconds", elapsed_s)
    if ok:
        collector.set_gauge("last.tool.seconds", elapsed_s)
    # Mirror into the shared event file (see _emit_command_metrics).
    _append_event("counter", f"tool.{tool_name}.count", 1.0)
    _append_event("histogram", "tool.elapsed.seconds", elapsed_s)
    if ok:
        _append_event("gauge", "last.tool.seconds", elapsed_s)


__all__ = [
    "MetricsCollector",
    "get_metrics_collector",
    "_emit_command_metrics",
    "_emit_tool_metrics",
    "record_command_metrics",
]
