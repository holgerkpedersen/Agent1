#!/usr/bin/env python3
"""Dashboard and metrics plumbing for Agent1.

Contains the process-wide MetricsCollector singleton, metric emission helpers
(shared by the REPL command path and the NLP tool loop), default alert rules,
and both dashboard entry points (--serve standalone server + --dashboard
in-process daemon thread).

This module was extracted from agent.py to keep the entrypoint focused on the
REPL / chat loop.  ``agent`` re-exports every public symbol here so existing
imports (``import agent; agent.record_command_metrics(...)``) continue to work
unchanged.
"""
from __future__ import annotations

import logging
import sys
import threading
from typing import TYPE_CHECKING, Any, Optional, cast

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from http.server import ThreadingHTTPServer

    from agent_core.monitoring import MetricsCollector
    from agent_core.monitoring.types import AlertRule


# ---------------------------------------------------------------------------
# Shared metrics plumbing
#
# The TTTHEME dashboard (upstream @LebToki) reads everything from a
# MetricsCollector instance, but run_dashboard_server() used to build its own
# EMPTY collector — so the web UI stayed blank even after a full REPL session.
# This module-level bridge lets the REPL loop feed the SAME collector the
# dashboard serves, and optionally boots the dashboard in-process.
# ---------------------------------------------------------------------------
_shared_metrics_collector: Optional["MetricsCollector"] = None


def get_metrics_collector() -> "MetricsCollector":
    """Return the process-wide collector shared by REPL and dashboard."""
    global _shared_metrics_collector
    if _shared_metrics_collector is None:
        from agent_core.monitoring import MetricsCollector as _MC

        _shared_metrics_collector = _MC()
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
    from agent_core.monitoring.metrics_file import append_event as _append_event

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
    from agent_core.monitoring.metrics_file import append_event as _append_event

    _append_event("counter", f"tool.{tool_name}.count", 1.0)
    _append_event("histogram", "tool.elapsed.seconds", elapsed_s)
    if ok:
        _append_event("gauge", "last.tool.seconds", elapsed_s)


def _default_alert_rules() -> "list[AlertRule]":
    """Dashboard alert rules evaluated live against the shared collector."""
    from agent_core.monitoring.types import AlertRule

    return [
        AlertRule(
            name="slow_command",
            metric_name="last.command.seconds",
            threshold=2.0,
            comparison_operator="greater_than",
            severity="warning",
            cooldown_seconds=30,
        ),
        AlertRule(
            name="command_volume_high",
            metric_name="command.analyze.count",
            threshold=50,
            comparison_operator="greater_than",
            severity="info",
            cooldown_seconds=300,
        ),
        AlertRule(
            name="fix_runs_elevated",
            metric_name="command.fix.count",
            threshold=20,
            comparison_operator="greater_than",
            severity="critical",
            cooldown_seconds=300,
        ),
    ]


def _build_dashboard(collector: "MetricsCollector", port: int) -> tuple[Any, Any]:
    """Wire collector + default alert rules into a DashboardAPIServer.

    Shared by :func:`start_dashboard_thread` (REPL + dashboard in-process)
    and :func:`run_dashboard_server` (`--serve`) so both surfaces always get
    identical rules and evaluator wiring.
    """
    from agent_core.monitoring import AlertSystem, DashboardAPIServer
    from agent_core.monitoring.metrics_file import make_event_tailer

    alert_system = AlertSystem(collector)
    for rule in _default_alert_rules():
        alert_system.add_rule(rule)
    server_holder = DashboardAPIServer(collector, port=port)
    # Tail the shared event file on every request so cross-process activity
    # (REPL sessions running beside a --serve dashboard) shows up.  Own-pid
    # events are skipped inside the tailer, so combined mode can't double-count.
    server_holder.set_refresh(make_event_tailer())
    return server_holder, alert_system


def start_dashboard_thread(port: int = 8081) -> Optional["ThreadingHTTPServer"]:
    """Serve the TTTHEME dashboard on a daemon thread from this process."""
    server_holder, alert_system = _build_dashboard(get_metrics_collector(), port)
    httpd = cast(
        "ThreadingHTTPServer",
        server_holder.start(
            alert_rules=alert_system.list_rules(),
            evaluate_alerts=alert_system.evaluate,
            refresh=server_holder.get_refresh(),
        ),
    )

    def _serve() -> None:
        try:
            httpd.serve_forever()
        except Exception:
            logger.warning(
                "Silenced exception in start_dashboard_thread"
            )

    threading.Thread(target=_serve, name="agent1-dashboard", daemon=True).start()
    print(f"  Dashboard: http://localhost:{port}  (Ctrl+C to stop)")
    return httpd


def _dashboard_port() -> int:
    """Resolve the dashboard port: --port N / --port=N, else 8081."""
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--port" and i + 1 < len(argv):
            try:
                return int(argv[i + 1])
            except ValueError:
                break
        if arg.startswith("--port="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                break
    return 8081


def run_dashboard_server() -> None:
    """Launch the TTTHEME web dashboard on localhost:8081.

    Port 8080 is reserved for the llama-server LLM backend (see AGENT_LLAMA_URL),
    so the dashboard defaults to 8081 and can be overridden with --port N."""
    port = _dashboard_port()
    # Reuse the process-wide collector so anything recorded before/while
    # serving (REPL commands, chat turns) is visible in the UI.
    server_holder, alert_system = _build_dashboard(get_metrics_collector(), port)
    print(f"Agent1 dashboard: http://localhost:{port}  (Ctrl+C to stop)")
    try:
        server_holder.run(
            alert_rules=alert_system.list_rules(),
            evaluate_alerts=alert_system.evaluate,
            refresh=server_holder.get_refresh(),
        )
    except KeyboardInterrupt:
        logger.warning("Silenced exception in agent_dashboard.run_dashboard_server")


# Re-export _shared_metrics_collector so tests that monkeypatch it via the
# ``agent`` module's re-export get a consistent object identity.  The functions
# above reference this module-level binding directly, so patching here is what
# actually resets state between test runs.
__all__ = [
    "get_metrics_collector",
    "record_command_metrics",
    "_emit_command_metrics",
    "_emit_tool_metrics",
    "_build_dashboard",
    "start_dashboard_thread",
    "run_dashboard_server",
    "_dashboard_port",
    "_default_alert_rules",
]
