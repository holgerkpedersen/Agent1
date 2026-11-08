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

# Shared metrics plumbing: canonical implementation lives in the CORE package
# at ``agent_core.monitoring.metrics_bridge`` (core -> monitoring, never
# core -> UI).  This module re-exports it so the dashboard entry points and
# every existing import / monkeypatch of ``agent_dashboard.<name>`` keep
# working against the SAME process-wide collector singleton.
from agent_core.monitoring.metrics_bridge import (  # noqa: F401
    _emit_command_metrics,
    _emit_tool_metrics,
    get_metrics_collector,
    record_command_metrics,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from http.server import ThreadingHTTPServer

    from agent_core.monitoring import MetricsCollector
    from agent_core.monitoring.types import AlertRule


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
        AlertRule(
            # Phase 3 t2: the plan-finish evaluator records this gauge ONLY
            # when a finished plan's grade is under the config threshold
            # (AGENT_PLAN_GRADE_ALERT_THRESHOLD, default 0.5), so the rule
            # fires on bad plans and stays silent on good ones.
            name="low_plan_grade",
            metric_name="last.plans.efficiency.score",
            threshold=0.5,
            comparison_operator="less_than",
            severity="warning",
            cooldown_seconds=120,
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


def _dashboard_port_is_explicit() -> bool:
    """True when the user passed ``--port`` (disable automatic fallback)."""
    argv = sys.argv[1:]
    return any(a == "--port" or a.startswith("--port=") for a in argv)


def start_dashboard_thread(port: int = 8081) -> Optional["ThreadingHTTPServer"]:
    """Serve the TTTHEME dashboard on a daemon thread from this process.

    When the port came from the default (no explicit ``--port``) and 8081 is
    already taken by another Agent1 instance, try the next ports so two
    side-by-side sessions can each run a dashboard.
    """
    candidates = (
        [port]
        if _dashboard_port_is_explicit()
        else list(range(port, port + 20))
    )
    last_error: OSError | None = None
    httpd: Optional["ThreadingHTTPServer"] = None
    actual_port = port
    for candidate in candidates:
        server_holder, alert_system = _build_dashboard(
            get_metrics_collector(), candidate
        )
        try:
            httpd = cast(
                "ThreadingHTTPServer",
                server_holder.start(
                    alert_rules=alert_system.list_rules(),
                    evaluate_alerts=alert_system.evaluate,
                    refresh=server_holder.get_refresh(),
                ),
            )
        except OSError as exc:
            last_error = exc
            continue
        actual_port = candidate
        break
    if httpd is None:
        if last_error is not None:
            raise last_error
        return None

    def _serve() -> None:
        try:
            httpd.serve_forever()
        except Exception:
            logger.warning("Silenced exception in start_dashboard_thread")

    threading.Thread(target=_serve, name="agent1-dashboard", daemon=True).start()
    print(f"  Dashboard: http://localhost:{actual_port}  (Ctrl+C to stop)")
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


# The collector singleton and the metric-emission functions now live in
# ``agent_core.monitoring.metrics_bridge`` (core -> monitoring, never core ->
# UI).  Tests reset state by monkeypatching the SINGLE canonical binding:
#   monkeypatch.setattr(agent_core.monitoring.metrics_bridge,
#                       "_shared_metrics_collector", None)
# which every consumer (REPL path, tool loop, dashboard) resolves through.
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


def main(argv: "Optional[list[str]]" = None) -> None:
    """Standalone dashboard CLI: ``python agent_dashboard.py [--port N]``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    for arg in argv:
        if arg in ("-h", "--help"):
            print("usage: agent_dashboard.py [--port N]  (default 8081)")
            print("Serve the TTTHEME metrics dashboard on localhost.")
            return
    run_dashboard_server()


if __name__ == "__main__":
    main()
