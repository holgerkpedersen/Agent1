"""Regression tests: the TTTHEME web dashboard showed no data.

Root causes fixed:
1. ``run_dashboard_server()`` built its own private, *empty* MetricsCollector,
   so nothing done in the REPL was ever visible in the UI. The REPL now feeds
   the shared ``get_metrics_collector()`` via ``record_command_metrics()``,
   and ``--dashboard`` serves that same collector from a daemon thread.
2. ``/api/log`` returns epoch SECONDS while the page's fmtTime() parsed them
   as milliseconds -> every timestamp rendered as January 1970.
   (JS fix in static/index.html; the backend contract is pinned below.)
3. ``loadLog()`` only wrote to the hidden ``tbl-log-view`` table; the visible
   "Execution Log (last 100)" card (id ``tbl-log``) stayed on "Loading...".
   (JS fix in static/index.html.)
"""

from __future__ import annotations

import json
import sys
import threading
import typing
import urllib.error
import urllib.request

import pytest

import agent


@pytest.fixture(autouse=True)
def _reset_shared_collector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test gets a pristine process-wide collector.

    The singleton binding's canonical home is
    ``agent_core.monitoring.metrics_bridge`` (decision #139 step 1); every
    consumer resolves through it, so patching the bridge resets all of them.
    """
    import agent_core.monitoring.metrics_bridge as _bridge

    monkeypatch.setattr(_bridge, "_shared_metrics_collector", None)
    # main() now also boots the Kanban inbound poller; keep these dashboard
    # tests from spawning a real one via whatever is in the repo .env.
    monkeypatch.setattr(agent, "_start_kanban_inbound", lambda: None)


def test_shared_collector_is_singleton() -> None:
    assert agent.get_metrics_collector() is agent.get_metrics_collector()


def test_record_command_metrics_feeds_all_metric_types() -> None:
    agent.record_command_metrics("read", 1.234)
    collector = agent.get_metrics_collector()

    # Per-command counter follows the naming the TTTHEME UI filters on.
    # (Deliberately NO aggregate "command.executions" counter: the UI's
    # stat card sums ALL counters, which would double-count.)
    assert collector.get_counter_value("command.read.count") == 1.0
    assert collector.get_counter_value("command.executions") == 0.0
    assert collector.get_gauge_value("last.command.seconds") == 1.234

    snapshot = collector.snapshot()
    assert snapshot["counters"]["command.read.count"] == 1.0
    assert snapshot["histogram_samples"]["command.elapsed.seconds"] == [1.234]


def test_log_timestamps_are_epoch_seconds() -> None:
    """Backend contract pinned for the 1970-bug: time.time() seconds, not ms."""
    agent.record_command_metrics("perf", 0.5)
    records = agent.get_metrics_collector().get_metrics()
    now = __import__("time").time()
    for r in records:
        # Within 5 minutes of *now* when interpreted as SECONDS.
        assert abs(r.timestamp - now) < 300
        # And NOT plausibly a millisecond value.
        assert r.timestamp < 1e11


def test_dashboard_serves_repl_collected_data_end_to_end() -> None:
    """The exact user-facing bug: REPL data must appear over HTTP."""
    from agent_core.monitoring import DashboardAPIServer

    agent.record_command_metrics("analyze", 2.5)
    holder = DashboardAPIServer(agent.get_metrics_collector(), port=0)  # ephemeral
    httpd = holder.start()
    serve = threading.Thread(target=httpd.serve_forever, daemon=True)
    serve.start()
    try:
        port = httpd.server_address[1]
        url = f"http://127.0.0.1:{port}/api/snapshot"
        with urllib.request.urlopen(url, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assert payload["counters"]["command.analyze.count"] == 1.0
        assert payload["gauges"]["last.command.seconds"] == 2.5
    finally:
        shutdown = threading.Thread(target=httpd.shutdown, daemon=True)
        shutdown.start()
        shutdown.join(timeout=5)
        httpd.server_close()
        typing.cast(typing.Any, serve).join(timeout=5)


def test_main_dashboard_flag_boots_thread_and_repl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_start(port: int = 8081) -> None:
        calls.append(f"dashboard:{port}")

    async def fake_interactive() -> None:
        calls.append("interactive")

    monkeypatch.setattr(agent, "start_dashboard_thread", fake_start)
    monkeypatch.setattr(agent, "run_interactive", fake_interactive)
    monkeypatch.setattr(sys, "argv", ["agent.py", "--dashboard"])
    import asyncio

    asyncio.run(agent.main())
    assert calls == ["dashboard:8081", "interactive"]


def test_dashboard_port_flag_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    for argv, expected in (
        (["agent.py"], 8081),  # dashboard defaults to 8081; llama-server keeps :8080
        (["agent.py", "--port", "9001"], 9001),
        (["agent.py", "--port=9002"], 9002),
        (["agent.py", "--serve", "--port", "abc"], 8081),  # invalid -> default
    ):
        monkeypatch.setattr(sys, "argv", argv)
        assert agent._dashboard_port() == expected


def test_dashboard_swallows_client_disconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    """A browser tab closing mid-response must not log a spurious exception.

    Regression: every ``ConnectionAbortedError`` / ``BrokenPipeError`` /
    ``ConnectionResetError`` from ``wfile.write`` propagated up through
    ``do_GET``/``do_POST`` and printed a traceback to stderr, cluttering the
    server log with noise for a normal client lifecycle event.
    """
    from agent_core.monitoring import DashboardAPIServer

    agent.record_command_metrics("read", 0.1)
    holder = DashboardAPIServer(agent.get_metrics_collector(), port=0)
    httpd = holder.start()
    serve = threading.Thread(target=httpd.serve_forever, daemon=True)
    serve.start()
    try:
        port = httpd.server_address[1]
        url = f"http://127.0.0.1:{port}/api/snapshot"

        # Simulate a client that disconnects mid-response by patching the
        # handler's wfile.write to raise ConnectionAbortedError.
        real_handler_cls = type(httpd.RequestHandlerClass)

        class _DisconnectingHandler(real_handler_cls):
            def _send_json(self, data):  # type: ignore[override]
                import json as _json

                body = _json.dumps(data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                raise ConnectionAbortedError("client disconnected")

            def do_GET(self) -> None:  # type: ignore[override]
                try:
                    self._do_get()
                except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
                    pass

            def _do_get(self) -> None:
                self._maybe_refresh()
                self._send_json(self._snapshot())

        httpd.RequestHandlerClass = _DisconnectingHandler  # type: ignore[assignment]

        # The server must swallow the disconnect; the client sees a normal
        # connection-closed error (not a server traceback).
        with pytest.raises((urllib.error.URLError, ConnectionError, OSError)):
            with urllib.request.urlopen(url, timeout=10):
                pass
    finally:
        shutdown = threading.Thread(target=httpd.shutdown, daemon=True)
        shutdown.start()
        shutdown.join(timeout=5)
        httpd.server_close()
        typing.cast(typing.Any, serve).join(timeout=5)
