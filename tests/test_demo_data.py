"""Regression tests: dashboard data feed (demo_data command + alert wiring).

User story: "I have no data on http://127.0.0.1:8080/index.html and don't
know how to get any." Fixes under test:

1. ``demo_data`` REPL command feeds the SHARED metrics collector through
   ``Agent.record_demo_activity()`` — same metric names the real REPL path
   (``agent.record_command_metrics``) writes, so every dashboard view gets
   data: stat card, Command Summary, Gauges, Histogram, Execution Log.
2. The dashboard servers (``--serve`` and ``--dashboard``) now wire an
   AlertSystem with default rules, so /api/alerts returns rules instead of
   ``{"error": "alert evaluator unavailable"}``.
"""
from __future__ import annotations

import json
import sys
import threading
import typing
import urllib.request

import pytest

import agent


@pytest.fixture(autouse=True)
def _reset_shared_collector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test gets a pristine process-wide collector."""
    monkeypatch.setattr(agent, "_shared_metrics_collector", None)


def _make_agent() -> "agent.Agent":
    return agent.Agent(workspace=".")


def test_record_demo_activity_writes_repl_identical_metrics() -> None:
    bot = _make_agent()
    result = bot.record_demo_activity(activity="analyze", latency_ms=250)

    assert result["events"] == 3
    collector = bot.get_metrics_collector()
    # Same names as record_command_metrics() -> UI filters match.
    assert collector.get_counter_value("command.analyze.count") == 1.0
    assert collector.get_gauge_value("last.command.seconds") == pytest.approx(0.25)
    samples = collector.snapshot()["histogram_samples"]["command.elapsed.seconds"]
    assert samples == [pytest.approx(0.25)]
    # And it lands in the log store the /api/log endpoint serves.
    assert len(collector.get_metrics()) == 3


def test_demo_data_command_feeds_dashboard_end_to_end(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The exact user-facing flow: run the command, then read the API."""
    from agent_core.commands.demo_data_cmd import DemoDataCommand
    from agent_core.monitoring import DashboardAPIServer

    bot = _make_agent()
    cmd = DemoDataCommand()

    # Deterministic asyncio execution of the async command.
    asyncio = __import__("asyncio")
    asyncio.run(cmd.execute(["--count", "6"], bot))

    out = capsys.readouterr().out
    assert "[demo]" in out

    holder = DashboardAPIServer(bot.get_metrics_collector(), port=0)
    httpd = holder.start()
    serve = threading.Thread(target=httpd.serve_forever, daemon=True)
    serve.start()
    try:
        port = httpd.server_address[1]
        base = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(base + "/api/snapshot", timeout=10) as r:
            snap = json.loads(r.read().decode("utf-8"))
        # 6 events over the 5-activity mix: read/search/analyze/fix/read
        assert sum(snap["counters"].values()) == 6
        assert snap["gauges"]["last.command.seconds"] > 0
        assert snap["histogram_samples"]["command.elapsed.seconds"]

        with urllib.request.urlopen(base + "/api/log", timeout=10) as r:
            log = json.loads(r.read().decode("utf-8"))
        assert len(log["records"]) == 18  # 3 metric events per activity
    finally:
        shutdown = threading.Thread(target=httpd.shutdown, daemon=True)
        shutdown.start()
        shutdown.join(timeout=5)
        httpd.server_close()
        typing.cast(typing.Any, serve).join(timeout=5)


def test_demo_data_clear_resets_collector(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_core.commands.demo_data_cmd import DemoDataCommand

    bot = _make_agent()
    bot.record_demo_activity(activity="read")
    assert bot.get_metrics_collector().get_metrics()

    asyncio = __import__("asyncio")
    asyncio.run(DemoDataCommand().execute(["--clear"], bot))
    assert not bot.get_metrics_collector().get_metrics()
    assert "cleared" in capsys.readouterr().out


def test_demo_data_bad_args_fail_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    from agent_core.commands.demo_data_cmd import DemoDataCommand

    bot = _make_agent()
    asyncio = __import__("asyncio")
    asyncio.run(DemoDataCommand().execute(["--count", "zero"], bot))
    assert "Error:" in capsys.readouterr().out


def test_default_alert_rules_cover_slow_and_volume() -> None:
    rules = agent._default_alert_rules()
    names = {r.name for r in rules}
    assert {"slow_command", "command_volume_high", "fix_runs_elevated"} <= names
    slow = next(r for r in rules if r.name == "slow_command")
    assert slow.metric_name == "last.command.seconds"
    assert slow.comparison_operator == "greater_than"


def test_serve_mode_wires_alert_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """/api/alerts must return rules + evaluated events, not an error."""
    from agent_core.monitoring import DashboardAPIServer

    captured: dict[str, object] = {}

    def fake_run(self: DashboardAPIServer, alert_rules=None, evaluate_alerts=None):
        captured["rules"] = list(alert_rules or [])
        captured["evaluate"] = evaluate_alerts
        raise KeyboardInterrupt()  # exit run() immediately

    monkeypatch.setattr(agent, "_shared_metrics_collector", None)
    monkeypatch.setattr(DashboardAPIServer, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["agent.py", "--serve"])
    # run_dashboard_server() deliberately SWALLOWS KeyboardInterrupt (its
    # Ctrl+C contract), so nothing propagates — the assertions below prove
    # fake_run executed and captured the wiring.
    agent.run_dashboard_server()

    rules = typing.cast(list, captured["rules"])
    evaluate = typing.cast(object, captured["evaluate"])
    assert len(rules) >= 3
    # Evaluator is AlertSystem.evaluate; feed it the rules and expect a list.
    events = evaluate(rules)  # type: ignore[operator]
    assert isinstance(events, list)


def test_serve_mode_api_alerts_endpoint_returns_rules() -> None:
    """Full HTTP check of the alerts surface after the wiring fix."""
    from agent_core.monitoring import AlertSystem, DashboardAPIServer

    bot = _make_agent()
    # 2.5s > the slow_command threshold -> should trigger one alert event.
    bot.record_demo_activity(activity="analyze", latency_ms=2500.0)

    alert_system = AlertSystem(bot.get_metrics_collector())
    for rule in agent._default_alert_rules():
        alert_system.add_rule(rule)

    holder = DashboardAPIServer(bot.get_metrics_collector(), port=0)
    httpd = holder.start(
        alert_rules=alert_system.list_rules(),
        evaluate_alerts=alert_system.evaluate,
    )
    serve = threading.Thread(target=httpd.serve_forever, daemon=True)
    serve.start()
    try:
        port = httpd.server_address[1]
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/alerts", timeout=10
        ) as r:
            payload = json.loads(r.read().decode("utf-8"))
        assert "error" not in payload, payload
        assert payload["count"] >= 1
        rule_names = {rule["name"] for rule in payload["rules"]}
        assert "slow_command" in rule_names
        assert payload["alerts"][0]["rule_name"] == "slow_command"
    finally:
        shutdown = threading.Thread(target=httpd.shutdown, daemon=True)
        shutdown.start()
        shutdown.join(timeout=5)
        httpd.server_close()
        typing.cast(typing.Any, serve).join(timeout=5)
