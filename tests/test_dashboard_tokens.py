"""Regression tests: Token efficiency panel (Phase 3 of the token ledger).

Three layers of guards:
1. Metric emission - ``record_plan_token_metrics`` writes collector facts
   (counters / histograms / gauges) and mirrors them into the shared event
   file, and NEVER invents values for missing facts.
2. API contract - ``GET /api/tokens`` exposes totals, per-plan mean, last
   values, histogram summaries and the read-only ledger history.
3. Frontend scaffolding - index.html carries the panel markup, the loader
   and the view label so the data is actually rendered.
"""

from __future__ import annotations

import json
import os
import threading
import typing
import urllib.request

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(REPO, "static", "index.html")


# --------------------------------------------------------------------------
# 1. Metric emission
# --------------------------------------------------------------------------

def test_record_plan_token_metrics_writes_facts(tmp_path, monkeypatch) -> None:
    from agent_core.monitoring import metrics_file as mf
    from agent_core.plan_execution.plan_finish_eval import record_plan_token_metrics

    events = tmp_path / "events.jsonl"
    monkeypatch.setattr(mf, "default_events_path", lambda: events)

    from agent_core.monitoring.metrics_collector import MetricsCollector

    collector = MetricsCollector()
    import agent_core.monitoring.metrics_bridge as mb

    monkeypatch.setattr(mb, "get_metrics_collector", lambda: collector, raising=False)
    # The bridge module re-exports the function; patch where it is looked up.
    monkeypatch.setattr(
        "agent_core.plan_execution.plan_finish_eval.get_metrics_collector",
        lambda: collector,
        raising=False,
    )

    # plans.finished.count drives the per-task divisor: seed it like a
    # previous plan finish would have.
    collector.increment_counter("plans.finished.count")

    record_plan_token_metrics(
        "p-1",
        {"prompt_tokens": 400, "completion_tokens": 100, "total_tokens": 500},
        0.9,
    )

    assert collector.get_counter_value("tokens.plan.total") == 500.0
    assert collector.get_gauge_value("last.tokens.plan.total") == 500.0
    assert collector.histogram_summary("tokens.plan.total")["count"] == 1.0
    # divisor = plans.finished.count (1) -> per-task sample equals the total
    assert collector.histogram_summary("tokens.plan.per_task")["mean"] == 500.0
    assert collector.get_gauge_value("last.plans.efficiency.score") == 0.9
    assert collector.histogram_summary("plans.efficiency.score")["count"] == 1.0
    # this plan's own finish bumps plans.finished.count
    assert collector.get_counter_value("plans.finished.count") == 2.0

    lines = [json.loads(x) for x in events.read_text(encoding="utf-8").splitlines()]
    names = {e["name"] for e in lines}
    assert {
        "tokens.plan.total",
        "last.tokens.plan.total",
        "tokens.plan.per_task",
        "plans.efficiency.score",
        "last.plans.efficiency.score",
        "plans.finished.count",
    } <= names


def test_record_plan_token_metrics_missing_facts(tmp_path, monkeypatch) -> None:
    """No tokens -> no token writes; no score -> no score writes."""
    from agent_core.monitoring import metrics_file as mf
    from agent_core.plan_execution.plan_finish_eval import record_plan_token_metrics

    events = tmp_path / "events.jsonl"
    monkeypatch.setattr(mf, "default_events_path", lambda: events)

    from agent_core.monitoring.metrics_collector import MetricsCollector

    collector = MetricsCollector()
    import agent_core.monitoring.metrics_bridge as mb

    monkeypatch.setattr(mb, "get_metrics_collector", lambda: collector, raising=False)

    record_plan_token_metrics("p-2", None, None)

    assert collector.get_counter_value("tokens.plan.total") == 0.0
    assert collector.get_gauge_value("last.tokens.plan.total") is None
    assert collector.histogram_summary("plans.efficiency.score")["count"] == 0.0
    # the finish itself is still counted (facts only, no invention)
    assert collector.get_counter_value("plans.finished.count") == 1.0


# --------------------------------------------------------------------------
# 2. API contract
# --------------------------------------------------------------------------

def _seeded_collector():
    from agent_core.monitoring.metrics_collector import MetricsCollector

    c = MetricsCollector()
    c.increment_counter("plans.finished.count", 2.0)
    c.increment_counter("tokens.plan.total", 3000.0)
    c.record_histogram("tokens.plan.total", 1000.0)
    c.record_histogram("tokens.plan.total", 2000.0)
    c.set_gauge("last.tokens.plan.total", 2000.0)
    c.record_histogram("plans.efficiency.score", 0.8)
    c.set_gauge("last.plans.efficiency.score", 0.8)
    # Per-component run histograms (recorded by record_token_histograms).
    c.record_histogram("tokens.run.prompt", 1000.0)
    c.record_histogram("tokens.run.prompt", 2000.0)
    c.record_histogram("tokens.run.completion", 250.0)
    return c


@pytest.fixture(scope="module")
def base_url(tmp_path_factory) -> typing.Iterator[str]:
    from _pytest.monkeypatch import MonkeyPatch

    from agent_core.monitoring import DashboardAPIServer
    from agent_core.monitoring.dashboard_api import DashboardAPIHandler

    ledger_dir = tmp_path_factory.mktemp("ledger") / "reports" / "history"
    ledger_dir.mkdir(parents=True)
    line = {
        "ts": 1700000000.0,
        "command": "plan_evaluation",
        "outcome": "success",
        "note": "plan p-9: score=0.85 tasks=3/3 tokens=1234",
    }
    (ledger_dir / "executions.jsonl").write_text(
        json.dumps(line) + "\n", encoding="utf-8"
    )

    holder = DashboardAPIServer(_seeded_collector(), port=0)
    # _token_history reads the handler class's _base_dir; point it at the tmp
    # ledger root (reports/history/executions.jsonl lives under it).
    monkey = MonkeyPatch()
    monkey.setattr(DashboardAPIHandler, "_base_dir", str(ledger_dir.parent.parent))
    httpd = holder.start()
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        shutdown = threading.Thread(target=httpd.shutdown, daemon=True)
        shutdown.start()
        shutdown.join(timeout=5)
        httpd.server_close()
        monkey.undo()


def test_api_tokens_totals_and_summaries(base_url: str) -> None:
    with urllib.request.urlopen(base_url + "/api/tokens", timeout=10) as r:
        payload = json.loads(r.read())
    assert payload["plans_finished"] == 2.0
    assert payload["tokens_total"] == 3000.0
    assert payload["tokens_per_plan"] == 1500.0
    assert payload["last_tokens"] == 2000.0
    assert payload["last_score"] == 0.8
    assert payload["total_summary"]["count"] == 2.0
    assert payload["total_summary"]["mean"] == 1500.0
    assert payload["score_summary"]["count"] == 1.0
    # Per-component distributions reach the API (prompt has 2 samples).
    assert payload["prompt_summary"]["count"] == 2.0
    assert payload["prompt_summary"]["mean"] == 1500.0
    assert payload["completion_summary"]["count"] == 1.0


def test_api_tokens_history_reads_ledger(base_url: str) -> None:
    with urllib.request.urlopen(base_url + "/api/tokens", timeout=10) as r:
        payload = json.loads(r.read())
    hist = payload["history"]
    assert len(hist) == 1
    row = hist[0]
    assert row["plan_id"] == "p-9"
    assert row["outcome"] == "success"
    assert row["score"] == 0.85
    assert row["tasks_completed"] == 3
    assert row["tasks_planned"] == 3
    assert row["tokens"] == 1234


def test_api_tokens_history_missing_ledger_is_empty(tmp_path) -> None:
    """Unreadable ledger degrades to an empty history, never raises."""
    from agent_core.monitoring.dashboard_api import DashboardAPIHandler

    class _Collector:
        def snapshot(self):
            return {"counters": {}, "gauges": {}, "histogram_samples": {}}

        def histogram_summary(self, name):
            return {"count": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0}

    handler = DashboardAPIHandler.__new__(DashboardAPIHandler)
    handler._collector = _Collector()
    handler._base_dir = str(tmp_path / "does-not-exist")
    data = handler._tokens()
    assert data["history"] == []
    assert data["plans_finished"] == 0.0
    assert data["tokens_per_plan"] is None


# --------------------------------------------------------------------------
# 2b. Low-grade plan alert (Phase 3 t2)
# --------------------------------------------------------------------------

def test_grade_alert_threshold_default_and_disable() -> None:
    """Config threshold drives the alert; 0.0 disables it entirely."""
    from agent_core.plan_execution.plan_finish_eval import plan_grade_alert_threshold

    assert plan_grade_alert_threshold() == pytest.approx(0.5)


def test_record_plan_grade_alert_only_fires_below_threshold(
    tmp_path, monkeypatch
) -> None:
    """Gauge recorded ONLY for grades under the threshold; good plans silent."""
    from agent_core.monitoring.metrics_collector import MetricsCollector
    import agent_core.monitoring.metrics_bridge as mb
    import agent_core.plan_execution.plan_finish_eval as pfe

    collector = MetricsCollector()
    monkeypatch.setattr(mb, "get_metrics_collector", lambda: collector)
    # Keep the shared event file inside tmp so the test never touches the repo.
    events = tmp_path / "events.jsonl"
    monkeypatch.setattr(
        pfe,
        "plan_grade_alert_threshold",
        lambda: 0.5,
        raising=False,
    )
    import agent_core.monitoring.metrics_file as mf

    monkeypatch.setattr(mf, "default_events_path", lambda: events)

    # Good grade: no gauge, no event line.
    assert pfe.record_plan_grade_alert("p-good", 0.85) is False
    assert collector.get_gauge_value("last.plans.efficiency.score") is None
    assert not events.exists()

    # Low grade: gauge recorded and mirrored to the shared event file.
    assert pfe.record_plan_grade_alert("p-bad", 0.31) is True
    gauge = collector.get_gauge_value("last.plans.efficiency.score")
    assert gauge == pytest.approx(0.31)
    lines = events.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    ev = json.loads(lines[0])
    assert ev["kind"] == "gauge"
    assert ev["name"] == "last.plans.efficiency.score"
    assert ev["value"] == pytest.approx(0.31)

    # Threshold 0.0 disables the alert even for a grade of 0.
    monkeypatch.setattr(pfe, "plan_grade_alert_threshold", lambda: 0.0)
    assert pfe.record_plan_grade_alert("p-off", 0.0) is False


def test_record_plan_grade_alert_degrades_silently(monkeypatch) -> None:
    """No collector, no score, junk score -> silent no-op, never raises."""
    import agent_core.monitoring.metrics_bridge as mb
    import agent_core.plan_execution.plan_finish_eval as pfe

    def _boom() -> None:
        raise RuntimeError("collector unavailable")

    monkeypatch.setattr(mb, "get_metrics_collector", _boom)
    assert pfe.record_plan_grade_alert("p-x", 0.1) is False
    assert pfe.record_plan_grade_alert("p-x", None) is False
    assert pfe.record_plan_grade_alert("p-x", "junk") is False
    assert pfe.record_plan_grade_alert("p-x", True) is False


def test_dashboard_has_low_plan_grade_rule() -> None:
    """The dashboard evaluates a low-grade plan alert against the score gauge."""
    import agent_dashboard

    rules = {r.name: r for r in agent_dashboard._default_alert_rules()}
    rule = rules["low_plan_grade"]
    assert rule.metric_name == "last.plans.efficiency.score"
    assert rule.comparison_operator == "less_than"
    assert rule.threshold == pytest.approx(0.5)
    assert rule.severity in ("warning", "critical")


def test_low_plan_grade_alert_fires_end_to_end() -> None:
    """Record a low grade -> AlertSystem.evaluate fires the dashboard rule."""
    from agent_core.monitoring.alert_system import AlertSystem
    from agent_core.monitoring.metrics_collector import MetricsCollector
    import agent_dashboard

    collector = MetricsCollector()
    collector.set_gauge("last.plans.efficiency.score", 0.2)
    system = AlertSystem(collector)
    for rule in agent_dashboard._default_alert_rules():
        system.add_rule(rule)
    events = system.evaluate()
    fired = [e for e in events if e.rule_name == "low_plan_grade"]
    assert len(fired) == 1
    assert fired[0].current_value == pytest.approx(0.2)


# --------------------------------------------------------------------------
# 3. Frontend scaffolding
# --------------------------------------------------------------------------

def test_index_has_token_panel_scaffolding() -> None:
    html = open(INDEX, encoding="utf-8").read()
    for needle in (
        'data-view-panel="tokens"',
        'id="tok-plans"',
        'id="tok-total"',
        'id="tok-per-plan"',
        'id="tbl-tokens"',
        "'api/tokens'",
        "loadTokens",
        "Token Efficiency",
    ):
        assert needle in html, f"index.html missing token panel marker: {needle}"


def test_index_has_token_trend_chart() -> None:
    """Phase 3 t1: the tokens panel carries a pure-CSS per-plan trend strip.

    Contract: a trend container exists, loadTokens feeds it from d.history
    via renderTokTrend, and the renderer degrades to an explicit empty
    state when there is no history (no invented bars).
    """
    html = open(INDEX, encoding="utf-8").read()
    for needle in (
        'id="tok-trend"',
        ".tok-trend-chart",
        ".tok-trend-inner",
        ".tok-trend-empty",
        "function renderTokTrend",
        "renderTokTrend(hist, fmtTok)",
        "No plan evaluations recorded yet",
    ):
        assert needle in html, f"index.html missing token trend marker: {needle}"
    # The renderer must be defined before loadTokens uses it and must key
    # bar heights off the ledger history (tokens), not invented values.
    assert html.index("function renderTokTrend") < html.index(
        "renderTokTrend(hist, fmtTok)"
    )
    assert "Number(h.tokens)" in html
