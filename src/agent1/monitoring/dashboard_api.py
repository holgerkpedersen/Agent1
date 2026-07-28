"""Dashboard API exposing monitoring metrics and alerts over HTTP."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from ..core import AlertEvent, AlertRule
from .metrics_collector import MetricsCollector


class DashboardAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler serving monitoring dashboard endpoints."""

    _collector: Optional[MetricsCollector] = None
    _alert_rules: List[AlertRule] = []
    _evaluate_alerts: Optional[Callable[[List[AlertRule]], List[AlertEvent]]] = None

    def do_GET(self) -> None:
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        params = parse_qs(parsed_path.query)
        if path == "/api/snapshot":
            self._send_json(self._snapshot())
        elif path == "/api/counters":
            self._send_json(self._counters())
        elif path == "/api/gauges":
            self._send_json(self._gauges())
        elif path == "/api/histograms":
            self._send_json(self._histogram(params))
        elif path == "/api/alerts":
            self._send_json(self._alerts())
        else:
            self.send_error(404, "Endpoint not found")

    def _send_json(self, data: Dict[str, Any]) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _snapshot(self) -> Dict[str, Any]:
        if self._collector is None:
            return {"error": "metrics collector unavailable"}
        snapshot = dict(self._collector.snapshot())
        typed_snapshot: Dict[str, Any] = snapshot
        typed_snapshot["status"] = "ok"
        return typed_snapshot

    def _extract_metrics(self, snapshot: Dict[str, Any], key: str) -> Dict[str, float]:
        raw = snapshot.get(key, {})
        result: Dict[str, float] = {}
        if isinstance(raw, dict):
            for name, value in raw.items():
                if isinstance(value, (int, float)):
                    result[name] = float(value)
        return result

    def _counters(self) -> Dict[str, Any]:
        if self._collector is None:
            return {"error": "metrics collector unavailable"}
        counters = self._extract_metrics(self._collector.snapshot(), "counters")
        response: Dict[str, Any] = {"counters": counters}
        return response

    def _gauges(self) -> Dict[str, Any]:
        if self._collector is None:
            return {"error": "metrics collector unavailable"}
        gauges = self._extract_metrics(self._collector.snapshot(), "gauges")
        response: Dict[str, Any] = {"gauges": gauges}
        return response

    def _histogram(self, params: Dict[str, List[str]]) -> Dict[str, Any]:
        if self._collector is None:
            return {"error": "metrics collector unavailable"}
        names = params.get("name", [])
        metric_name = names[0] if names else ""
        summary = dict(self._collector.histogram_summary(metric_name))
        response: Dict[str, Any] = {
            "histogram": summary,
            "metric_name": metric_name,
        }
        return response

    def _serialize_rule(self, rule: AlertRule) -> Dict[str, Any]:
        response: Dict[str, Any] = {
            "name": rule.name,
            "metric_name": rule.metric_name,
            "threshold": rule.threshold,
            "comparison_operator": rule.comparison_operator,
            "severity": rule.severity,
            "cooldown_seconds": rule.cooldown_seconds,
        }
        return response

    def _serialize_event(self, event: AlertEvent) -> Dict[str, Any]:
        response: Dict[str, Any] = {
            "rule_name": event.rule_name,
            "triggered_at": event.triggered_at,
            "current_value": event.current_value,
            "threshold": event.threshold,
            "severity": event.severity,
            "message": event.message,
        }
        return response

    def _alerts(self) -> Dict[str, Any]:
        rules = [self._serialize_rule(r) for r in self._alert_rules]
        if self._evaluate_alerts is None:
            return {"error": "alert evaluator unavailable", "rules": rules}
        events = list(self._evaluate_alerts(list(self._alert_rules)))
        response: Dict[str, Any] = {
            "alerts": [self._serialize_event(e) for e in events],
            "count": len(events),
            "rules": rules,
        }
        return response


class DashboardAPIServer:
    """Serves the monitoring dashboard API over HTTP."""

    def __init__(self, collector: MetricsCollector, port: int = 8080) -> None:
        self._collector: MetricsCollector = collector
        self._port: int = port
        self._server: Optional[ThreadingHTTPServer] = None

    def configure(
        self,
        alert_rules: List[AlertRule],
        evaluate_alerts: Optional[Callable[[List[AlertRule]], List[AlertEvent]]] = None,
    ) -> None:
        DashboardAPIHandler._collector = self._collector
        DashboardAPIHandler._alert_rules = list(alert_rules)
        DashboardAPIHandler._evaluate_alerts = evaluate_alerts

    def start(
        self,
        alert_rules: Optional[List[AlertRule]] = None,
        evaluate_alerts: Optional[Callable[[List[AlertRule]], List[AlertEvent]]] = None,
    ) -> ThreadingHTTPServer:
        self.configure(alert_rules or [], evaluate_alerts)
        server = ThreadingHTTPServer(("localhost", self._port), DashboardAPIHandler)
        self._server = server
        return server

    def run(
        self,
        alert_rules: Optional[List[AlertRule]] = None,
        evaluate_alerts: Optional[Callable[[List[AlertRule]], List[AlertEvent]]] = None,
    ) -> None:
        server = self.start(alert_rules or [], evaluate_alerts)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop(server)

    def stop(self, server: ThreadingHTTPServer) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


__all__: List[str] = ["DashboardAPIHandler", "DashboardAPIServer"]