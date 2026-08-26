"""Dashboard API exposing monitoring metrics and alerts over HTTP."""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from .metrics_collector import MetricsCollector
from .types import AlertEvent, AlertRule


class DashboardAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler serving monitoring dashboard endpoints."""

    _collector: Optional[MetricsCollector] = None
    _alert_rules: List[AlertRule] = []
    _evaluate_alerts: Optional[Callable[[List[AlertRule]], List[AlertEvent]]] = None
    #: Optional pre-request hook (e.g. tailing a shared metrics event file so
    #: cross-process activity shows up). Called before every API dispatch.
    _refresh: Optional[Callable[[MetricsCollector], None]] = None
    _base_dir: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def do_GET(self) -> None:
        self._maybe_refresh()
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        params = parse_qs(parsed_path.query)
        if path in ("/", "/index.html"):
            self._send_html_page()
        elif path == "/api/snapshot":
            self._send_json(self._snapshot())
        elif path == "/api/counters":
            self._send_json(self._counters())
        elif path == "/api/gauges":
            self._send_json(self._gauges())
        elif path == "/api/histograms":
            self._send_json(self._histogram(params))
        elif path == "/api/alerts":
            self._send_json(self._alerts())
        elif path == "/api/log":
            self._send_json(self._log())
        elif path.startswith("/static/"):
            self._send_static(path)
        else:
            self.send_error(404, "Not found")

    def _send_json(self, data: Dict[str, Any]) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, path: str) -> None:
        """Serve files from the repository static/ directory."""
        rel = path.lstrip("/")
        if ".." in rel:
            self.send_error(403, "Forbidden")
            return
        full = os.path.join(self._base_dir, rel)
        if not os.path.isfile(full):
            self.send_error(404, "Not found")
            return
        with open(full, "rb") as f:
            body = f.read()
        ctype = self._content_type(full)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html_page(self) -> None:
        """Assemble partials into the TTTHEME dashboard page and serve it."""
        index_path = os.path.join(self._base_dir, "static", "index.html")
        if not os.path.isfile(index_path):
            self.send_error(404, "index.html not found")
            return
        with open(index_path, "r", encoding="utf-8") as f:
            page = f.read()
        partials = {
            "@@HEAD@@": self._read_partial("_head.html"),
            "@@SIDEBAR@@": self._read_partial("_sidebar.html"),
            "@@HEADER@@": self._read_partial("_header.html"),
            "@@FOOTER@@": self._read_partial("_footer.html"),
        }
        for marker, content in partials.items():
            page = page.replace(marker, content)
        body = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _read_partial(name: str) -> str:
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(base, "partials", name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except OSError:
            return ""

    @staticmethod
    def _content_type(path: str) -> str:
        mapping = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".json": "application/json",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
            ".ico": "image/x-icon",
            ".woff": "font/woff",
            ".woff2": "font/woff2",
            ".ttf": "font/ttf",
            ".eot": "application/vnd.ms-fontobject",
        }
        ext = os.path.splitext(path)[1].lower()
        return mapping.get(ext, "application/octet-stream")

    def _maybe_refresh(self) -> None:
        """Run the optional refresh hook before serving an API request."""
        # Read via the CLASS: a plain-function class attribute accessed through
        # the instance would bind as a method (descriptor protocol) and then be
        # called with a stray ``self`` — silently breaking the hook.
        fn = type(self)._refresh
        if fn is None or self._collector is None:
            return
        try:
            fn(self._collector)
        except Exception:
            pass  # metrics must never break request handling

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
            #: Raw sample lists per metric (most recent last) so the UI can
            #: bin them into a distribution chart. Copies, safe to serialize.
            "samples": self._collector.all_histogram_samples(),
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

    def _log(self) -> Dict[str, Any]:
        if self._collector is None:
            return {"error": "metrics collector unavailable"}
        metrics = self._collector.get_metrics()
        records = [
            {
                "timestamp": m.timestamp,
                "name": m.name,
                "type": m.metric_type.value,
                "value": m.value,
            }
            for m in metrics[-100:]
        ]
        records.reverse()
        return {"records": records}


class DashboardAPIServer:
    """Serves the monitoring dashboard API over HTTP."""

    def __init__(self, collector: MetricsCollector, port: int = 8080) -> None:
        self._collector: MetricsCollector = collector
        self._port: int = port
        self._server: Optional[ThreadingHTTPServer] = None
        #: Instance-level pre-request hook; used when ``start()``/``run()`` are
        #: called without an explicit ``refresh=`` argument.
        self._refresh: Optional[Callable[[MetricsCollector], None]] = None

    def set_refresh(self, refresh: Optional[Callable[[MetricsCollector], None]]) -> None:
        """Register a pre-request hook (e.g. a shared metrics-file tailer)."""
        self._refresh = refresh

    def get_refresh(self) -> Optional[Callable[[MetricsCollector], None]]:
        """Return the registered pre-request hook (if any)."""
        return self._refresh

    def configure(
        self,
        alert_rules: List[AlertRule],
        evaluate_alerts: Optional[Callable[[List[AlertRule]], List[AlertEvent]]] = None,
        refresh: Optional[Callable[[MetricsCollector], None]] = None,
    ) -> None:
        DashboardAPIHandler._collector = self._collector
        DashboardAPIHandler._alert_rules = list(alert_rules)
        DashboardAPIHandler._evaluate_alerts = evaluate_alerts
        DashboardAPIHandler._refresh = refresh if refresh is not None else self._refresh

    def start(
        self,
        alert_rules: Optional[List[AlertRule]] = None,
        evaluate_alerts: Optional[Callable[[List[AlertRule]], List[AlertEvent]]] = None,
        refresh: Optional[Callable[[MetricsCollector], None]] = None,
    ) -> ThreadingHTTPServer:
        if refresh is not None:
            self.set_refresh(refresh)
        self.configure(alert_rules or [], evaluate_alerts, self._refresh)
        server = ThreadingHTTPServer(("localhost", self._port), DashboardAPIHandler)
        self._server = server
        return server

    def run(
        self,
        alert_rules: Optional[List[AlertRule]] = None,
        evaluate_alerts: Optional[Callable[[List[AlertRule]], List[AlertEvent]]] = None,
        refresh: Optional[Callable[[MetricsCollector], None]] = None,
    ) -> None:
        server = self.start(alert_rules or [], evaluate_alerts, refresh)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Silenced exception in dashboard_api.py:291")
        finally:
            self.stop(server)

    def stop(self, server: ThreadingHTTPServer) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


__all__: List[str] = ["DashboardAPIHandler", "DashboardAPIServer"]
