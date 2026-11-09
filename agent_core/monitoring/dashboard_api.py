"""Dashboard API exposing monitoring metrics and alerts over HTTP."""

from __future__ import annotations

import json
import os
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
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
    _base_dir: str = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    def do_GET(self) -> None:
        self._maybe_refresh()
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        params = parse_qs(parsed_path.query)
        if path in ("/", "/index.html"):
            self._send_html_page()
        elif path == "/mcp":
            self._send_mcp_page()
        elif path == "/autonomous" or path == "/autonomous.html":
            self._send_autonomous_page()
        elif path == "/api/autonomous/status":
            self._send_json(self._autonomous_status())
        elif path == "/api/snapshot":
            self._send_json(self._snapshot())
        elif path == "/api/counters":
            self._send_json(self._counters())
        elif path == "/api/gauges":
            self._send_json(self._gauges())
        elif path == "/api/histograms":
            self._send_json(self._histogram(params))
        elif path == "/api/tokens":
            self._send_json(self._tokens())
        elif path == "/api/prompt_cache":
            self._send_json(self._prompt_cache())
        elif path == "/api/alerts":
            self._send_json(self._alerts())
        elif path == "/api/log":
            self._send_json(self._log())
        elif path == "/api/mcp/state":
            self._send_json(self._mcp_state())
        elif path == "/api/mcp/tools":
            self._send_json(self._mcp_tools(params))
        elif path.startswith("/static/"):
            self._send_static(path)
        else:
            self.send_error(404, "Not found")

    # -- MCP endpoints -----------------------------------------------------
    #
    # Safety contract for everything under /mcp* :
    # - the server binds localhost ONLY (see DashboardAPIServer.start);
    # - POST is accepted solely for connect/disconnect/call on ALREADY
    #   configured servers - there is NO endpoint that can read or write
    #   mcp.json, so a browser can never alter configuration;
    # - cross-origin POSTs are rejected (CSRF guard) before any work.

    _MCP_POST_PATHS = ("/api/mcp/connect", "/api/mcp/disconnect", "/api/mcp/call")
    _MCP_MAX_BODY = 64 * 1024

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in self._MCP_POST_PATHS:
            self.send_error(404, "Not found")
            return
        # CSRF / DNS-rebinding guards: the dashboard binds localhost only,
        # so both the Origin (browser requests) and the Host header must
        # name a loopback host EXACTLY - "http://localhost.evil.com" must
        # not slip through a prefix match.
        origin = self.headers.get("Origin", "")
        if origin:
            try:
                ohost = urlparse(origin).hostname or ""
            except ValueError:
                ohost = ""
            if ohost not in ("localhost", "127.0.0.1", "::1"):
                self.send_error(403, "Cross-origin MCP calls are not allowed")
                return
        host = (self.headers.get("Host") or "").split(":")[0].strip("[]")
        if host and host not in ("localhost", "127.0.0.1", "::1"):
            self.send_error(403, "Host header is not loopback")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = -1
        if not 0 <= length <= self._MCP_MAX_BODY:
            # Drain (bounded) the rejected body and close the connection:
            # answering while the request body is still in flight leaves
            # unread bytes on the socket, and on Windows the abort that
            # follows can reach the client before the 413 response does
            # (WinError 10053 in urllib).  Closing makes the close explicit
            # instead of relying on the OS teardown.
            self.close_connection = True
            self._drain_request_body(length)
            self.send_error(413, "Body too large")
            return
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("body must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_json({"ok": False, "error": f"bad request body: {exc}"})
            return
        name = str(payload.get("name", ""))
        try:
            if path == "/api/mcp/connect":
                result = self._mcp().connect(name)
                self._send_json({"ok": True, "status": result})
            elif path == "/api/mcp/disconnect":
                self._mcp().disconnect(name)
                self._send_json({"ok": True})
            else:
                text = self._mcp().call_tool(
                    str(payload.get("server", "")),
                    str(payload.get("tool", "")),
                    payload.get("arguments") or {},
                )
                self._send_json({"ok": True, "result": text})
        except Exception as exc:  # surface manager/protocol errors to the UI
            self._send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @staticmethod
    def _mcp() -> Any:
        # Late import: keeps monitoring free of any hard dependency on the
        # mcp package when only metrics are served.
        from agent_core.mcp.manager import get_manager

        return get_manager()

    def _mcp_state(self) -> Dict[str, Any]:
        mgr = self._mcp()
        state = {
            "servers": mgr.status(),
            "llm_catalog": mgr.llm_catalog(),
        }
        return state

    def _mcp_tools(self, params: Dict[str, List[str]]) -> Dict[str, Any]:
        names = params.get("server", [])
        if not names:
            return {"error": "missing ?server="}
        try:
            return {"server": names[0], "tools": self._mcp().tools(names[0])}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _send_mcp_page(self) -> None:
        """Serve static/mcp.html verbatim."""
        page_path = os.path.join(self._base_dir, "static", "mcp.html")
        if not os.path.isfile(page_path):
            self.send_error(404, "mcp.html not found")
            return
        with open(page_path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_autonomous_page(self) -> None:
        """Serve static/autonomous.html verbatim (real-time autonomous view)."""
        page_path = os.path.join(self._base_dir, "static", "autonomous.html")
        if not os.path.isfile(page_path):
            self.send_error(404, "autonomous.html not found")
            return
        with open(page_path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    #: A run-status heartbeat older than this (seconds) means the process is
    #: assumed dead even if it left ``running=true`` (e.g. it was killed).
    _AUTONOMOUS_STALE_SECONDS = 180

    def _autonomous_status(self) -> Dict[str, Any]:
        """Assemble the live autonomous-run state for the dashboard.

        Merges the per-phase beacon (run_status.json), the latest finished
        iteration summary (summary.json), the cross-iteration history
        (run_history.jsonl) and the most recent commits, plus a derived
        ``running`` flag so the UI can show a live/idle indicator.  ``history``
        is delivered newest-first (by timestamp, iteration as tiebreaker) so the
        dashboard table shows the latest iteration at the top.
        """
        from harnessfix.progress import read_history, read_progress

        status = read_progress()
        # Derive liveness: explicit running flag AND a recent heartbeat.
        now = time.time()
        ts = status.get("ts")
        stale = (
            not isinstance(ts, (int, float))
            or (now - float(ts)) > self._AUTONOMOUS_STALE_SECONDS
        )
        running = bool(status.get("running")) and not stale
        if "running" in status:
            status = dict(status)
            status["running"] = running

        # Latest summary.json (written at end of each iteration).
        summary: Dict[str, Any] = {}
        summary_path = os.path.join(
            self._base_dir, "reports", "harnessfix", "summary.json"
        )
        if os.path.isfile(summary_path):
            try:
                summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                summary = {}

        history = read_history(limit=50)
        # The dashboard table reads the most recent iteration first, so deliver
        # history newest-first (records are appended chronologically; sort by
        # timestamp with iteration as a stable tiebreaker).
        history = sorted(
            history,
            key=lambda r: (r.get("timestamp") or 0, r.get("iteration") or 0),
            reverse=True,
        )

        # Recent commits touching the autonomous run (most recent first).
        recent_commits: list[str] = []
        try:
            out = subprocess.run(
                ["git", "log", "-5", "--oneline"],
                cwd=self._base_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=5,
            )
            if out.returncode == 0:
                recent_commits = [ln for ln in out.stdout.splitlines() if ln.strip()]
        except Exception:
            recent_commits = []

        # Wiki knowledge state (WikiSkill layer 2). Fail-open: absent/corrupt
        # wiki yields an empty dict so the dashboard degrades gracefully.
        wiki_stats: Dict[str, Any] = {}
        try:
            from harnessfix.wiki import wiki_stats as _wiki_stats

            wiki_path = os.path.join(self._base_dir, "reports", "wiki", "wiki.jsonl")
            if not os.path.isfile(wiki_path):
                # Fall back to the repo-root wiki (module-level default).
                wiki_path = None
            wiki_stats = _wiki_stats(path=Path(wiki_path) if wiki_path else None) or {}
        except Exception:
            pass

        return {
            "status": status,
            "summary": summary,
            "history": history,
            "recent_commits": recent_commits,
            "wiki": wiki_stats,
            "server_time": now,
        }

    def _drain_request_body(self, length: int) -> None:
        """Read and discard up to ``_MCP_MAX_BODY`` bytes of request body.

        Used on the 413 path: leaving the rejected body in the socket buffer
        makes the connection's teardown an abort on Windows, which can reach
        the client before the response does.  The drain is bounded so a
        hostile Content-Length cannot make us read unbounded data.
        """
        remaining = min(max(length, 0), self._MCP_MAX_BODY)
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 8192))
            if not chunk:
                break
            remaining -= len(chunk)

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
        base = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
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

    def _tokens(self) -> Dict[str, Any]:
        """Token efficiency: live collector metrics + read-only ledger history.

        Counters/histograms come from the shared MetricsCollector (fed by
        ``record_plan_token_metrics`` at plan finish, mirrored across
        processes via the event file).  The per-plan history is read from the
        ``executions.jsonl`` ledger lines written by ``write_plan_evaluation``
        - strictly read-only: this endpoint never writes any file and touches
        no Kanban surface.
        """
        if self._collector is None:
            return {"error": "metrics collector unavailable"}
        snapshot = self._collector.snapshot()
        counters = self._extract_metrics(snapshot, "counters")
        gauges = self._extract_metrics(snapshot, "gauges")

        def _summary(name: str) -> Dict[str, float]:
            return self._collector.histogram_summary(name)

        plans_finished = counters.get("plans.finished.count", 0.0)
        tokens_total = counters.get("tokens.plan.total", 0.0)
        response: Dict[str, Any] = {
            "plans_finished": plans_finished,
            "tokens_total": tokens_total,
            # Mean tokens per finished plan (facts only; 0 plans -> None).
            "tokens_per_plan": (
                tokens_total / plans_finished if plans_finished else None
            ),
            "last_tokens": gauges.get("last.tokens.plan.total"),
            "last_score": gauges.get("last.plans.efficiency.score"),
            "total_summary": _summary("tokens.plan.total"),
            "per_task_summary": _summary("tokens.plan.per_task"),
            # Per-component distributions (prompt / completion) recorded by
            # record_token_histograms; summaries of empty histograms are
            # count=0, so the panel shows nothing until facts exist.
            "prompt_summary": _summary("tokens.run.prompt"),
            "completion_summary": _summary("tokens.run.completion"),
            "score_summary": _summary("plans.efficiency.score"),
            "history": self._token_history(limit=20),
        }
        return response

    #: Ledger note format written by harnessfix.plan_eval.write_plan_evaluation.
    _TOKEN_NOTE_RE = None  # compiled lazily below

    @classmethod
    def _token_note_re(cls):
        if cls._TOKEN_NOTE_RE is None:
            import re as _re

            cls._TOKEN_NOTE_RE = _re.compile(
                r"plan (?P<plan_id>.*?): score=(?P<score>[\d.]+) "
                r"tasks=(?P<completed>\S+)/(?P<planned>\S+) "
                r"tokens=(?P<tokens>\d+|None)"
            )
        return cls._TOKEN_NOTE_RE

    def _token_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Newest ``limit`` plan evaluations from the execution ledger.

        Parses the note of every ``plan_evaluation`` line in
        ``reports/history/executions.jsonl`` (the artifact Phase 1/2 write).
        Read-only; any unreadable/corrupt source degrades to an empty list.
        """
        import json as _json
        import os as _os

        ledger = _os.path.join(
            self._base_dir, "reports", "history", "executions.jsonl"
        )
        records: List[Dict[str, Any]] = []
        try:
            with open(ledger, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
        except OSError:
            return records
        rx = self._token_note_re()
        for raw in reversed(lines):
            try:
                line = _json.loads(raw)
            except Exception:
                continue
            if not isinstance(line, dict) or line.get("command") != "plan_evaluation":
                continue
            m = rx.search(str(line.get("note", "")))
            if not m:
                continue

            def _num(text: str):
                return None if text == "None" else int(text)

            records.append(
                {
                    "ts": line.get("ts"),
                    "plan_id": m.group("plan_id"),
                    "outcome": line.get("outcome"),
                    "score": float(m.group("score")),
                    "tasks_completed": _num(m.group("completed")),
                    "tasks_planned": _num(m.group("planned")),
                    "tokens": _num(m.group("tokens")),
                }
            )
            if len(records) >= limit:
                break
        return records

    def _prompt_cache(self) -> Dict[str, Any]:
        """Prompt/template-cache hit-miss stats (overall + per command)."""
        try:
            from agent_core.llm.prompt_cache import get_prompt_cache_stats

            data = get_prompt_cache_stats()
            if not isinstance(data, dict):
                data = {}
            data.setdefault("overall", {})
            data.setdefault("by_command", {})
            return data
        except Exception as exc:
            return {
                "error": f"{type(exc).__name__}: {exc}",
                "overall": {},
                "by_command": {},
            }

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

    def set_refresh(
        self, refresh: Optional[Callable[[MetricsCollector], None]]
    ) -> None:
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
