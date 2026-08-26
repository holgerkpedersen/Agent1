"""Regression tests for the MCP safety review findings (2026-08-26).

F1: mcp.json must be gitignored (server URLs / env keys never enter git).
F2: dashboard POST origin/host checks must match loopback EXACTLY -
    "http://localhost.evil.com" must not pass a prefix check.
F3: the LLM bridge's expose_to_llm gate must be re-validated under the
    manager lock at call time (no TOCTOU between gate check and invoke).
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

FIXTURE = [sys.executable,
           str(Path(__file__).parent / "fixtures" / "fake_mcp_server.py")]


class TestF1Gitignored:
    def test_mcp_json_is_ignored(self) -> None:
        import pathlib
        gi = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        assert any(
            line.strip() == "mcp.json" for line in gi.splitlines()
        ), "mcp.json missing from .gitignore - config could be committed"

    def test_git_ignores_the_actual_file(self, tmp_path: Path) -> None:
        import subprocess
        r = subprocess.run(
            ["git", "check-ignore", "-q", "mcp.json"],
            capture_output=True,
        )
        assert r.returncode == 0, "git does not ignore mcp.json"


class _Handler:
    """Canned JSON-RPC-over-HTTP endpoint for origin/host tests."""

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        from http.server import BaseHTTPRequestHandler
        handler: BaseHTTPRequestHandler = self  # type: ignore[assignment]
        length = int(handler.headers.get("content-length", 0))
        body = handler.rfile.read(length)
        payload = json.dumps({"ok": True, "echo_host":
                              handler.headers.get("Host", "")}).encode()
        handler.send_response(200)
        handler.send_header("content-type", "application/json")
        handler.send_header("content-length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)


class TestF2OriginHostExactMatch:
    @pytest.fixture()
    def server_url(self):
        import threading
        from http.server import ThreadingHTTPServer

        from agent_core.monitoring.dashboard_api import DashboardAPIHandler
        httpd = ThreadingHTTPServer(("localhost", 0), DashboardAPIHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield f"http://localhost:{httpd.server_address[1]}"
        httpd.shutdown()
        httpd.server_close()

    @staticmethod
    def post(url: str, host_header: str | None = None,
             origin: str | None = None) -> int:
        headers = {"Content-Type": "application/json"}
        if host_header is not None:
            headers["Host"] = host_header
        if origin is not None:
            headers["Origin"] = origin
        req = urllib.request.Request(
            url + "/api/mcp/call",
            data=json.dumps({"server": "x", "tool": "y"}).encode(),
            headers=headers, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status
        except urllib.error.HTTPError as exc:
            return exc.code

    def test_lookalike_origin_rejected(self, server_url: str) -> None:
        status = self.post(server_url, origin="http://localhost.evil.com")
        assert status == 403

    def test_evil_origin_rejected(self, server_url: str) -> None:
        status = self.post(server_url, origin="https://evil.example")
        assert status == 403

    def test_exact_localhost_origin_allowed(self, server_url: str) -> None:
        assert self.post(server_url, origin="http://localhost") == 200

    def test_nonloopback_host_header_rejected(self, server_url: str) -> None:
        # DNS-rebinding shape: request ARRIVES at 127.0.0.1 but claims a
        # remote hostname.
        status = self.post(server_url, host_header="attacker.example:8080")
        assert status == 403

    def test_loopback_host_with_port_allowed(self, server_url: str) -> None:
        port = server_url.rsplit(":", 1)[1]
        assert self.post(server_url, host_header=f"127.0.0.1:{port}") == 200


class TestF3ExposeGateAtCallTime:
    @pytest.fixture()
    def mgr(self, tmp_path: Path):
        from agent_core.mcp.config import save_mcp_servers, validate_entry
        from agent_core.mcp.manager import McpManager
        path = tmp_path / "mcp.json"
        cfgs = {"open": validate_entry("open", {
            "transport": "stdio", "command": FIXTURE, "expose_to_llm": True,
        })}
        save_mcp_servers(cfgs, path)
        return McpManager(config_path=path)

    def test_expose_off_between_check_and_call_blocks(self, mgr) -> None:
        from agent_core.mcp.manager import McpManagerError
        mgr.connect("open")
        try:
            # The caller's stale snapshot says exposed...
            assert "open" in mgr.llm_catalog()
            configs = mgr.configured()
            configs["open"].expose_to_llm = False
            mgr.save_config(configs)
            # ...but call_tool(require_exposed=True) re-reads and blocks.
            with pytest.raises(McpManagerError, match="not exposed"):
                mgr.call_tool("open", "add", {"a": 1, "b": 2},
                              require_exposed=True)
        finally:
            mgr.disconnect_all()

    def test_require_exposed_true_still_works_when_on(self, mgr) -> None:
        mgr.connect("open")
        try:
            out = mgr.call_tool("open", "add", {"a": 3, "b": 4},
                                require_exposed=True)
            assert out == "7"
        finally:
            mgr.disconnect_all()

    def test_plain_call_unaffected_by_flag(self, mgr) -> None:
        # REPL/dashboard calls keep working regardless of the LLM flag.
        mgr.connect("open")
        configs = mgr.configured()
        configs["open"].expose_to_llm = False
        mgr.save_config(configs)
        try:
            assert mgr.call_tool("open", "add", {"a": 1, "b": 1}) == "2"
        finally:
            mgr.disconnect_all()

    def test_nlp_handler_goes_through_locked_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import asyncio

        import agent
        from agent_core.mcp.config import save_mcp_servers, validate_entry
        from agent_core.mcp.manager import McpManager
        path = tmp_path / "mcp.json"
        cfgs = {"open": validate_entry("open", {
            "transport": "stdio", "command": FIXTURE, "expose_to_llm": True,
        })}
        save_mcp_servers(cfgs, path)
        mgr = McpManager(config_path=path)
        monkeypatch.setattr("agent_core.mcp.manager.get_manager", lambda: mgr)
        a = agent.Agent(workspace=str(tmp_path))
        mgr.connect("open")
        configs = mgr.configured()
        configs["open"].expose_to_llm = False
        mgr.save_config(configs)
        try:
            out = asyncio.run(a._nlp_mcp_call({
                "server": "open", "tool": "add", "arguments": {"a": 1, "b": 1},
            }))
            assert "not exposed" in out
        finally:
            mgr.disconnect_all()
