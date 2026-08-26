"""Tests for the dashboard MCP surface: /mcp page, /api/mcp/* GET + POST.

Safety behaviours pinned here:
- POST works ONLY for connect/disconnect/call (config mutation is impossible
  over HTTP by construction);
- cross-origin POSTs are rejected (CSRF guard);
- oversized bodies are rejected;
- the page is served and never exposes mcp.json contents.
"""
from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from agent_core.monitoring.dashboard_api import DashboardAPIHandler
from agent_core.mcp.config import save_mcp_servers, validate_entry
from agent_core.mcp.manager import McpManager

FIXTURE = [sys.executable,
           str(Path(__file__).parent / "fixtures" / "fake_mcp_server.py")]


@pytest.fixture()
def server(tmp_path: Path):
    """A live dashboard server whose shared manager uses a temp mcp.json."""
    path = tmp_path / "mcp.json"
    cfgs = {"fake": validate_entry("fake", {
        "transport": "stdio", "command": FIXTURE,
    })}
    save_mcp_servers(cfgs, path)

    mgr = McpManager(config_path=path)
    import agent_core.monitoring.dashboard_api as api_mod
    orig = api_mod.DashboardAPIHandler.__dict__["_mcp"]  # raw staticmethod
    DashboardAPIHandler._mcp = staticmethod(lambda: mgr)  # type: ignore[assignment]

    httpd = ThreadingHTTPServer(("localhost", 0), DashboardAPIHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://localhost:{httpd.server_address[1]}"
    yield base, mgr, path
    httpd.shutdown()
    httpd.server_close()
    mgr.disconnect_all()
    DashboardAPIHandler._mcp = orig  # type: ignore[assignment]


def get(url: str) -> tuple[int, bytes]:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return resp.status, resp.read()


def post(url: str, body: dict, origin: str | None = "http://localhost") -> tuple[int, bytes]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 **({"Origin": origin} if origin else {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


class TestMcpPage:
    def test_page_served(self, server) -> None:
        base, _, _ = server
        status, body = get(base + "/mcp")
        assert status == 200
        assert b"MCP Services" in body

    def test_page_does_not_embed_config(self, server) -> None:
        base, _, config_path = server
        _, body = get(base + "/mcp")
        assert json.dumps(json.loads(config_path.read_text())).encode() not in body


class TestMcpGetApi:
    def test_state_lists_configured_server(self, server) -> None:
        base, _, _ = server
        _, body = get(base + "/api/mcp/state")
        state = json.loads(body)
        names = [s["name"] for s in state["servers"]]
        assert "fake" in names
        assert isinstance(state["llm_catalog"], dict)

    def test_tools_requires_server_param(self, server) -> None:
        base, _, _ = server
        _, body = get(base + "/api/mcp/tools")
        assert "error" in json.loads(body)


class TestMcpPostApi:
    def test_connect_call_disconnect_roundtrip(self, server) -> None:
        base, mgr, _ = server
        status, body = post(base + "/api/mcp/connect", {"name": "fake"})
        assert status == 200 and json.loads(body)["ok"] is True
        assert mgr.is_connected("fake")

        status, body = post(base + "/api/mcp/call",
                            {"server": "fake", "tool": "add",
                             "arguments": {"a": 40, "b": 2}})
        out = json.loads(body)
        assert out["ok"] is True and out["result"] == "42"

        status, _ = post(base + "/api/mcp/disconnect", {"name": "fake"})
        assert status == 200
        assert not mgr.is_connected("fake")

    def test_call_unknown_server_reports_error_not_crash(self, server) -> None:
        base, _, _ = server
        status, body = post(base + "/api/mcp/call",
                            {"server": "ghost", "tool": "x"})
        out = json.loads(body)
        assert out["ok"] is False and "ghost" in out["error"]

    def test_unknown_post_path_404s(self, server) -> None:
        base, _, _ = server
        status, _ = post(base + "/api/mcp/remove-config-entry", {"name": "x"})
        assert status == 404

    def test_cross_origin_post_rejected(self, server) -> None:
        base, mgr, _ = server
        status, _ = post(base + "/api/mcp/connect", {"name": "fake"},
                         origin="https://evil.example.com")
        assert status == 403
        assert not mgr.is_connected("fake")

    def test_no_origin_header_allowed_for_local_tools(self, server) -> None:
        # Same-origin fetches from the served page may omit Origin entirely.
        base, _, _ = server
        status, _ = post(base + "/api/mcp/connect", {"name": "fake"}, origin=None)
        assert status == 200

    def test_oversized_body_rejected(self, server) -> None:
        base, _, _ = server
        big = {"pad": "x" * (DashboardAPIHandler._MCP_MAX_BODY + 10)}
        status, _ = post(base + "/api/mcp/call", big)
        assert status == 413
