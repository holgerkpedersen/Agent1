"""End-to-end tests for the MCP consumer core (jsonrpc/transports/config/client).

Everything runs against tests/fixtures/fake_mcp_server.py over real pipes -
no network, no external packages.  Safety behaviours pinned here:

- per-request wall-clock cap (slow tool -> TransportError, not a hang)
- schema validation BEFORE the call leaves the process
- fail-closed secret references in config
- argv-list-only stdio spawning (no shell anywhere)
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from agent_core.mcp import jsonrpc
from agent_core.mcp.client import (
    MAX_RESULT_CHARS,
    McpClient,
    McpProtocolError,
    McpToolError,
    truncate_result,
    validate_against_schema,
)
from agent_core.mcp.config import (
    McpConfigError,
    load_mcp_servers,
    resolve_value,
    save_mcp_servers,
    validate_entry,
)
from agent_core.mcp.manager import McpManager, McpManagerError, get_manager
from agent_core.mcp.transports import StdioTransport, TransportError

FIXTURE = Path(__file__).parent / "fixtures" / "fake_mcp_server.py"
FAKE_SERVER_CMD = [sys.executable, str(FIXTURE)]


def make_client(timeout_s: float = 30) -> McpClient:
    transport = StdioTransport(FAKE_SERVER_CMD)
    client = McpClient(transport, timeout_s=timeout_s)
    client.initialize()
    client.refresh_tools()
    return client


# ---------------------------------------------------------------------------
# jsonrpc framing
# ---------------------------------------------------------------------------

class TestJsonRpc:
    def test_request_envelope_and_unique_ids(self) -> None:
        a = jsonrpc.make_request("tools/list")
        b = jsonrpc.make_request("tools/list")
        assert a["jsonrpc"] == "2.0" and a["method"] == "tools/list"
        assert a["id"] != b["id"]

    def test_notification_has_no_id(self) -> None:
        note = jsonrpc.make_request("notifications/initialized", {})
        assert "id" not in note

    def test_parse_rejects_garbage(self) -> None:
        with pytest.raises(jsonrpc.JsonRpcError) as ei:
            jsonrpc.parse_message("{not json")
        assert ei.value.code == jsonrpc.PARSE_ERROR

    def test_parse_rejects_non_object(self) -> None:
        with pytest.raises(jsonrpc.JsonRpcError):
            jsonrpc.parse_message("[1,2,3]")

    def test_result_of_raises_typed_error(self) -> None:
        msg = {"jsonrpc": "2.0", "id": 1,
               "error": {"code": -32601, "message": "nope"}}
        with pytest.raises(jsonrpc.JsonRpcError) as ei:
            jsonrpc.result_of(msg)
        assert ei.value.code == -32601 and "nope" in ei.value.message


# ---------------------------------------------------------------------------
# config validation + secret resolution
# ---------------------------------------------------------------------------

class TestMcpConfig:
    def test_stdio_entry_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "mcp.json"
        cfgs = {
            "fs": validate_entry("fs", {
                "transport": "stdio", "command": ["python", "-m", "x"],
                "expose_to_llm": True,
            }),
        }
        save_mcp_servers(cfgs, path)
        loaded = load_mcp_servers(path)
        assert loaded["fs"].command == ["python", "-m", "x"]
        assert loaded["fs"].expose_to_llm is True
        # atomic write leaves no tmp litter
        assert not (tmp_path / "mcp.json.tmp").exists()

    def test_http_entry_requires_url_scheme(self) -> None:
        with pytest.raises(McpConfigError):
            validate_entry("bad", {"transport": "http", "url": "ftp://x"})
        ok = validate_entry("ok", {"transport": "http", "url": "https://a/mcp"})
        assert ok.url == "https://a/mcp"

    def test_stdio_requires_argv_list(self) -> None:
        with pytest.raises(McpConfigError):
            validate_entry("bad", {"transport": "stdio", "command": "rm -rf /"})
        with pytest.raises(McpConfigError):
            validate_entry("bad2", {"transport": "stdio"})

    def test_unknown_transport_rejected(self) -> None:
        with pytest.raises(McpConfigError):
            validate_entry("bad", {"transport": "carrier-pigeon"})

    def test_timeout_clamped(self) -> None:
        huge = validate_entry("t", {"transport": "http",
                                    "url": "http://x", "timeout_s": 10_000})
        assert huge.timeout_s == 600

    def test_missing_file_loads_empty(self, tmp_path: Path) -> None:
        assert load_mcp_servers(tmp_path / "absent.json") == {}

    def test_secret_ref_resolved_from_keyring(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "agent_core.security.secrets.get_secret",
            lambda name, default="": "tok-123" if name == "gh" else "",
        )
        assert resolve_value("secret:gh", "h") == "tok-123"

    def test_dangling_secret_ref_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "agent_core.security.secrets.get_secret", lambda name, default="": "",
        )
        with pytest.raises(McpConfigError):
            resolve_value("secret:missing", "h")

    def test_env_ref_resolution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_MCP_TOKEN", "abc")
        assert resolve_value("${MY_MCP_TOKEN}", "h") == "abc"
        with pytest.raises(McpConfigError):
            resolve_value("${NOT_SET_ANYWHERE_9137}", "h")


# ---------------------------------------------------------------------------
# end-to-end over real stdio pipes
# ---------------------------------------------------------------------------

class TestStdioEndToEnd:
    def test_initialize_tools_call(self) -> None:
        client = make_client()
        try:
            assert client.server_info["serverInfo"]["name"] == "fake-mcp"
            names = client.tool_names()
            assert names == ["echo", "add", "slow"]
            assert client.call_tool("add", {"a": 2, "b": 40}) == "42"
            assert client.call_tool("echo", {"text": "hi"}) == "hi"
        finally:
            client.close()

    def test_schema_violation_blocked_before_send(self) -> None:
        client = make_client()
        try:
            with pytest.raises(McpProtocolError, match="argument validation"):
                client.call_tool("add", {"a": "two", "b": 1})
            with pytest.raises(McpProtocolError, match="missing required"):
                client.call_tool("add", {"a": 1})
        finally:
            client.close()

    def test_unknown_tool_rejected(self) -> None:
        client = make_client()
        try:
            with pytest.raises(McpProtocolError, match="unknown tool"):
                client.call_tool("does_not_exist", {})
        finally:
            client.close()

    def test_timeout_is_capped_not_hanging(self) -> None:
        client = make_client(timeout_s=2)
        try:
            with pytest.raises(TransportError, match="within 2s"):
                client.call_tool("slow", {"seconds": 30})
        finally:
            client.close()

    def test_resources(self) -> None:
        client = make_client()
        try:
            res = client.resources_list()
            assert res[0]["uri"] == "fake://note"
            assert "hello from fake resource" in client.resources_read("fake://note")
        finally:
            client.close()

    def test_close_kills_child_process(self) -> None:
        client = make_client()
        transport = client._transport
        assert isinstance(transport, StdioTransport)
        proc = transport._proc
        assert proc is not None
        client.close()
        assert proc.poll() is not None, "child must be terminated on close"

    def test_dead_server_surfaces_as_error(self) -> None:
        transport = StdioTransport([sys.executable, "-c", "import sys; sys.exit(0)"])
        client = McpClient(transport, timeout_s=10)
        with pytest.raises((TransportError, McpProtocolError)):
            client.initialize()


# ---------------------------------------------------------------------------
# result capping
# ---------------------------------------------------------------------------

class TestResultCap:
    def test_truncate_result_marks_overflow(self) -> None:
        big = "x" * (MAX_RESULT_CHARS + 500)
        out = truncate_result(big)
        assert len(out) < len(big)
        assert "[truncated" in out

    def test_small_result_untouched(self) -> None:
        assert truncate_result("fine") == "fine"


class TestSchemaValidation:
    def test_nested_object_and_array(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
                "meta": {"type": "object",
                         "properties": {"n": {"type": "integer"}}},
            },
            "required": ["tags"],
        }
        assert validate_against_schema(schema, {"tags": ["a"], "meta": {"n": 1}}) is None
        err = validate_against_schema(schema, {"tags": [1]})
        assert err and "tags[0]" in err
        err = validate_against_schema(schema, {"meta": {}})
        assert err and "required" in err

    def test_enum_and_boolean(self) -> None:
        s = {"type": "string", "enum": ["a", "b"]}
        assert validate_against_schema(s, "a") is None
        assert validate_against_schema(s, "z") is not None
        assert validate_against_schema({"type": "boolean"}, True) is None
        assert validate_against_schema({"type": "boolean"}, "true") is not None


# ---------------------------------------------------------------------------
# manager lifecycle (uses a temp mcp.json so the repo root is never touched)
# ---------------------------------------------------------------------------

class TestManagerLifecycle:
    @pytest.fixture()
    def manager(self, tmp_path: Path) -> McpManager:
        path = tmp_path / "mcp.json"
        cfgs = {"fake": validate_entry(
            "fake", {"transport": "stdio", "command": FAKE_SERVER_CMD},
        )}
        save_mcp_servers(cfgs, path)
        return McpManager(config_path=path)

    def test_connect_status_tools_call_disconnect(self, manager: McpManager) -> None:
        st = manager.connect("fake")
        assert st["state"] == "connected"
        assert st["tool_count"] == 3
        tools = manager.tools("fake")
        assert {t["name"] for t in tools} >= {"echo", "add"}
        assert manager.call_tool("fake", "add", {"a": 20, "b": 22}) == "42"
        manager.disconnect_all()
        assert manager.status_one("fake")["state"] == "stopped"

    def test_connect_unknown_server_raises(self, manager: McpManager) -> None:
        with pytest.raises(McpManagerError, match="no MCP server named"):
            manager.connect("ghost")

    def test_call_before_connect_raises_with_hint(self, manager: McpManager) -> None:
        with pytest.raises(McpManagerError, match="mcp connect"):
            manager.call_tool("fake", "add", {})

    def test_error_state_recorded_on_failure(self, tmp_path: Path) -> None:
        path = tmp_path / "mcp.json"
        cfgs = {"broken": validate_entry(
            "broken", {"transport": "stdio",
                       "command": [sys.executable, "-c", "import sys; sys.exit(3)"]},
        )}
        save_mcp_servers(cfgs, path)
        mgr = McpManager(config_path=path)
        with pytest.raises(Exception):
            mgr.connect("broken")
        st = mgr.status_one("broken")
        assert st["state"] == "error" and st["detail"]

    def test_llm_catalog_only_exposed_servers(self, tmp_path: Path) -> None:
        path = tmp_path / "mcp.json"
        cfgs = {
            "open": validate_entry("open", {
                "transport": "stdio", "command": FAKE_SERVER_CMD,
                "expose_to_llm": True,
            }),
            "shut": validate_entry("shut", {
                "transport": "stdio", "command": FAKE_SERVER_CMD,
            }),
        }
        save_mcp_servers(cfgs, path)
        mgr = McpManager(config_path=path)
        assert mgr.llm_catalog() == {}  # nothing connected yet
        mgr.connect("open")
        mgr.connect("shut")
        catalog = mgr.llm_catalog()
        assert set(catalog) == {"open"}
        assert "echo" in catalog["open"]
        mgr.disconnect_all()

    def test_shared_singleton_is_stable(self) -> None:
        assert get_manager() is get_manager()


# ---------------------------------------------------------------------------
# HTTP transport against a threaded local server (JSON + SSE bodies)
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    """Canned JSON-RPC-over-HTTP endpoint: JSON on /, SSE on /sse."""

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("content-length", 0))
        req = jsonrpc.parse_message(self.rfile.read(length))
        if self.path == "/sse":
            body = ('data: {"jsonrpc":"2.0","id":%s,"result":{"ok":true}}\n\n'
                    % json.dumps(req["id"]))
            ctype = "text/event-stream"
        else:
            body = json.dumps({"jsonrpc": "2.0", "id": req["id"],
                               "result": {"ok": True}})
            ctype = "application/json"
        payload = body.encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: object) -> None:
        pass


class TestHttpTransport:
    @pytest.fixture()
    def server_url(self):
        import threading

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
        httpd.shutdown()
        httpd.server_close()

    def test_plain_json_roundtrip(self, server_url: str) -> None:
        from agent_core.mcp.transports import HttpTransport
        t = HttpTransport(server_url)
        assert t.request("ping", {}, 10) == {"ok": True}

    def test_sse_body_parsed(self, server_url: str) -> None:
        from agent_core.mcp.transports import HttpTransport
        t = HttpTransport(server_url + "/sse")
        assert t.request("ping", {}, 10) == {"ok": True}

    def test_bad_scheme_rejected_upfront(self) -> None:
        from agent_core.mcp.transports import HttpTransport
        with pytest.raises(TransportError, match="scheme"):
            HttpTransport("gopher://x")
