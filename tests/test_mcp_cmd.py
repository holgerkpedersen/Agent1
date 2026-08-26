"""Tests for the `mcp` REPL command (agent_core/commands/mcp_cmd.py).

Drives the REAL command class against a manager pointed at a temp mcp.json,
with output captured.  The repo-root mcp.json is never touched.
"""
from __future__ import annotations

import asyncio
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from agent_core.commands.mcp_cmd import MCPCommand
from agent_core.mcp.config import save_mcp_servers, validate_entry
from agent_core.mcp.manager import McpManager, get_manager

FIXTURE = [sys.executable,
           str(Path(__file__).parent / "fixtures" / "fake_mcp_server.py")]


@pytest.fixture()
def cmd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MCPCommand:
    """A command whose manager uses a temp config pre-loaded with the fake server."""
    path = tmp_path / "mcp.json"
    cfgs = {"fake": validate_entry("fake", {
        "transport": "stdio", "command": FIXTURE,
    })}
    save_mcp_servers(cfgs, path)
    mgr = McpManager(config_path=path)
    monkeypatch.setattr("agent_core.commands.mcp_cmd.get_manager", lambda: mgr)
    return MCPCommand()


def run(cmd: MCPCommand, *args: str) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        cont = asyncio.run(cmd.execute(list(args), agent=None))  # type: ignore[arg-type]
    assert cont is True  # REPL must always continue after an mcp command
    return buf.getvalue()


class TestStatusAndConfig:
    def test_status_shows_configured_server(self, cmd: MCPCommand) -> None:
        out = run(cmd)
        assert "fake" in out and "stopped" in out and "stdio" in out

    def test_add_stdio_persists_argv(self, cmd: MCPCommand, tmp_path: Path) -> None:
        out = run(cmd, "add", "local2", "stdio", *FIXTURE)
        assert "'local2'" in out
        configs = cmd._manager.configured()
        assert "local2" in configs
        assert configs["local2"].command == FIXTURE

    def test_add_http_with_headers(self, cmd: MCPCommand) -> None:
        run(cmd, "add", "remote", "http", "https://example.com/mcp",
            "--header", "Authorization=secret:tok", "--timeout", "45")
        cfg = cmd._manager.configured()["remote"]
        assert cfg.url == "https://example.com/mcp"
        assert cfg.headers["Authorization"] == "secret:tok"
        assert cfg.timeout_s == 45

    def test_remove_deletes_and_disconnects(self, cmd: MCPCommand) -> None:
        run(cmd, "connect", "fake")
        out = run(cmd, "remove", "fake")
        assert "removed" in out
        assert "fake" not in cmd._manager.configured()
        assert not cmd._manager.is_connected("fake")

    def test_bad_json_args_rejected_cleanly(self, cmd: MCPCommand) -> None:
        run(cmd, "connect", "fake")
        out = run(cmd, "call", "fake", "echo", "{not json")
        assert "[mcp] error" in out


class TestConnectCallFlow:
    def test_connect_then_call_roundtrip(self, cmd: MCPCommand) -> None:
        out = run(cmd, "connect", "fake")
        assert "connected" in out and "(3 tools)" in out
        out = run(cmd, "call", "fake", "add", '{"a": 19, "b": 23}')
        assert "42" in out

    def test_call_without_arguments_object(self, cmd: MCPCommand) -> None:
        run(cmd, "connect", "fake")
        out = run(cmd, "call", "fake", "echo", '{"text": "ping"}')
        assert "ping" in out

    def test_tools_listing(self, cmd: MCPCommand) -> None:
        run(cmd, "connect", "fake")
        out = run(cmd, "tools")
        assert "add" in out and "schema" in out

    def test_disconnect(self, cmd: MCPCommand) -> None:
        run(cmd, "connect", "fake")
        out = run(cmd, "disconnect", "all")
        assert "disconnected" in out
        assert not cmd._manager.is_connected("fake")

    def test_resources_read(self, cmd: MCPCommand) -> None:
        run(cmd, "connect", "fake")
        out = run(cmd, "resources", "fake")
        assert "fake://note" in out
        out = run(cmd, "read", "fake", "fake://note")
        assert "hello from fake resource" in out


class TestExposeToggle:
    def test_expose_on_off_persists(self, cmd: MCPCommand) -> None:
        run(cmd, "expose", "fake", "on")
        assert cmd._manager.configured()["fake"].expose_to_llm is True
        run(cmd, "expose", "fake", "off")
        assert cmd._manager.configured()["fake"].expose_to_llm is False

    def test_expose_requires_valid_state(self, cmd: MCPCommand) -> None:
        out = run(cmd, "expose", "fake", "maybe")
        assert "usage" in out

    def test_expose_unknown_server(self, cmd: MCPCommand) -> None:
        out = run(cmd, "expose", "ghost", "on")
        assert "no server named" in out


class TestRegistryIntegration:
    def test_command_registered_in_agent_registry(self) -> None:
        import agent
        registry = agent._build_registry()
        assert registry.get("mcp") is not None

    def test_singleton_manager_is_shared_with_dashboard_bridge(self) -> None:
        # The dashboard API must observe the same connections the REPL makes.
        from agent_core.mcp import manager as m
        assert get_manager() is m.get_manager()
