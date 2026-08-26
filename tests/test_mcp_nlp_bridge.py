"""Tests for the MCP NLP bridge (mcp_tools / mcp_call) and its safety gates.

The bridge is opt-in: only servers marked ``expose_to_llm`` in mcp.json AND
currently connected are reachable, and the gate is re-checked at call time.
Plan mode excludes both tools automatically (not in PLAN_MODE_TOOLS).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

import agent
from agent_core.mcp.config import save_mcp_servers, validate_entry
from agent_core.mcp.manager import McpManager
from agent_core.tool_schemas import NLP_TOOL_NAMES

FIXTURE = [sys.executable,
           str(Path(__file__).parent / "fixtures" / "fake_mcp_server.py")]


@pytest.fixture()
def bot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "mcp.json"
    cfgs = {
        "open": validate_entry("open", {
            "transport": "stdio", "command": FIXTURE, "expose_to_llm": True,
        }),
        "shut": validate_entry("shut", {
            "transport": "stdio", "command": FIXTURE,
        }),
    }
    save_mcp_servers(cfgs, path)
    mgr = McpManager(config_path=path)
    monkeypatch.setattr(
        "agent_core.mcp.manager.get_manager", lambda: mgr,
    )
    # The handlers late-import get_manager from agent_core.mcp.manager, so
    # patching there covers every call path in this test module.
    a = agent.Agent(workspace=str(tmp_path))
    a._test_mcp_manager = mgr  # tests fetch THIS instance, not the singleton
    yield a
    mgr.disconnect_all()


def run(coro) -> str:
    return asyncio.run(coro)


class TestMcpToolsListing:
    def test_empty_when_nothing_exposed_or_connected(self, bot: agent.Agent) -> None:
        out = run(bot._nlp_mcp_tools({}))
        assert "No MCP tools available" in out

    def test_lists_only_exposed_connected(self, bot: agent.Agent) -> None:
        mgr = bot._test_mcp_manager
        mgr.connect("open")
        mgr.connect("shut")
        try:
            out = run(bot._nlp_mcp_tools({}))
            assert "open.echo" in out
            assert "shut." not in out
        finally:
            mgr.disconnect_all()


class TestMcpCallGate:
    def test_call_on_exposed_server_roundtrip(self, bot: agent.Agent) -> None:
        mgr = bot._test_mcp_manager
        mgr.connect("open")
        try:
            out = run(bot._nlp_mcp_call({
                "server": "open", "tool": "add", "arguments": {"a": 5, "b": 7},
            }))
            assert out == "12"
        finally:
            mgr.disconnect_all()

    def test_non_exposed_server_rejected_even_if_connected(
        self, bot: agent.Agent,
    ) -> None:
        mgr = bot._test_mcp_manager
        mgr.connect("shut")
        try:
            out = run(bot._nlp_mcp_call({"server": "shut", "tool": "echo"}))
            assert "not exposed" in out
        finally:
            mgr.disconnect_all()

    def test_unknown_server_rejected(self, bot: agent.Agent) -> None:
        out = run(bot._nlp_mcp_call({"server": "ghost", "tool": "x"}))
        assert "not exposed" in out

    def test_missing_args_rejected(self, bot: agent.Agent) -> None:
        out = run(bot._nlp_mcp_call({"server": "open"}))
        assert "needs both" in out

    def test_schema_violation_blocked_before_send(self, bot: agent.Agent) -> None:
        mgr = bot._test_mcp_manager
        mgr.connect("open")
        try:
            out = run(bot._nlp_mcp_call({
                "server": "open", "tool": "add",
                "arguments": {"a": "two", "b": 1},
            }))
            assert "argument validation" in out
        finally:
            mgr.disconnect_all()

    def test_non_dict_arguments_rejected(self, bot: agent.Agent) -> None:
        mgr = bot._test_mcp_manager
        mgr.connect("open")
        try:
            out = run(bot._nlp_mcp_call({
                "server": "open", "tool": "add", "arguments": [1, 2],
            }))
            assert "must be a JSON object" in out
        finally:
            mgr.disconnect_all()


class TestModeAndParity:
    def test_bridge_tools_not_in_plan_mode(self) -> None:
        # Both tools can trigger external side effects - plan mode must
        # filter the schemas AND reject calls at the choke point.
        from agent_core.modes import check_tool_allowed, filter_tool_schemas
        rejection = check_tool_allowed("mcp_call", "plan")
        assert rejection is not None
        names = {
            s["function"]["name"]
            for s in filter_tool_schemas(list(agent.NLP_TOOL_SCHEMAS), "plan")
        }
        assert "mcp_call" not in names and "mcp_tools" not in names

    def test_dispatch_table_parity_holds(self) -> None:
        a = agent.Agent(workspace=".")
        handlers = a._nlp_tool_handlers()
        missing = NLP_TOOL_NAMES - set(handlers)
        extra = set(handlers) - NLP_TOOL_NAMES
        assert not missing and not extra

    def test_build_mode_offers_both_tools(self) -> None:
        from agent_core.modes import filter_tool_schemas
        names = {
            s["function"]["name"]
            for s in filter_tool_schemas(list(agent.NLP_TOOL_SCHEMAS), "build")
        }
        assert {"mcp_tools", "mcp_call"} <= names
