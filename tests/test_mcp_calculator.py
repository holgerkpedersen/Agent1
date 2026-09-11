"""Tests for the MCP calculator server."""
from __future__ import annotations

import json
import sys
from io import StringIO
from unittest.mock import patch

import pytest


def _call_tool(name: str, arguments: dict) -> dict:
    """Send a tools/call message to the calculator and return the response."""
    from mcp_servers.calculator import _handle

    msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    resp = _handle(msg)
    assert resp is not None
    return resp


def _text(resp: dict) -> str:
    """Extract the text content from a tools/call response."""
    return resp["result"]["content"][0]["text"]


def _is_error(resp: dict) -> bool:
    return resp["result"].get("isError", False)


# -----------------------------------------------------------------------
# Basic arithmetic
# -----------------------------------------------------------------------

class TestAdd:
    def test_basic(self):
        assert _text(_call_tool("add", {"a": 2, "b": 3})) == "5"

    def test_negative(self):
        assert _text(_call_tool("add", {"a": -1, "b": 1})) == "0"

    def test_floats(self):
        assert _text(_call_tool("add", {"a": 0.1, "b": 0.2})).startswith("0.3")


class TestSubtract:
    def test_basic(self):
        assert _text(_call_tool("subtract", {"a": 10, "b": 4})) == "6"

    def test_negative_result(self):
        assert _text(_call_tool("subtract", {"a": 3, "b": 7})) == "-4"


class TestMultiply:
    def test_basic(self):
        assert _text(_call_tool("multiply", {"a": 3, "b": 4})) == "12"

    def test_zero(self):
        assert _text(_call_tool("multiply", {"a": 99, "b": 0})) == "0"


class TestDivide:
    def test_basic(self):
        assert _text(_call_tool("divide", {"a": 10, "b": 2})) == "5"

    def test_non_integer_result(self):
        assert _text(_call_tool("divide", {"a": 1, "b": 3})).startswith("0.333")

    def test_division_by_zero(self):
        resp = _call_tool("divide", {"a": 1, "b": 0})
        assert _is_error(resp)
        assert "Division by zero" in _text(resp)


# -----------------------------------------------------------------------
# Expression calculator
# -----------------------------------------------------------------------

class TestCalculate:
    def test_simple_addition(self):
        assert _text(_call_tool("calculate", {"expression": "2 + 3"})) == "5"

    def test_operator_precedence(self):
        assert _text(_call_tool("calculate", {"expression": "2 + 3 * 4"})) == "14"

    def test_parentheses(self):
        assert _text(_call_tool("calculate", {"expression": "(2 + 3) * 4"})) == "20"

    def test_power(self):
        assert _text(_call_tool("calculate", {"expression": "2 ** 10"})) == "1024"

    def test_floor_div(self):
        assert _text(_call_tool("calculate", {"expression": "7 // 2"})) == "3"

    def test_modulo(self):
        assert _text(_call_tool("calculate", {"expression": "10 % 3"})) == "1"

    def test_unary_negative(self):
        assert _text(_call_tool("calculate", {"expression": "-5 + 3"})) == "-2"

    def test_nested_expression(self):
        assert _text(_call_tool("calculate", {"expression": "(1 + 2) * (3 + 4)"})) == "21"

    def test_invalid_expression(self):
        resp = _call_tool("calculate", {"expression": "import os"})
        assert _is_error(resp)

    def test_empty_expression(self):
        resp = _call_tool("calculate", {"expression": ""})
        assert _is_error(resp)


# -----------------------------------------------------------------------
# Error handling
# -----------------------------------------------------------------------

class TestErrors:
    def test_unknown_tool(self):
        resp = _call_tool("modulus", {"a": 1, "b": 2})
        assert _is_error(resp)

    def test_missing_required_arg(self):
        resp = _call_tool("add", {"a": 1})
        assert _is_error(resp)


# -----------------------------------------------------------------------
# MCP protocol
# -----------------------------------------------------------------------

class TestMcpProtocol:
    def test_initialize(self):
        from mcp_servers.calculator import _handle
        resp = _handle({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0.1"},
            },
        })
        assert resp is not None
        result = resp["result"]
        assert result["protocolVersion"] == "2024-11-05"
        assert result["serverInfo"]["name"] == "calculator"
        assert "tools" in result["capabilities"]

    def test_tools_list(self):
        from mcp_servers.calculator import _handle
        resp = _handle({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        })
        assert resp is not None
        tools = resp["result"]["tools"]
        names = {t["name"] for t in tools}
        assert names == {"add", "subtract", "multiply", "divide", "calculate"}

    def test_tool_schemas_have_required_fields(self):
        from mcp_servers.calculator import TOOLS
        for tool in TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            schema = tool["inputSchema"]
            assert schema.get("type") == "object"
            assert "properties" in schema
            assert "required" in schema

    def test_notifications_no_response(self):
        from mcp_servers.calculator import _handle
        resp = _handle({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })
        assert resp is None

    def test_unknown_method_returns_error(self):
        from mcp_servers.calculator import _handle
        resp = _handle({
            "jsonrpc": "2.0",
            "id": 99,
            "method": "bogus/method",
            "params": {},
        })
        assert resp is not None
        assert resp["error"]["code"] == -32601


# -----------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------

class TestMain:
    def test_main_reads_and_responds(self, capsys):
        from mcp_servers.calculator import main

        request = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "add", "arguments": {"a": 10, "b": 20}},
        })
        with patch("sys.stdin", StringIO(request + "\n")):
            main()

        out = capsys.readouterr().out.strip()
        resp = json.loads(out)
        assert _text({"result": resp["result"]}) == "30"
