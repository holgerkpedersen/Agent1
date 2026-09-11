"""Minimal MCP calculator server over stdio.

Exposes five arithmetic tools (add, subtract, multiply, divide, calculate)
compatible with Agent1's MCP client (StdioTransport + McpClient).

Usage:
    python mcp_servers/calculator.py

Zero external dependencies — stdlib only.
"""
from __future__ import annotations

import ast
import operator
import sys
import json
import json as _json
from typing import Any

# ---------------------------------------------------------------------------
# JSON-RPC helpers (inlined to keep this zero-dependency)
# ---------------------------------------------------------------------------

REQUEST_ID_COUNTER = 0

def _next_id() -> int:
    global REQUEST_ID_COUNTER
    REQUEST_ID_COUNTER += 1
    return REQUEST_ID_COUNTER

def _parse_message(raw: str) -> dict[str, Any]:
    msg = json.loads(raw)
    if not isinstance(msg, dict):
        raise ValueError("not a JSON object")
    return msg

def _make_response(req_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}

def _make_error(req_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}

def _write(msg: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(msg, separators=(",", ":")) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# MCP protocol constants
# ---------------------------------------------------------------------------

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INTERNAL_ERROR = -32603

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "calculator", "version": "1.0"}
SERVER_CAPABILITIES: dict[str, Any] = {"tools": {}}


# ---------------------------------------------------------------------------
# Safe arithmetic evaluator for the "calculate" tool
# ---------------------------------------------------------------------------

_SAFE_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """Evaluate a limited subset of Python expressions (arithmetic only)."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return float(_SAFE_OPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return float(_SAFE_OPS[type(node.op)](_safe_eval(node.operand)))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def safe_calculate(expression: str) -> float:
    """Parse and evaluate a math expression safely (no exec/eval)."""
    tree = ast.parse(expression.strip(), mode="eval")
    return _safe_eval(tree)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "add",
        "description": "Add two numbers.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First operand"},
                "b": {"type": "number", "description": "Second operand"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "subtract",
        "description": "Subtract b from a.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "Minuend"},
                "b": {"type": "number", "description": "Subtrahend"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "multiply",
        "description": "Multiply two numbers.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First factor"},
                "b": {"type": "number", "description": "Second factor"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "divide",
        "description": "Divide a by b.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "Dividend"},
                "b": {"type": "number", "description": "Divisor (must not be zero)"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "calculate",
        "description": (
            "Evaluate a math expression string. "
            "Supports +, -, *, /, //, %, ** and parentheses."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate, e.g. '2 + 3 * 4'",
                },
            },
            "required": ["expression"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def _execute_tool(name: str, args: dict[str, Any]) -> str:
    """Run a tool and return the result as a string."""
    if name == "add":
        return str(args["a"] + args["b"])
    if name == "subtract":
        return str(args["a"] - args["b"])
    if name == "multiply":
        return str(args["a"] * args["b"])
    if name == "divide":
        if args["b"] == 0:
            raise ValueError("Division by zero")
        result = args["a"] / args["b"]
        # Return clean number when possible
        return str(int(result)) if result == int(result) else str(result)
    if name == "calculate":
        result = safe_calculate(args["expression"])
        return str(int(result)) if result == int(result) else str(result)
    raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# Message routing
# ---------------------------------------------------------------------------

def _handle(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Route an incoming JSON-RPC message; return a response dict (or None for notifications)."""
    req_id = msg.get("id")
    method = msg.get("method")
    params = msg.get("params", {})

    # Notifications have no id — no response expected
    if method == "notifications/initialized":
        return None

    # ---- initialize ----
    if method == "initialize":
        return _make_response(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": SERVER_CAPABILITIES,
            "serverInfo": SERVER_INFO,
        })

    # ---- tools/list ----
    if method == "tools/list":
        return _make_response(req_id, {"tools": TOOLS})

    # ---- tools/call ----
    if method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        if not isinstance(tool_name, str):
            return _make_error(req_id, -32602, "missing tool name")
        try:
            text = _execute_tool(tool_name, arguments)
            return _make_response(req_id, {
                "content": [{"type": "text", "text": text}],
                "isError": False,
            })
        except (KeyError, ValueError, TypeError, ZeroDivisionError, SyntaxError) as exc:
            return _make_response(req_id, {
                "content": [{"type": "text", "text": f"Error: {exc}"}],
                "isError": True,
            })

    # ---- unknown method ----
    if req_id is not None:
        return _make_error(req_id, METHOD_NOT_FOUND, f"Unknown method: {method}")

    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Read newline-delimited JSON-RPC from stdin, write responses to stdout."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = _parse_message(line)
        except (json.JSONDecodeError, ValueError) as exc:
            _write(_make_error(None, PARSE_ERROR, f"Parse error: {exc}"))
            continue
        response = _handle(msg)
        if response is not None:
            _write(response)


if __name__ == "__main__":
    main()
