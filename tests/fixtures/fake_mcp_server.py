#!/usr/bin/env python3
"""Hermetic fake MCP server for tests (JSON-RPC over stdio, newline-framed).

Tools:
- echo(text)            -> returns the text verbatim
- add(a, b)             -> integer sum
- slow(seconds)         -> sleeps before answering (timeout testing)
Resources:
- fake://note           -> a tiny text document
Exits on stdin EOF or the notifications/shutdown message.
"""
import json
import sys
import time

TOOLS = [
    {
        "name": "echo",
        "description": "Return the input text unchanged.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "add",
        "description": "Add two integers.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
    {
        "name": "slow",
        "description": "Sleep then answer (for timeout tests).",
        "inputSchema": {
            "type": "object",
            "properties": {"seconds": {"type": "number"}},
            "required": ["seconds"],
        },
    },
]

RESOURCES = [
    {"uri": "fake://note", "name": "note", "mimeType": "text/plain"},
]


def reply(req_id, result):
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def handle(msg):
    method = msg.get("method", "")
    params = msg.get("params") or {}
    if method.startswith("notifications/"):
        if method == "notifications/shutdown":
            return None  # sentinel: quit after replying nothing
        return []  # no response to notifications
    rid = msg.get("id")
    if method == "initialize":
        return [reply(rid, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}, "resources": {}},
            "serverInfo": {"name": "fake-mcp", "version": "0.1"},
        })]
    if method == "tools/list":
        return [reply(rid, {"tools": TOOLS})]
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "echo":
            out = [{"type": "text", "text": args.get("text", "")}]
        elif name == "add":
            out = [{"type": "text", "text": str(int(args["a"]) + int(args["b"]))}]
        elif name == "slow":
            time.sleep(float(args.get("seconds", 0)))
            out = [{"type": "text", "text": "finally awake"}]
        else:
            return [reply(rid, {"content": [], "isError": True,
                                "structuredContent": {"error": f"unknown tool {name}"}})]
        return [reply(rid, {"content": out, "isError": False})]
    if method == "resources/list":
        return [reply(rid, {"resources": RESOURCES})]
    if method == "resources/read":
        return [reply(rid, {"contents": [
            {"uri": params.get("uri", ""), "mimeType": "text/plain",
             "text": "hello from fake resource"},
        ]})]
    return [{"jsonrpc": "2.0", "id": rid,
             "error": {"code": -32601, "message": f"unknown method {method}"}}]


def main() -> None:
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for out in handle(msg):
            sys.stdout.write(json.dumps(out, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        if msg.get("method") == "notifications/shutdown":
            break


if __name__ == "__main__":
    main()
