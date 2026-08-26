"""MCP protocol client: initialize handshake, tools, resources.

Sits on a :class:`~agent_core.mcp.transports.BaseTransport` and adds the
MCP-level semantics.  Tool arguments are validated against the server's
declared JSON schema BEFORE the call leaves the process, and results are
size-capped so one chatty tool cannot balloon the agent's context (the
read-loop spiral of 2026-08-25 must not be re-importable through MCP).
"""
from __future__ import annotations

from typing import Any

from .transports import BaseTransport

PROTOCOL_VERSION = "2024-11-05"
CLIENT_INFO = {"name": "Agent1", "version": "1.0"}

#: Hard cap on any tool result returned to callers/LLM.
MAX_RESULT_CHARS = 20_000


class McpProtocolError(RuntimeError):
    """The server answered outside MCP expectations."""


class McpToolError(RuntimeError):
    """The tool ran but reported ``isError`` (its message is carried)."""


def truncate_result(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return (
        text[:MAX_RESULT_CHARS]
        + f"\n[truncated at {MAX_RESULT_CHARS} chars - {len(text)} total]"
    )


# ---------------------------------------------------------------------------
# Minimal JSON-Schema argument validation (fail-closed before network IO)
# ---------------------------------------------------------------------------

def validate_against_schema(schema: Any, value: Any, where: str = "args") -> str | None:
    """Return an error string when *value* violates *schema*, else None."""
    if not isinstance(schema, dict):
        return None  # no usable schema from the server: do not guess
    declared = schema.get("type")
    checks = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "null": lambda v: v is None,
    }
    if declared in checks and not checks[str(declared)](value):
        return f"{where}: expected {declared}, got {type(value).__name__}"
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        return f"{where}: value not in allowed set {enum!r}"
    if declared == "object" or isinstance(value, dict):
        props = schema.get("properties", {})
        for req in schema.get("required", []) or []:
            if isinstance(req, str) and req not in value:
                return f"{where}: missing required property '{req}'"
        if isinstance(props, dict) and isinstance(value, dict):
            for key, sub in props.items():
                if key in value:
                    err = validate_against_schema(sub, value[key], f"{where}.{key}")
                    if err:
                        return err
    if declared == "array" and isinstance(value, list):
        items = schema.get("items")
        for i, item in enumerate(value):
            err = validate_against_schema(items, item, f"{where}[{i}]")
            if err:
                return err
    return None


class McpClient:
    """One connected MCP server (initialize + tools + resources)."""

    def __init__(self, transport: BaseTransport, timeout_s: float = 30) -> None:
        self._transport = transport
        self._timeout_s = timeout_s
        self.server_info: dict[str, Any] = {}
        self._tools: list[dict[str, Any]] = []

    # -- lifecycle ---------------------------------------------------------

    def initialize(self) -> dict[str, Any]:
        """Perform the MCP initialize handshake; returns server info."""
        result = self._transport.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": CLIENT_INFO,
            },
            self._timeout_s,
        )
        if not isinstance(result, dict):
            raise McpProtocolError("initialize returned a non-object result")
        self.server_info = result
        # The spec wants a notifications/initialized announcement; it has no
        # reply by definition, so it must never wait on one.
        try:
            self._transport.notify("notifications/initialized")
        except Exception:
            pass
        return result

    def close(self) -> None:
        self._transport.close()

    # -- tools -------------------------------------------------------------

    def refresh_tools(self) -> list[dict[str, Any]]:
        raw = self._transport.request("tools/list", {}, self._timeout_s)
        tools = raw.get("tools") if isinstance(raw, dict) else None
        if not isinstance(tools, list):
            raise McpProtocolError("tools/list returned no tool array")
        self._tools = [t for t in tools if isinstance(t, dict) and t.get("name")]
        return [self._public_tool(t) for t in self._tools]

    @staticmethod
    def _public_tool(tool: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": str(tool.get("name")),
            "description": str(tool.get("description", "")),
            "inputSchema": tool.get("inputSchema", {}),
        }

    def tool_names(self) -> list[str]:
        return [str(t.get("name")) for t in self._tools]

    def call_tool(self, name: str, arguments: dict[str, Any] | None) -> str:
        """Validate args against the schema, invoke, and return capped text."""
        entry = next((t for t in self._tools if t.get("name") == name), None)
        if entry is None:
            raise McpProtocolError(
                f"unknown tool '{name}' (known: {', '.join(self.tool_names()) or 'none'})"
            )
        args = arguments if isinstance(arguments, dict) else {}
        err = validate_against_schema(entry.get("inputSchema", {}), args)
        if err:
            raise McpProtocolError(f"argument validation failed: {err}")
        result = self._transport.request(
            "tools/call", {"name": name, "arguments": args}, self._timeout_s,
        )
        if not isinstance(result, dict):
            raise McpProtocolError("tools/call returned a non-object result")
        text = self._content_text(result)
        if result.get("isError"):
            raise McpToolError(text or f"tool '{name}' reported an error")
        return truncate_result(text)

    @staticmethod
    def _content_text(result: dict[str, Any]) -> str:
        parts: list[str] = []
        content = result.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif isinstance(block, dict) and "data" in block:
                    parts.append(f"[{block.get('type', 'binary')} data omitted]")
        if not parts and "structuredContent" in result:
            import json
            parts.append(json.dumps(result["structuredContent"], ensure_ascii=False))
        return "\n".join(parts)

    # -- resources ---------------------------------------------------------

    def resources_list(self) -> list[dict[str, Any]]:
        raw = self._transport.request("resources/list", {}, self._timeout_s)
        res = raw.get("resources") if isinstance(raw, dict) else None
        return [r for r in res if isinstance(r, dict)] if isinstance(res, list) else []

    def resources_read(self, uri: str) -> str:
        result = self._transport.request(
            "resources/read", {"uri": uri}, self._timeout_s,
        )
        if not isinstance(result, dict):
            raise McpProtocolError("resources/read returned a non-object result")
        texts: list[str] = []
        contents = result.get("contents")
        if isinstance(contents, list):
            for block in contents:
                if isinstance(block, dict):
                    if isinstance(block.get("text"), str):
                        texts.append(block["text"])
                    elif "uri" in block:
                        texts.append(f"[{block.get('mimeType', 'resource')}: "
                                     f"{block['uri']}]")
        return truncate_result("\n".join(texts))
