"""Minimal JSON-RPC 2.0 framing for the MCP consumer.

Only what the Model Context Protocol actually needs over its two wire
transports: build requests, parse responses, surface protocol errors as
typed exceptions.  Deliberately dependency-free.
"""
from __future__ import annotations

import itertools
import json
from typing import Any

#: Standard JSON-RPC 2.0 / MCP error codes surfaced to callers.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

_ids = itertools.count(1)


class JsonRpcError(Exception):
    """The remote side answered a request with a JSON-RPC error object."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"JSON-RPC {code}: {message}")


def next_id() -> int:
    """Process-wide unique request id (never reused, monotonic)."""
    return next(_ids)


def make_request(method: str, params: dict[str, Any] | None = None,
                 req_id: int | None = None) -> dict[str, Any]:
    """Build a JSON-RPC request object (serialization is the transport's job)."""
    req: dict[str, Any] = {"jsonrpc": "2.0", "id": next_id() if req_id is None else req_id}
    if method.startswith("notifications/"):
        # JSON-RPC notification: no id, no response expected.
        del req["id"]
    req["method"] = method
    if params is not None:
        req["params"] = params
    return req


def parse_message(raw: str | bytes) -> dict[str, Any]:
    """Parse one wire message and verify the minimal envelope.

    Raises :class:`JsonRpcError` with ``PARSE_ERROR`` on malformed JSON and
    ``INVALID_REQUEST`` when the envelope is structurally wrong — a hostile
    or broken server must never crash the caller with a raw KeyError.
    """
    try:
        msg = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JsonRpcError(PARSE_ERROR, f"malformed JSON-RPC payload: {exc}") from exc
    if not isinstance(msg, dict):
        raise JsonRpcError(INVALID_REQUEST, "payload is not a JSON object")
    if msg.get("jsonrpc") != "2.0":
        raise JsonRpcError(INVALID_REQUEST, "missing jsonrpc: \"2.0\" envelope")
    if "method" not in msg and "result" not in msg and "error" not in msg:
        raise JsonRpcError(INVALID_REQUEST, "message has neither method nor result/error")
    return msg


def is_response(msg: dict[str, Any]) -> bool:
    """True for a reply to one of OUR requests (has an id, no method)."""
    return "id" in msg and "method" not in msg


def result_of(msg: dict[str, Any]) -> Any:
    """Extract ``result`` from a response, raising :class:`JsonRpcError` on error."""
    if "error" in msg:
        err = msg["error"]
        if not isinstance(err, dict):
            err = {"code": INTERNAL_ERROR, "message": str(err)}
        raise JsonRpcError(
            int(err.get("code", INTERNAL_ERROR)),
            str(err.get("message", "unspecified error")),
            err.get("data"),
        )
    if "result" not in msg:
        raise JsonRpcError(INVALID_REQUEST, "response carries neither result nor error")
    return msg["result"]


def make_error_response(req_id: Any, code: int, message: str) -> dict[str, Any]:
    """Build an error response (used when WE act as a JSON-RPC endpoint)."""
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}
