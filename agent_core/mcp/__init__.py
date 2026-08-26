"""MCP (Model Context Protocol) consumer for Agent1.

Lets the agent and its web dashboard USE external MCP servers - browse
their tools/resources, invoke tools with schema-validated arguments -
without exposing configuration control to anything but the human user:

* ``mcp.json`` (repo root, gitignored) is written only via the REPL
  ``mcp add/remove`` command;
* the web UI can call tools on ALREADY-CONNECTED servers over a
  localhost-only POST endpoint that never touches config files;
* the NLP tool loop sees servers only when ``expose_to_llm`` is true
  (default false) in the server's config entry.

Modules: :mod:`.jsonrpc` (framing), :mod:`.transports` (stdio + HTTP),
:mod:`.config` (validated config), :mod:`.client` (protocol),
:mod:`.manager` (lifecycle, shared singleton).
"""
from .config import (
    DEFAULT_MCP_JSON,
    McpConfigError,
    McpServerConfig,
    load_mcp_servers,
    resolve_value,
    save_mcp_servers,
)
from .jsonrpc import JsonRpcError
from .manager import McpManager, McpManagerError, get_manager
from .transports import HttpTransport, StdioTransport, TransportError

__all__ = [
    "DEFAULT_MCP_JSON",
    "HttpTransport",
    "JsonRpcError",
    "McpConfigError",
    "McpManager",
    "McpManagerError",
    "McpServerConfig",
    "StdioTransport",
    "TransportError",
    "get_manager",
    "load_mcp_servers",
    "resolve_value",
    "save_mcp_servers",
]
