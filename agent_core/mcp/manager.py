"""Multi-server lifecycle manager for MCP connections.

One process-wide :class:`McpManager` owns every live connection.  All state
transitions are lock-guarded; the web UI and the NLP bridge share this
instance, so a server connected in the REPL is instantly usable from the
dashboard page (and vice versa).
"""
from __future__ import annotations

import threading
from typing import Any

from .client import McpClient
from .config import McpConfigError, McpServerConfig, load_mcp_servers
from .transports import BaseTransport, HttpTransport, StdioTransport, TransportError

STATE_CONNECTED = "connected"
STATE_ERROR = "error"
STATE_STOPPED = "stopped"


class McpManagerError(RuntimeError):
    """Raised for unknown servers / invalid operations at manager level."""


class McpManager:
    """Owns connections to configured MCP servers."""

    def __init__(self, config_path: Any = None) -> None:
        self._config_path = config_path  # None -> default mcp.json
        self._clients: dict[str, McpClient] = {}
        self._errors: dict[str, str] = {}
        self._lock = threading.RLock()

    # -- config ------------------------------------------------------------

    def configured(self) -> dict[str, McpServerConfig]:
        if self._config_path is None:
            return load_mcp_servers()
        return load_mcp_servers(self._config_path)

    def save_config(self, configs: dict[str, McpServerConfig]) -> None:
        from .config import save_mcp_servers
        if self._config_path is None:
            save_mcp_servers(configs)
        else:
            save_mcp_servers(configs, self._config_path)

    # -- lifecycle ---------------------------------------------------------

    def connect(self, name: str) -> dict[str, Any]:
        """Connect + handshake; raises McpManagerError/McpConfigError on failure."""
        cfg = self.configured().get(name)
        if cfg is None:
            raise McpManagerError(
                f"no MCP server named '{name}' in mcp.json "
                f"(configured: {', '.join(self.configured()) or 'none'})"
            )
        with self._lock:
            self.disconnect(name, _silent=True)
            try:
                transport: BaseTransport
                if cfg.transport == "stdio":
                    transport = StdioTransport(cfg.command, env=cfg.resolved_env())
                else:
                    transport = HttpTransport(cfg.url, headers=cfg.resolved_headers())
                client = McpClient(transport, timeout_s=cfg.timeout_s)
                client.initialize()
                client.refresh_tools()
                self._clients[name] = client
                self._errors.pop(name, None)
            except (TransportError, McpConfigError) as exc:
                self._errors[name] = str(exc)
                raise
            except Exception as exc:  # protocol-level surprise: contain it
                self._errors[name] = f"{type(exc).__name__}: {exc}"
                raise McpManagerError(str(exc)) from exc
        return self.status_one(name)

    def disconnect(self, name: str, _silent: bool = False) -> None:
        with self._lock:
            client = self._clients.pop(name, None)
        if client is not None:
            client.close()
        elif not _silent and name not in self.configured():
            raise McpManagerError(f"no MCP server named '{name}'")

    def disconnect_all(self) -> None:
        for name in list(self._clients):
            self.disconnect(name, _silent=True)

    # -- queries -----------------------------------------------------------

    def is_connected(self, name: str) -> bool:
        with self._lock:
            return name in self._clients

    def status_one(self, name: str) -> dict[str, Any]:
        cfg = self.configured().get(name)
        if cfg is None:
            raise McpManagerError(f"no MCP server named '{name}'")
        with self._lock:
            client = self._clients.get(name)
            entry: dict[str, Any] = {
                "name": name,
                "transport": cfg.transport,
                "expose_to_llm": cfg.expose_to_llm,
                "timeout_s": cfg.timeout_s,
                "state": STATE_STOPPED,
                "detail": "",
                "tool_count": 0,
                "server_info": {},
            }
            if name in self._errors:
                entry["state"] = STATE_ERROR
                entry["detail"] = self._errors[name]
            elif client is not None:
                entry["state"] = (
                    STATE_CONNECTED if client._transport.is_alive()
                    else STATE_STOPPED
                )
                entry["tool_count"] = len(client.tool_names())
                info = client.server_info.get("serverInfo")
                entry["server_info"] = info if isinstance(info, dict) else {}
            return entry

    def status(self) -> list[dict[str, Any]]:
        names = sorted(set(self.configured()) | set(self._clients))
        out: list[dict[str, Any]] = []
        for name in names:
            try:
                out.append(self.status_one(name))
            except McpManagerError:
                continue
        return out

    def tools(self, name: str) -> list[dict[str, Any]]:
        with self._lock:
            client = self._clients.get(name)
        if client is None:
            raise McpManagerError(f"MCP server '{name}' is not connected")
        return client.refresh_tools()

    # -- operations --------------------------------------------------------

    def call_tool(self, server: str, tool: str,
                  arguments: dict[str, Any] | None,
                  require_exposed: bool = False) -> str:
        """Invoke a tool on a connected server.

        ``require_exposed=True`` (the LLM bridge path) re-reads the config
        entry under the lock so an 'mcp expose <name> off' issued between
        the caller's gate check and this call still blocks - closing the
        TOCTOU window instead of trusting a stale catalog snapshot.
        """
        with self._lock:
            client = self._clients.get(server)
            if require_exposed:
                cfg = self.configured().get(server)
                if cfg is None or not cfg.expose_to_llm:
                    raise McpManagerError(
                        f"MCP server '{server}' is not exposed to the LLM "
                        f"(user controls this with 'mcp expose {server} on')"
                    )
        if client is None:
            raise McpManagerError(
                f"MCP server '{server}' is not connected - run 'mcp connect {server}'"
            )
        return client.call_tool(tool, arguments)

    def resources_list(self, server: str) -> list[dict[str, Any]]:
        with self._lock:
            client = self._clients.get(server)
        if client is None:
            raise McpManagerError(f"MCP server '{server}' is not connected")
        return client.resources_list()

    def read_resource(self, server: str, uri: str) -> str:
        with self._lock:
            client = self._clients.get(server)
        if client is None:
            raise McpManagerError(f"MCP server '{server}' is not connected")
        return client.resources_read(uri)

    def llm_catalog(self) -> dict[str, list[str]]:
        """Servers opted into LLM exposure with their current tool names."""
        out: dict[str, list[str]] = {}
        for name, cfg in self.configured().items():
            if not cfg.expose_to_llm:
                continue
            with self._lock:
                client = self._clients.get(name)
            if client is not None:
                out[name] = client.tool_names()
        return out


_manager: McpManager | None = None
_manager_lock = threading.Lock()


def get_manager() -> McpManager:
    """Process-wide shared manager (REPL, dashboard and NLP bridge share it)."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = McpManager()
        return _manager
