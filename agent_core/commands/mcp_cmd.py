"""``mcp`` REPL command - manage and consume MCP servers.

Subcommands (all output human-readable text, errors printed never raised):

    mcp                                  status of every configured server
    mcp add <name> stdio <command...>    register a local server (argv list)
    mcp add <name> http <url> [--header K=V]... [--env K=V]... [--timeout N]
    mcp remove <name>                    delete a config entry (+ disconnect)
    mcp connect <name>|all              spawn/handshake one or every server
    mcp disconnect <name>|all
    mcp tools [server]                   list tools + input schemas
    mcp call <server> <tool> [json]      invoke a tool ('{}' / omitted args)
    mcp resources [server]               list resources
    mcp read <server> <uri>              read one resource
    mcp expose <name> on|off             opt this server into the LLM bridge

Safety notes: ``add``/``remove``/``expose`` are the ONLY ways configuration
changes (human REPL actions); tool arguments must be valid JSON; results are
size-capped by the client layer.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from .base import Command
if TYPE_CHECKING:
    from agent import Agent
from agent_core.mcp.config import (
    McpConfigError,
    McpServerConfig,
    validate_entry,
)
from agent_core.mcp.manager import McpManagerError, get_manager


class MCPCommand(Command):
    """REPL front-end over the shared :class:`~agent_core.mcp.McpManager`."""

    def __init__(self) -> None:
        self._manager = get_manager()

    @property
    def name(self) -> str:
        return "mcp"

    @property
    def help_text(self) -> str:
        return (
            "mcp [add|remove|connect|disconnect|tools|call|resources|read"
            "|expose] - Manage and consume MCP servers"
        )

    # ------------------------------------------------------------------

    async def execute(self, args: list[str], agent: Agent) -> bool:
        sub = args[0].lower() if args else ""
        try:
            if not sub or sub in ("status", "list"):
                self._status()
            elif sub == "add":
                self._add(args[1:])
            elif sub == "remove":
                self._remove(args[1:])
            elif sub == "connect":
                self._connect(args[1:])
            elif sub == "disconnect":
                self._disconnect(args[1:])
            elif sub == "tools":
                self._tools(args[1:])
            elif sub == "call":
                await self._call(args[1:])
            elif sub == "resources":
                self._resources(args[1:])
            elif sub == "read":
                self._read(args[1:])
            elif sub == "expose":
                self._expose(args[1:])
            else:
                print(f"  Unknown mcp subcommand '{sub}'. See 'help mcp'.")
        except (McpManagerError, McpConfigError, ValueError) as exc:
            print(f"  [mcp] error: {exc}")
        return True

    # -- subcommands -------------------------------------------------------

    def _status(self) -> None:
        rows = self._manager.status()
        if not rows:
            print("  No MCP servers configured. Try: mcp add <name> http <url>")
            return
        print(f"  {len(rows)} MCP server(s):")
        for r in rows:
            mark = {"connected": "+", "error": "x", "stopped": "-"}.get(
                r["state"], "?")
            extra = f"  [{r['detail']}]" if r["detail"] else ""
            llm = "  llm-exposed" if r["expose_to_llm"] else ""
            print(f"   [{mark}] {r['name']} ({r['transport']}) "
                  f"state={r['state']} tools={r['tool_count']}{llm}{extra}")

    def _add(self, args: list[str]) -> None:
        if len(args) < 3:
            print("  usage: mcp add <name> stdio <command...>"
                  " | mcp add <name> http <url> [--header K=V]"
                  " [--env K=V] [--timeout N]")
            return
        name, transport = args[0], args[1].lower()
        raw: dict[str, Any] = {}
        if transport == "stdio":
            raw = {"transport": "stdio", "command": args[2:]}
        else:
            url_parts: list[str] = []
            headers: dict[str, str] = {}
            env: dict[str, str] = {}
            timeout = 30
            i = 2
            while i < len(args):
                tok = args[i]
                if tok == "--header" and i + 1 < len(args):
                    key, _, val = args[i + 1].partition("=")
                    headers[key.strip()] = val.strip()
                    i += 2
                elif tok == "--env" and i + 1 < len(args):
                    key, _, val = args[i + 1].partition("=")
                    env[key.strip()] = val.strip()
                    i += 2
                elif tok == "--timeout" and i + 1 < len(args):
                    timeout = int(args[i + 1])
                    i += 2
                else:
                    url_parts.append(tok)
                    i += 1
            if len(url_parts) != 1:
                print("  usage: mcp add <name> http <url> [flags]")
                return
            raw = {"transport": "http", "url": url_parts[0],
                   "headers": headers, "env": env, "timeout_s": timeout}
        cfg = validate_entry(name, raw)
        configs = self._manager.configured()
        existed = name in configs
        configs[name] = cfg
        self._manager.save_config(configs)
        verb = "updated" if existed else "added"
        print(f"  [mcp] {verb} '{name}' ({cfg.transport}). "
              f"Connect it: mcp connect {name}")

    def _remove(self, args: list[str]) -> None:
        if not args:
            print("  usage: mcp remove <name>")
            return
        name = args[0]
        configs = self._manager.configured()
        if name not in configs:
            print(f"  [mcp] no server named '{name}'")
            return
        del configs[name]
        self._manager.save_config(configs)
        self._manager.disconnect(name, _silent=True)
        print(f"  [mcp] removed '{name}'")

    def _connect(self, args: list[str]) -> None:
        targets = (sorted(self._manager.configured()) if args and args[0] == "all"
                   else args[:1])
        if not targets:
            print("  usage: mcp connect <name>|all")
            return
        for name in targets:
            try:
                st = self._manager.connect(name)
                print(f"  [mcp] {name}: {st['state']} "
                      f"({st['tool_count']} tools)")
            except Exception as exc:
                print(f"  [mcp] {name}: FAILED - {exc}")

    def _disconnect(self, args: list[str]) -> None:
        if not args or (args[0] != "all" and len(args) != 1):
            print("  usage: mcp disconnect <name>|all")
            return
        names = (sorted(self._manager.configured()) if args[0] == "all"
                 else args[:1])
        for name in names:
            self._manager.disconnect(name, _silent=True)
            print(f"  [mcp] {name}: disconnected")

    def _tools(self, args: list[str]) -> None:
        servers = args[:1] or sorted(
            n for n, s in
            ((r["name"], r) for r in self._manager.status())
            if s["state"] == "connected"
        )
        for server in servers:
            try:
                tools = self._manager.tools(server)
            except Exception as exc:
                print(f"  [mcp] {server}: {exc}")
                continue
            print(f"  {server}: {len(tools)} tool(s)")
            for t in tools:
                schema = json.dumps(t.get("inputSchema", {}),
                                    ensure_ascii=False)
                desc = t["description"][:70]
                print(f"   - {t['name']}: {desc}")
                print(f"       schema: {schema[:120]}")

    async def _call(self, args: list[str]) -> None:
        if len(args) < 2:
            print("  usage: mcp call <server> <tool> ['{json args}']")
            return
        server, tool = args[0], args[1]
        arguments: dict[str, Any] = {}
        if len(args) > 2:
            arguments = json.loads(" ".join(args[2:]))
            if not isinstance(arguments, dict):
                raise ValueError("arguments JSON must be an object")
        result = self._manager.call_tool(server, tool, arguments)
        print(f"  [{server}.{tool}]")
        print(f"  {result}")

    def _resources(self, args: list[str]) -> None:
        if not args:
            print("  usage: mcp resources <server>")
            return
        res = self._manager.resources_list(args[0])
        print(f"  {args[0]}: {len(res)} resource(s)")
        for r in res:
            print(f"   - {r.get('uri')}  {r.get('name', '')}"
                  f"  ({r.get('mimeType', '?')})")

    def _read(self, args: list[str]) -> None:
        if len(args) < 2:
            print("  usage: mcp read <server> <uri>")
            return
        print(f"  {self._manager.read_resource(args[0], args[1])}")

    def _expose(self, args: list[str]) -> None:
        if not args or args[0] in ("-h", "--help"):
            print("  usage: mcp expose <name> on|off"
                  "  (opt server tools into the LLM bridge; default off)")
            return
        name = args[0]
        state = args[1].lower() if len(args) > 1 else ""
        if state not in ("on", "off"):
            print("  usage: mcp expose <name> on|off")
            return
        configs = self._manager.configured()
        if name not in configs:
            print(f"  [mcp] no server named '{name}'")
            return
        cfg: McpServerConfig = configs[name]
        cfg.expose_to_llm = state == "on"
        self._manager.save_config(configs)
        scope = ("the LLM can now call its tools via mcp_call"
                 if cfg.expose_to_llm else "hidden from the LLM again")
        print(f"  [mcp] {name}: LLM exposure {state} ({scope})")
