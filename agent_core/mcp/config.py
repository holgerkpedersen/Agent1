"""Configuration for MCP servers: ``mcp.json`` at the repo root.

Follows the ``model.json`` conventions (tolerant load, JSON on disk) but
writes atomically like the chat-history sidecars.  Secret material is NEVER
stored in plaintext: header/env values may reference the OS keyring via
``secret:<name>`` or an environment variable via ``${VAR}``; resolution
happens at connect time and fails closed.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_core.security import secrets

#: Default config location: repo root (derived from this module's location,
#: same pattern as MODEL_JSON_PATH in agent_core.constants).
DEFAULT_MCP_JSON = Path(__file__).resolve().parents[2] / "mcp.json"

_TRANSPORTS = ("stdio", "http")
_MAX_TIMEOUT_S = 600
_ENV_REF = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*\}$")
_SECRET_REF = re.compile(r"^secret:[A-Za-z0-9_.-]+$")


class McpConfigError(ValueError):
    """Raised when mcp.json is structurally invalid; message names the server."""


@dataclass
class McpServerConfig:
    """One validated server entry from mcp.json."""

    name: str
    transport: str                      # "stdio" | "http"
    command: list[str] = field(default_factory=list)   # stdio argv (no shell)
    url: str = ""                       # http endpoint
    env: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    expose_to_llm: bool = False         # opt-IN bridge into the NLP tool loop
    timeout_s: int = 30

    def resolved_env(self) -> dict[str, str]:
        """Env with secret:/env refs resolved; unresolved refs fail closed."""
        return {k: resolve_value(v, f"{self.name}.env.{k}") for k, v in self.env.items()}

    def resolved_headers(self) -> dict[str, str]:
        return {k: resolve_value(v, f"{self.name}.headers.{k}")
                for k, v in self.headers.items()}


def resolve_value(raw: str, where: str) -> str:
    """Resolve a ``secret:``/``${VAR}`` reference; plain values pass through.

    Fail-closed policy: a dangling reference raises instead of silently
    sending an empty credential.
    """
    if _SECRET_REF.match(raw):
        name = raw[len("secret:"):]
        value = secrets.get_secret(name)
        if not value:
            raise McpConfigError(
                f"{where}: no secret named '{name}' in the OS keyring "
                f"(store one via the secrets store first)"
            )
        return value
    if _ENV_REF.match(raw):
        var = raw[2:-1]
        env_value: str | None = os.environ.get(var)
        if not env_value:
            raise McpConfigError(f"{where}: environment variable {var} is not set")
        return env_value
    return raw


def validate_entry(name: str, raw: Any) -> McpServerConfig:
    """Validate one mcp.json entry; raise :class:`McpConfigError` with context."""
    if not isinstance(raw, dict):
        raise McpConfigError(f"{name}: entry must be a JSON object")
    transport = raw.get("transport")
    if transport not in _TRANSPORTS:
        raise McpConfigError(f"{name}: transport must be one of {_TRANSPORTS}")
    cfg = McpServerConfig(name=str(name), transport=str(transport))
    if transport == "stdio":
        cmd = raw.get("command")
        if (
            not isinstance(cmd, list) or not cmd
            or not all(isinstance(p, str) and p.strip() for p in cmd)
        ):
            raise McpConfigError(
                f"{name}: stdio transport needs a non-empty 'command' string array"
            )
        cfg.command = list(cmd)
    else:
        url = raw.get("url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise McpConfigError(f"{name}: http transport needs an http(s) 'url'")
        cfg.url = url
    for attr in ("env", "headers"):
        val = raw.get(attr, {})
        if not isinstance(val, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in val.items()
        ):
            raise McpConfigError(f"{name}: '{attr}' must map strings to strings")
        setattr(cfg, attr, dict(val))
    cfg.expose_to_llm = bool(raw.get("expose_to_llm", False))
    try:
        cfg.timeout_s = max(1, min(int(raw.get("timeout_s", 30)), _MAX_TIMEOUT_S))
    except (TypeError, ValueError) as exc:
        raise McpConfigError(f"{name}: timeout_s must be a number") from exc
    return cfg


def load_mcp_servers(path: Path = DEFAULT_MCP_JSON) -> dict[str, McpServerConfig]:
    """Load + validate all entries; tolerant of a missing file (empty config)."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, McpServerConfig] = {}
    for name, raw in data.items():
        out[str(name)] = validate_entry(str(name), raw)
    return out


def save_mcp_servers(configs: dict[str, McpServerConfig],
                     path: Path = DEFAULT_MCP_JSON) -> None:
    """Write atomically (tmp + os.replace) so a crash cannot truncate config."""
    payload = {
        cfg.name: {
            "transport": cfg.transport,
            **({"command": cfg.command} if cfg.transport == "stdio" else {}),
            **({"url": cfg.url} if cfg.transport == "http" else {}),
            **({"env": cfg.env} if cfg.env else {}),
            **({"headers": cfg.headers} if cfg.headers else {}),
            "expose_to_llm": cfg.expose_to_llm,
            "timeout_s": cfg.timeout_s,
        }
        for cfg in configs.values()
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
