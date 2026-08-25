"""Safe shell command allow-list and validation logic.

This module provides an explicit allow-list of permitted shell commands to replace
the fragile blacklist approach previously used in tool execution.
"""

from __future__ import annotations

import logging
from typing import Final, Set

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Allow-list of safe shell commands (binary names only).
SAFE_COMMANDS: Final[Set[str]] = {
    "python",
    "python3",
    "git",
    "ls",
    "dir",
    "cat",
    "type",
    "head",
    "tail",
    "echo",
    "pwd",
}

#: Structural shell patterns that must never appear in a command string —
#: pipes, redirection, chaining, command substitution.  Even with
#: ``shell=False`` these are rejected up front so no caller can smuggle a
#: chained command past the binary allow-list (plan OPS item 2).
_UNSAFE_SHELL_TOKENS: Final[tuple[tuple[str, str], ...]] = (
    ("&&", "command chaining (&&)"),
    ("||", "command chaining (||)"),
    (";", "command separator (;)"),
    ("|", "pipe (|)"),
    (">", "redirection (>)"),
    ("<", "redirection (<)"),
    ("`", "command substitution (`)"),
    ("$(", "command substitution ($()"),
    ("\n", "embedded newline"),
    ("\r", "embedded newline"),
)

# ---------------------------------------------------------------------------
# Dynamic Allowlist State & Logger
# ---------------------------------------------------------------------------

_dynamic_allowlist: Set[str] = set()
_logger = logging.getLogger(__name__)


def _normalize(binary_name: str) -> str:
    """Normalize a binary name for allow-list comparison."""
    normalized = binary_name.strip().lower()
    for suffix in (".exe", ".bat", ".cmd"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_unsafe_shell_pattern(cmd_str: str) -> str | None:
    """Return a description of the first unsafe shell pattern in *cmd_str*.

    Rejects shell metacharacters, pipes, redirection operators, command
    chaining (``&&`` / ``||`` / ``;``), and command substitution.  Returns
    ``None`` when the command string is structurally safe — the binary
    allow-list then decides whether the command may run.
    """
    if not cmd_str:
        return None
    for token, description in _UNSAFE_SHELL_TOKENS:
        if token in cmd_str:
            return description
    return None


def is_command_allowed(binary_name: str) -> bool:
    """Check if a shell command binary is in the allow-list.

    The check is case-insensitive and strips common Windows executable suffixes
    (``.exe``, ``.bat``, ``.cmd``) before comparison so that commands like
    ``python.exe`` are accepted alongside ``python``.

    Parameters
    ----------
    binary_name : str
        The raw command string as provided by the caller (e.g. ``"Git"``,
        ``"cat.exe"``).

    Returns
    -------
    bool
        ``True`` when *binary_name* corresponds to an allowed command,
        ``False`` otherwise.
    """
    normalized = _normalize(binary_name)
    allowed = normalized in SAFE_COMMANDS or normalized in _dynamic_allowlist

    source: str = "static" if normalized in SAFE_COMMANDS else ("dynamic" if allowed else "none")
    _logger.info(
        "Command allowlist check",
        extra={
            "event": "command_allowlist_check",
            "binary": binary_name,
            "normalized": normalized,
            "allowed": allowed,
            "source": source,
        },
    )

    return allowed


def register_command(binary_name: str) -> None:
    """Dynamically add a command to the allow-list.

    Parameters
    ----------
    binary_name : str
        The command name to allow (suffixes will be stripped during normalization).
    """
    normalized = _normalize(binary_name)
    if normalized not in SAFE_COMMANDS and normalized not in _dynamic_allowlist:
        _dynamic_allowlist.add(normalized)
        _logger.info(
            "Command registered",
            extra={
                "event": "command_allowlist_register",
                "binary": binary_name,
                "normalized": normalized,
            },
        )


def unregister_command(binary_name: str) -> bool:
    """Dynamically remove a command from the allow-list.

    Parameters
    ----------
    binary_name : str
        The command name to disallow.

    Returns
    -------
    bool
        ``True`` if the command was removed, ``False`` otherwise.
    """
    normalized = _normalize(binary_name)
    if normalized in SAFE_COMMANDS:
        _logger.warning(
            "Cannot unregister static command",
            extra={
                "event": "command_allowlist_unregister_failed",
                "binary": binary_name,
                "reason": "static_command",
            },
        )
        return False

    if normalized in _dynamic_allowlist:
        _dynamic_allowlist.discard(normalized)
        _logger.info(
            "Command unregistered",
            extra={
                "event": "command_allowlist_unregister",
                "binary": binary_name,
                "normalized": normalized,
            },
        )
        return True

    return False


def get_allowed_commands() -> Set[str]:
    """Return the current union of static and dynamic allowed commands."""
    return SAFE_COMMANDS | _dynamic_allowlist