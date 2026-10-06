"""Safe shell command allow-list and validation logic.

This module provides an explicit allow-list of permitted shell commands to replace
the fragile blacklist approach previously used in tool execution.
"""

from __future__ import annotations

import logging
import re
from typing import Final, List, Set

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

#: Destructive shell patterns refused outright (word-boundary,
#: case-insensitive) — the command injection surface of the NLP loop.
#: Moved here from ``agent.py`` so the policy has ONE owner (plan #16);
#: ``agent._DANGEROUS_SHELL_PATTERNS`` is now an alias of this object.
DESTRUCTIVE_SHELL_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"rm\s+-r[f]?", re.I), "recursive file removal (rm -r/-rf)"),
    (re.compile(r"\bdeltree\b", re.I), "deltree"),
    (re.compile(r"\brd\s+/s", re.I), "rd /s"),
    (re.compile(r"\brmdir\s+/s", re.I), "rmdir /s"),
    (re.compile(r"\bdel\s+/[sqf]", re.I), "del /s /q /f"),
    (re.compile(r"\bformat\s+[a-z]:", re.I), "format <drive>:"),
    (re.compile(r"\bshutdown\b", re.I), "shutdown"),
    (re.compile(r"\breboot\b", re.I), "reboot"),
    (re.compile(r"restart-computer", re.I), "restart-computer"),
    (re.compile(r"stop-computer", re.I), "stop-computer"),
    (re.compile(r"\bdiskpart\b", re.I), "diskpart"),
    (re.compile(r"\bmkfs\b", re.I), "mkfs"),
    (re.compile(r"wipefs", re.I), "wipefs"),
    (re.compile(r"\bdd\s+of=", re.I), "dd of="),
    (re.compile(r"taskkill\s+/f", re.I), "taskkill /f"),
    (re.compile(r"\breg\s+delete", re.I), "reg delete"),
    (re.compile(r"remove-item\s+-recurse", re.I), "Remove-Item -Recurse"),
    (re.compile(r"clear-recyclebin", re.I), "Clear-RecycleBin"),
    (re.compile(r"format-volume", re.I), "Format-Volume"),
    (re.compile(r"invoke-expression", re.I), "Invoke-Expression"),
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


def _strip_quoted_segments(cmd_str: str) -> str:
    """Blank out single/double-quoted spans so their contents are not scanned.

    ``python -c "import sys; sys.exit(3)"`` is a legitimate command: the ``;``
    is data inside a quoted argument, not a shell operator.  Quoted spans are
    replaced with an equal-length run of spaces so reported offsets stay valid
    and the surrounding text is still checked.
    """
    out: List[str] = []
    quote: str | None = None
    escaped = False
    for ch in cmd_str:
        if escaped:
            out.append(" " if quote else ch)
            escaped = False
            continue
        if ch == "\\" and quote:
            out.append(" ")
            escaped = True
            continue
        if quote:
            if ch == quote:
                quote = None
            out.append(" ")
            continue
        if ch in ("'", '"'):
            quote = ch
            out.append(" ")
            continue
        out.append(ch)
    return "".join(out)


def find_destructive_shell_pattern(cmd_str: str) -> str | None:
    """Return the first destructive pattern description in *cmd_str*, or None.

    This is the TIER-1 policy: the destructive block-list that every execution
    path shares (plan #16).  It is the only gate the permissive NLP ``run``
    dev-shell applies — that path deliberately still executes pipelines and
    chaining (see :func:`scan_command` for why), so it must NOT be given the
    structural scan.
    """
    if not cmd_str:
        return None
    for pattern, description in DESTRUCTIVE_SHELL_PATTERNS:
        if pattern.search(cmd_str):
            return description
    return None


def scan_command(cmd_str: str) -> str | None:
    """Return a description of the first policy violation in *cmd_str*.

    This is the STRICT policy (TIER-1 + TIER-2), for callers that execute
    through the binary allow-list (``tool_router``, ``tools/shell_ops``).  It
    applies, in order:

    1. destructive-pattern matching (:data:`DESTRUCTIVE_SHELL_PATTERNS`);
    2. the structural metacharacter scan (:func:`find_unsafe_shell_pattern`)
       on the command with quoted spans blanked out.

    Returns ``None`` when the command is acceptable — the caller then decides
    whether to apply the binary allow-list or its own fallback.

    Note the two-tier split: the NLP ``run`` tool intentionally does NOT use
    this function.  It keeps ``shell=True`` and must still execute pipes
    (``... 2>&1 | ...``), which on cmd.exe fail silently and are surfaced to
    the model as a wrong-shell Hint — a documented feature pinned by
    ``tests/test_tool_loop_nlp.py::TestRunToolShellAwareness``.  Rejecting
    metacharacters there would delete that diagnostic.
    """
    destructive = find_destructive_shell_pattern(cmd_str)
    if destructive is not None:
        return destructive
    return find_unsafe_shell_pattern(_strip_quoted_segments(cmd_str))


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
