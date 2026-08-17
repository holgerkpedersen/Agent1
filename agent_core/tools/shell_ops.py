"""Sandboxed shell command execution."""

import logging
import shlex
import subprocess
from pathlib import Path
from typing import Tuple

from agent_core.security.allowlist import (
    find_unsafe_shell_pattern,
    is_command_allowed,
)

logger = logging.getLogger(__name__)


def run_command(workspace_root: Path, cmd_str: str) -> Tuple[int, str, str]:
    """Execute a sandboxed shell command.

    1. Reject structural shell patterns (pipes, redirection, chaining).
    2. Parse the binary name from *cmd_str* and verify the allow-list.
    3. Run with ``shell=False`` to prevent injection attacks.

    Returns ``(returncode, stdout, stderr)``.
    """
    if not cmd_str.strip():
        return 1, "", "Empty command string."

    # --- Structural safety: no pipes / redirection / chaining ---------
    unsafe = find_unsafe_shell_pattern(cmd_str)
    if unsafe is not None:
        logger.warning("Rejected command with unsafe shell pattern %r: %s", cmd_str, unsafe)
        return 1, "", f"Command rejected: {unsafe}."

    # --- Parse safely -----------------------------------------------
    try:
        parts = shlex.split(cmd_str)
    except ValueError as exc:
        logger.warning("Failed to parse command %r: %s", cmd_str, exc)
        return 1, "", f"Malformed command: {exc}"

    if not parts:
        return 1, "", "Empty command."

    binary = Path(parts[0]).name  # strip any path component for allowlist check

    # --- Allow-list check -------------------------------------------
    if not is_command_allowed(binary):
        logger.warning("Blocked disallowed command %r", binary)
        return 1, "", f"Command '{binary}' is not allowed."

    # --- Execute (shell=False prevents injection) -------------------
    try:
        result = subprocess.run(
            parts,
            cwd=workspace_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as exc:
        logger.error("Command not found %r: %s", binary, exc)
        return 127, "", f"Command '{binary}' not found."
    except subprocess.TimeoutExpired:
        logger.warning("Command timed out after 30s: %s", cmd_str)
        return -1, "", "Command execution timed out (30s limit)."

    stdout = result.stdout if isinstance(result.stdout, str) else ""
    stderr = result.stderr if isinstance(result.stderr, str) else ""
    logger.debug(
        "Executed %r -> rc=%d | stdout_len=%d | stderr_len=%d",
        cmd_str,
        result.returncode,
        len(stdout),
        len(stderr),
    )
    return result.returncode, stdout, stderr