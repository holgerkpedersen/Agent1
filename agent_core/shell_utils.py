"""Shell detection, command hints, and subprocess helpers.

Extracted from agent.py as the t00 shell-concern block (step t01).
Provides platform-specific command suggestions and subprocess utilities
used throughout the agent.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from typing import Optional

# ── Shell detection ─────────────────────────────────────────────────────────


def detect_shell() -> str:
    """Return the current shell name ('powershell', 'cmd', 'bash', ...)."""
    if sys.platform == "win32":
        ps_module = os.environ.get("PSModulePath", "")
        if ps_module:
            return "powershell"
        return "cmd"
    return os.environ.get("SHELL", "bash").split("/")[-1]


def shell_name_token() -> str:
    """Return a short token for the current shell, used in prompts."""
    return detect_shell()


def current_git_branch(cwd: str | None = None) -> str:
    """Return the current Git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "HEAD"


# ── Command hints ────────────────────────────────────────────────────────────


def unix_command_hint(command: str) -> str:
    """Return a hint line suggesting the Unix equivalent of a Windows command."""
    hints = {
        "dir": "ls",
        "type": "cat",
        "copy": "cp",
        "move": "mv",
        "del": "rm",
        "rmdir": "rmdir",
        "findstr": "grep",
        "fc": "diff",
        "where": "which",
    }
    cmd = command.strip().split()[0].lower() if command.strip() else ""
    if cmd in hints:
        return f"(on Unix: {hints[cmd]})"
    return ""


def windows_command_hint(command: str) -> str:
    """Return a hint line suggesting the Windows equivalent of a Unix command."""
    hints = {
        "ls": "dir",
        "cat": "type",
        "cp": "copy",
        "mv": "move",
        "rm": "del",
        "rmdir": "rmdir",
        "grep": "findstr",
        "diff": "fc",
        "which": "where",
    }
    cmd = command.strip().split()[0].lower() if command.strip() else ""
    if cmd in hints:
        return f"(on Windows: {hints[cmd]})"
    return ""


def shell_command_hint(command: str) -> str:
    """Return a platform-appropriate hint for the given command."""
    if sys.platform == "win32":
        return windows_command_hint(command)
    return unix_command_hint(command)


# ── Subprocess helpers ───────────────────────────────────────────────────────


def truncate_output(text: str, max_lines: int = 200, max_bytes: int = 50_000) -> str:
    """Truncate output to max_lines and max_bytes, appending a marker."""
    lines = text.splitlines()
    if len(lines) > max_lines:
        text = "\n".join(lines[:max_lines]) + f"\n... [{len(lines) - max_lines} more lines]"
    if len(text) > max_bytes:
        text = text[:max_bytes] + f"\n... [{len(text) - max_bytes} more bytes]"
    return text


def run_subprocess_captured(
    args: list[str],
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 60,
    capture_stderr: bool = True,
) -> tuple[int, str, str]:
    """Run a subprocess, capture stdout (and optionally stderr), return (code, stdout, stderr)."""
    kwargs: dict = {
        "capture_output": capture_stderr,
        "text": True,
        "timeout": timeout,
        "cwd": cwd,
    }
    if env is not None:
        merged_env = dict(os.environ)
        merged_env.update(env)
        kwargs["env"] = merged_env
    try:
        result = subprocess.run(args, **kwargs)  # type: ignore[arg-type]
        return result.returncode, result.stdout or "", result.stderr or ""
    except subprocess.TimeoutExpired:
        return -1, "", f"Timed out after {timeout}s"
    except Exception as exc:
        return -1, "", str(exc)


def shape_run_stderr(stderr: str) -> str:
    """Clean up stderr for display (remove ANSI codes, normalize newlines)."""
    # Remove ANSI escape codes
    ansi_escape = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
    cleaned = ansi_escape.sub("", stderr)
    return cleaned.strip()


def blocked_shell_command(command: str, shell_name: str | None = None) -> bool:
    """Return True if the command is blocked (destructive) in the current shell."""
    from agent_core.security import DESTRUCTIVE_SHELL_PATTERNS

    if shell_name is None:
        shell_name = detect_shell()
    lower = command.lower()
    for pattern in DESTRUCTIVE_SHELL_PATTERNS:
        if pattern.lower() in lower:
            return True
    return False


# ── Environment helpers ──────────────────────────────────────────────────────


def read_env_value(key: str, default: str = "") -> str:
    """Read an environment variable, returning default if not set."""
    return os.environ.get(key, default)


# ── Pytest helpers ──────────────────────────────────────────────────────────


def pytest_full_suite_timeout() -> int:
    """Return the pytest full-suite timeout in seconds from the environment."""
    default = 300
    try:
        from agent_core.constants import DEFAULT_FULL_SUITE_TIMEOUT
        return DEFAULT_FULL_SUITE_TIMEOUT
    except ImportError:
        pass
    return int(os.environ.get("PYTEST_FULL_SUITE_TIMEOUT", str(default)))


def is_full_pytest_command(args: list[str]) -> bool:
    """Return True if the pytest invocation is a whole-suite run."""
    if not args:
        return False
    # Full suite: bare 'pytest' or 'pytest tests/' with no --lf/--only flags
    no_target = len(args) <= 1 or (len(args) == 2 and args[1].startswith("-"))
    has_full_markers = any(a in args for a in ["tests/", "--all", "-m=", "--rootdir="])
    return no_target or has_full_markers


def is_pytest_command(args: list[str]) -> bool:
    """Return True if the command is any pytest invocation."""
    if not args:
        return False
    return "pytest" in args[0] or args[0].endswith("pytest")


def is_subset_pytest_command(args: list[str]) -> bool:
    """Return True if the pytest invocation targets a subset (not full suite)."""
    if not is_pytest_command(args):
        return False
    return is_full_pytest_command(args) is False


# ── SYSTEM_PROMPT ───────────────────────────────────────────────────────────


SYSTEM_PROMPT = ""
