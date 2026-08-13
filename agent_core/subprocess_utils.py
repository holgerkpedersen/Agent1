from __future__ import annotations

import asyncio
import logging
import os
import shlex
import sys

from .exceptions import ToolExecutionError

logger = logging.getLogger(__name__)

# Shell builtins that exist only inside the shell (no standalone executable).
# On POSIX these are also available as binaries (/bin/echo), but on Windows
# cmd.exe they are builtins, so create_subprocess_exec() raises FileNotFoundError.
_SHELL_BUILTINS: frozenset[str] = frozenset(
    {"echo", "cd", "dir", "cls", "set", "del", "copy", "move", "type"}
)


async def run_subprocess_with_timeout(
    cmd: list[str],
    timeout_sec: float,
    cwd: str | None = None,
) -> tuple[int, bytes, bytes]:
    """Run subprocess with timeout and return (returncode, stdout, stderr).

    Uses ``create_subprocess_exec`` for platform-neutral, injection-safe execution.
    If the first token is a shell builtin that has no standalone executable on the
    current platform (e.g. ``echo`` on Windows cmd.exe), it transparently falls back
    to a hardened shell invocation so simple builtins still work without exposing an
    injection surface: arguments are validated against metacharacters before being
    passed through.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        return (proc.returncode or 0), stdout, stderr
    except FileNotFoundError as e:
        # The executable does not exist on this platform. If it is a known shell
        # builtin with safe arguments, fall back to the shell; otherwise propagate
        # the error so callers can report it meaningfully.
        if cmd and cmd[0].lower() in _SHELL_BUILTINS and _args_are_shell_safe(cmd):
            logger.debug(
                "Falling back to shell for builtin %r on cwd=%s", cmd[0], cwd or os.getcwd()
            )
            return await _run_via_shell(cmd, timeout_sec, cwd)
        raise ToolExecutionError("subprocess", f"Executable not found: {cmd[0]}") from e
    except asyncio.TimeoutError as e:
        logger.warning(
            "Subprocess timed out after %ss: %s", timeout_sec, " ".join(cmd)
        )
        raise ToolExecutionError(
            "subprocess", f"Timed out after {timeout_sec} seconds"
        ) from e


def _args_are_shell_safe(cmd: list[str]) -> bool:
    """Return True if command arguments contain no shell metacharacters.

    This is the guard that lets us safely fall back to a shell for builtins while
    keeping model-supplied arbitrary args injection-free: anything containing a
    separator, redirector, or substitution marker is rejected and must be run via
    ``create_subprocess_exec`` instead.
    """
    if not cmd[1:]:
        return True  # builtin with no arguments (e.g. just `echo`) is safe
    _dangerous: frozenset[str] = frozenset(
        {"|", "&", ";", "$", "(", ")", "<", ">", "`", "\n", "\r"}
    )
    for arg in cmd[1:]:
        if any(ch in arg for ch in _dangerous):
            return False
    return True


async def _run_via_shell(
    cmd: list[str], timeout_sec: float, cwd: str | None
) -> tuple[int, bytes, bytes]:
    """Hardened shell fallback for safe shell builtins.

    Builds a single command string from already-validated arguments and runs it via
    ``create_subprocess_shell`` with an explicit timeout. Used only when the builtin
    has no standalone executable on the current platform (e.g. echo on Windows).
    """
    if sys.platform == "win32":
        # cmd.exe: join validated args directly; builtins like echo are safe here.
        command_str = _join_for_cmd(cmd)
        shell_executable = None  # create_subprocess_shell uses the default shell
    else:
        command_str = shlex.join(cmd) if hasattr(shlex, "join") else " ".join(
            shlex.quote(a) for a in cmd
        )
        shell_executable = "/bin/sh"

    try:
        proc = await asyncio.create_subprocess_shell(
            command_str,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            executable=shell_executable,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        return (proc.returncode or 0), stdout, stderr
    except asyncio.TimeoutError as e:
        logger.warning(
            "Shell subprocess timed out after %ss: %s", timeout_sec, command_str
        )
        raise ToolExecutionError(
            "subprocess", f"Timed out after {timeout_sec} seconds"
        ) from e


def _join_for_cmd(cmd: list[str]) -> str:
    """Join builtin arguments for cmd.exe without introducing metacharacters.

    Only called after ``_args_are_shell_safe`` has rejected dangerous characters, so
    a plain join is safe here.
    """
    return " ".join(cmd)


# Type alias for consistent subprocess result handling
SubprocessResult = tuple[int, bytes, bytes]  # (returncode, stdout, stderr)

__all__: list[str] = ["run_subprocess_with_timeout", "SubprocessResult"]
