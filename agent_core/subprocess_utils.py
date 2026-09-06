from __future__ import annotations

import asyncio
import logging
import os
import shlex
import sys

try:
    import psutil as _psutil
except ImportError:  # pragma: no cover – psutil is optional at import time
    _psutil = None  # type: ignore[assignment]

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


_PSHELL_NAMES = frozenset({"powershell.exe", "pwsh.exe"})


def _walk_for_powershell() -> bool:
    """Return True if ``powershell.exe`` or ``pwsh.exe`` appears anywhere in
    the ancestor process chain of the current process.

    Uses :mod:`psutil` to walk up from the direct parent.  If *psutil* is not
    installed the check is silently skipped (returns ``False``).
    """
    if _psutil is None:  # pragma: no cover
        return False
    try:
        proc = _psutil.Process(os.getppid())
    except (_psutil.NoSuchProcess, _psutil.AccessDenied):  # pragma: no cover
        return False
    while proc is not None:
        if proc.name().lower() in _PSHELL_NAMES:
            return True
        try:
            proc = proc.parent()
        except (_psutil.NoSuchProcess, _psutil.AccessDenied):  # pragma: no cover
            break
    return False


def shell_info() -> dict[str, str]:
    """Return shell metadata for inclusion in LLM system prompts.

    Detects the *actual* shell the agent runs in by inspecting the ``SHELL``
    environment variable first, then falling back to platform defaults.

    Returns a dictionary with ``name`` (e.g. ``cmd.exe`` or ``/bin/bash``),
    ``flag`` (the flag used to pass a command string, e.g. ``/c`` or ``-c``),
    ``separator`` (e.g. ``&`` or ``&&``), and ``guidance`` (a short human
    sentence the LLM can use).
    """
    shell_env = os.environ.get("SHELL", "")
    shell_name = os.path.splitext(os.path.basename(shell_env))[0].lower() if shell_env else ""

    # POSIX-like shell detected (bash, zsh, sh, fish, …) – even on Windows
    # when Git Bash / MSYS2 / WSL is in use.
    if shell_name in ("bash", "sh", "zsh", "fish", "dash", "ash", "ksh"):
        return {
            "name": shell_name,
            "flag": "-c",
            "separator": "&&",
            "guidance": f"Use {shell_name} syntax. Path separators use forward slash.",
        }

    # Windows: detect PowerShell by walking the process tree.  The run-tool
    # injects cmd.exe as the direct parent, but the *user's* shell lives
    # higher up in the chain (e.g. Python → cmd.exe → powershell.exe → …).
    # Walking ancestors with psutil is the only reliable way to tell them
    # apart, because environment variables like PSModulePath are persistent
    # and inherited even in plain cmd.exe sessions.
    if sys.platform == "win32":
        ps_parent = _walk_for_powershell()
        if ps_parent:
            return {
                "name": "powershell.exe",
                "flag": "-Command",
                "separator": ";",
                "guidance": "Use PowerShell syntax. Path separators use backslash.",
            }
        return {
            "name": "cmd.exe",
            "flag": "/c",
            "separator": "&",
            "guidance": "Use cmd.exe syntax. Path separators use backslash.",
        }

    # POSIX default
    return {
        "name": "/bin/bash",
        "flag": "-c",
        "separator": "&&",
        "guidance": "Use bash/sh syntax. Path separators use forward slash.",
    }


def _join_for_cmd(cmd: list[str]) -> str:
    """Join builtin arguments for cmd.exe without introducing metacharacters.

    Only called after ``_args_are_shell_safe`` has rejected dangerous characters, so
    a plain join is safe here.
    """
    return " ".join(cmd)


# Type alias for consistent subprocess result handling
SubprocessResult = tuple[int, bytes, bytes]  # (returncode, stdout, stderr)

__all__: list[str] = ["run_subprocess_with_timeout", "shell_info", "SubprocessResult"]
