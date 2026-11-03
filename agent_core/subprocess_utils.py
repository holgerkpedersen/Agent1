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
    proc: asyncio.subprocess.Process | None = None
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
                "Falling back to shell for builtin %r on cwd=%s",
                cmd[0],
                cwd or os.getcwd(),
            )
            return await _run_via_shell(cmd, timeout_sec, cwd)
        raise ToolExecutionError("subprocess", f"Executable not found: {cmd[0]}") from e
    except asyncio.TimeoutError as e:
        logger.warning("Subprocess timed out after %ss: %s", timeout_sec, " ".join(cmd))
        # Kill and reap the child so its pipe transports close before the loop ends.
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
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
        command_str = (
            shlex.join(cmd)
            if hasattr(shlex, "join")
            else " ".join(shlex.quote(a) for a in cmd)
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


def _posix_guidance(shell: str) -> str:
    """Guidance for a POSIX shell — the mirror of the Windows wording: no
    cmd.exe builtins, no PowerShell cmdlets."""
    return (
        f"Use {shell} syntax; forward-slash paths. There are NO cmd.exe "
        "builtins (dir/type/findstr/copy/del/where) and NO PowerShell cmdlets "
        "(Get-ChildItem/Select-String/Get-Content) — use ls/cat/grep/find, or a "
        'Python one-liner (python -c "..."). Run output is auto-truncated.'
    )


def shell_info() -> dict[str, str]:
    """Return shell metadata for inclusion in LLM system prompts.

    DERIVES from :func:`agent_core.shell_utils.detect_shell` — the ONE
    detector that also governs real execution (card "Communicating current
    shell": this module used to run its own second, independent detector, so
    the tool-schema text could advertise one shell while ``shell_utils``
    executed another).  The token is the canonical display name; on Windows
    it gains the ``.exe`` suffix naming the executable the executor invokes.

    Returns a dictionary with ``name`` (e.g. ``cmd.exe`` or ``/bin/bash``),
    ``flag`` (the flag used to pass a command string, e.g. ``/c`` or ``-c``),
    ``separator`` (e.g. ``&`` or ``&&``), and ``guidance`` (a short human
    sentence the LLM can use).
    """
    from agent_core import shell_utils

    token = shell_utils.detect_shell()

    if token in ("powershell", "pwsh"):
        # Windows: a PowerShell session owns this process tree.  The psutil
        # ancestor walk is authoritative when installed; PS session env vars
        # are the documented best-effort fallback without it (see
        # shell_utils._windows_shell_token).
        return {
            "name": "powershell.exe" if token == "powershell" else "pwsh",
            "flag": "-Command",
            "separator": ";",
            "guidance": (
                "Use PowerShell syntax; paths use backslash. There is NO "
                "tail/head/sed/awk/wc and NO piping to them (run output is "
                "auto-truncated). PowerShell aliases ls/grep/cat/rm/curl do "
                "exist; call Python tools as 'python -m <tool>'."
            ),
        }

    if token == "cmd":
        return {
            "name": "cmd.exe",
            "flag": "/c",
            "separator": "&",
            "guidance": (
                "Use cmd.exe syntax; paths use backslash. cmd.exe has NO Unix "
                "tools — never use tail/head/grep/ls/cat/sed/awk/wc or pipe to "
                "them (run output is auto-truncated). Use dir/type/findstr/more "
                'instead, or a Python one-liner (python -c "..."). Call Python '
                "tools as 'python -m <tool>'."
            ),
        }

    # POSIX shell — the token IS the display name.  With no SHELL evidence at
    # all, report the full default path (/bin/bash) rather than a bare token.
    if sys.platform != "win32" and not os.environ.get("SHELL"):
        return {
            "name": "/bin/bash",
            "flag": "-c",
            "separator": "&&",
            "guidance": _posix_guidance("bash/sh"),
        }
    return {
        "name": token,
        "flag": "-c",
        "separator": "&&",
        "guidance": _posix_guidance(token),
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
