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
from typing import Any, Optional

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
#: Destructive shell block-list (TIER-1).  One owner: the shared security
#: module.  agent.py re-exports this as the legacy ``_DANGEROUS_SHELL_PATTERNS``
#: alias, so a rename here must keep that alias working.
from agent_core.security.allowlist import DESTRUCTIVE_SHELL_PATTERNS

# Two distinct hint families live here — do not conflate them:
#  * platform_hints (zero-arg): static text about what the CURRENT shell
#    lacks, appended to failed run results. Contract asserted by
#    tests/test_agent_robustness.py::test_shell_command_hint_is_platform_aware
#    (zero-arg calls; "no Unix tools" on Windows, "cmd.exe" on POSIX).
#    These were the original inline agent.py hints and were lost when the
#    import block was rewritten at ac9a3df.
#  * command translation maps (take a command string): suggest the
#    equivalent command on the other OS.


def platform_unix_hint() -> str:
    """Hint appended to run results when a Unix-ism fails on Windows
    (tail/grep/ls/mypy-as-command are not available in the current shell)."""
    return (
        f"\nHint: this shell ({detect_shell()}) has no Unix tools like tail/grep/ls, "
        "and installed Python tools must be called as 'python -m <tool>'. Use Python "
        'one-liners (python -c "...") or the built-in tools — run output is truncated '
        "automatically, so pipes like '2>&1 | tail -40' are neither needed nor supported."
    )


def platform_windows_hint() -> str:
    """Hint appended to run results when a Windows-ism fails on POSIX
    (cmd.exe builtins and PowerShell cmdlets are not available there)."""
    return (
        f"\nHint: this shell ({detect_shell()}) has no cmd.exe builtins "
        "(dir/type/findstr/copy/del/where) and no PowerShell cmdlets "
        "(Get-ChildItem/Select-String/Get-Content). Use Python one-liners "
        '(python -c "...") or the built-in tools.'
    )


def platform_shell_command_hint() -> str:
    """Return the platform-appropriate static hint for the CURRENT shell."""
    if os.name == "nt":
        return platform_unix_hint()
    return platform_windows_hint()


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
    """Return a command-specific translation hint for *command*.

    Distinct from :func:`platform_shell_command_hint`, which takes no
    argument and returns the static hint about the CURRENT shell.
    """
    if sys.platform == "win32":
        return windows_command_hint(command)
    return unix_command_hint(command)


# ── Subprocess helpers ───────────────────────────────────────────────────────


def truncate_output(
    text: str | None,
    max_lines: int = 200,
    max_bytes: int = 50_000,
    keep: int = 2500,
) -> str:
    """Truncate output, keeping the head and tail, with a marker in between.

    Empty/None output becomes the ``(no output)`` placeholder: a subprocess
    that produced nothing yields ``None`` from ``communicate()``, and the
    agent's run tool needs a non-empty string so the LLM can tell "ran,
    printed nothing" from "output was lost".

    Long output keeps ``keep`` characters at each end (the tail matters —
    pytest failures end at the summary line) and collapses the middle into
    the ``... [truncated] ...`` marker.  The middle is measured in
    characters, not lines: a mid-section that is one giant line (no
    newlines) must not survive the cap.  The result is always strictly
    shorter than the input — the marker counts against the budget.
    """
    if not text:
        return "(no output)"
    if len(text) <= max_bytes and len(text) <= 2 * keep:
        # Short enough to pass through — subject to the line cap only.
        lines = text.splitlines()
        if len(lines) > max_lines:
            dropped = len(lines) - max_lines
            return (
                "\n".join(lines[:max_lines])
                + f"\n... [truncated] ... [{dropped} more lines]"
            )
        return text

    # Char-based head/tail split.  Reserve room for the marker so the
    # result is guaranteed shorter than the input.
    dropped = len(text) - 2 * keep
    marker = f"\n... [truncated] ... [{dropped} chars omitted]\n"
    budget = len(text) - len(marker)
    half = max(budget // 2, 1)
    head = text[:half]
    tail = text[-half:]
    return head + marker + tail


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


#: cmd.exe's wording when a command name is not on PATH.
_COMMAND_NOT_FOUND_RE = re.compile(
    r"is not recognized as an internal or external command", re.I
)
#: POSIX shells' wording for the same failure.
_COMMAND_NOT_FOUND_UNIX_RE = re.compile(
    r"(command not found|No such file or directory)", re.I
)


def shape_run_stderr(
    stderr: Any,
    output: Any = None,
    returncode: Any = None,
) -> str:
    """Clean up run-tool output for display.

    Combines the captured *output* with cleaned *stderr* (ANSI codes
    stripped, newlines normalised) and — when *returncode* is non-zero —
    appends ``[EXIT CODE: N]`` so the model can read the exit code
    directly; ``None``/``0`` returncodes add no marker.

    A non-zero returncode also gets a shell Hint (the run tool's
    shell-awareness contract, pinned by
    ``tests/test_tool_loop_nlp.py::TestRunToolShellAwareness``):

    * the shell's "command not found" wording → the static
      :func:`platform_shell_command_hint` text, so the model learns which
      shell it is on and what it lacks;
    * the Windows cmd.exe whole-pipeline failure (rc 255, no output at
      all) → the static hint PLUS a note that ``|`` runs each stage in a
      separate process.

    Called with only *stderr*, *output*/*returncode* default and
    the result is the cleaned stderr string alone.
    """
    # Remove ANSI escape codes
    ansi_escape = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
    cleaned = ansi_escape.sub("", stderr or "").strip()
    out = "" if output is None else str(output)
    result = f"{out}\n{cleaned}" if cleaned else out
    if returncode is None or returncode == 0:
        return result
    result = f"{result}\n[EXIT CODE: {returncode}]"
    # Shell-awareness hints (only on failure, so a passing command stays clean).
    if _COMMAND_NOT_FOUND_RE.search(cleaned) or (
        os.name != "nt" and _COMMAND_NOT_FOUND_UNIX_RE.search(cleaned)
    ):
        result += platform_shell_command_hint()
    if (
        not out.strip()
        and not cleaned
        and returncode == 255
        and os.name == "nt"
    ):
        # The Windows cmd.exe whole-pipeline failure path: a `|` runs each
        # stage separately and only the LAST stage's code survives, so an
        # early failure surfaces as 255 with no output at all.
        result += platform_shell_command_hint() + (
            "\nNote: a pipeline (|) runs each stage in a separate process;"
            " on Windows the exit code is the LAST stage's. On Unix-like"
            " shells use 'set -o pipefail' to surface a failing stage."
        )
    return result


def blocked_shell_command(command: str, shell_name: str | None = None) -> str | None:
    """Return the description of the first destructive pattern in *command*,
    or ``None`` when it is safe.

    Delegates to the shared TIER-1 scanner
    (:func:`agent_core.security.allowlist.find_destructive_shell_pattern`) —
    the destructive block-list has ONE owner.  The strict structural scan
    (pipes, chaining) is deliberately NOT applied here: the NLP ``run``
    dev-shell must still execute pipelines.
    """
    from agent_core.security.allowlist import find_destructive_shell_pattern

    return find_destructive_shell_pattern(command)


# ── Environment helpers ──────────────────────────────────────────────────────


def read_env_value(key: str, default: str = "") -> str:
    """Read an environment variable, returning default if not set."""
    return os.environ.get(key, default)


# ── Pytest helpers ──────────────────────────────────────────────────────────


def pytest_full_suite_timeout() -> float:
    """Return the pytest full-suite timeout budget in seconds.

    Delegates to :mod:`agent_core.pytest_cmd_policy` (process env wins, else
    the repo ``.env``, ``max(floor, last_run * 1.5)``) so the tool timeout and
    conftest's watchdog share ONE budget implementation.
    """
    from agent_core import pytest_cmd_policy
    return pytest_cmd_policy._pytest_full_suite_timeout()


def is_full_pytest_command(args: list[str] | str) -> bool:
    """Return True if the pytest invocation is a whole-suite run.

    Delegates to :func:`agent_core.pytest_cmd_policy._is_full_pytest_command`
    (segment/shlex based).  The old inline check here was list membership over
    ``args`` — when the run tool passed a command STRING, ``"tests/" in args``
    became a substring test and any targeted ``tests/test_x.py::test_y`` was
    misclassified as a full run (see the policy module docstring).
    """
    from agent_core import pytest_cmd_policy
    return pytest_cmd_policy._is_full_pytest_command(args)


def is_pytest_command(args: list[str] | str) -> bool:
    """Return True if the command is any pytest invocation (delegates to policy)."""
    from agent_core import pytest_cmd_policy
    return pytest_cmd_policy._is_pytest_command(args)


def is_subset_pytest_command(args: list[str] | str) -> bool:
    """Return True if the pytest invocation targets a subset (not full suite)."""
    from agent_core import pytest_cmd_policy
    return pytest_cmd_policy._is_subset_pytest_command(args)


# ── SYSTEM_PROMPT ───────────────────────────────────────────────────────────


#: Shell name for the system prompt, evaluated ONCE at import time (the
#: frozen ``agent._SHELL_NAME`` surface — tests/test_tool_loop_nlp.py
#: ::TestRunToolShellAwareness asserts it equals _shell_name_token() and
#: appears as "SHELL_NAME=..." inside _SYSTEM_PROMPT).
SHELL_NAME: str = shell_name_token()

SYSTEM_PROMPT = (
    "You are a senior coding assistant working inside this project workspace.\n"
    "The user speaks natural language; you have tools to search, read, write, "
    "edit, run, and test the code.\n"
    "You also talk normally: for greetings, small talk, or general questions "
    "that do not need the workspace, reply directly in text without calling "
    "any tool.\n\n"
    "WORKING METHOD:\n"
    "1. Understand the request.\n"
    "2. Take concrete action with the tools - never just describe what you "
    "would do. If you intend to read, search, edit or run something, call the tool.\n"
    "3. After a write/edit, a py_compile verification summary is returned "
    "automatically - report it.\n"
    "4. Finish with a short report: what you changed, where, and the "
    "verification/test evidence.\n\n"
    "RULES:\n"
    f"You are running inside SHELL_NAME={shell_name_token()} ({detect_shell()}). "
    "This is a hard constraint: on Windows (cmd/PowerShell) there is NO "
    "tail/grep/ls/find; on POSIX, no PowerShell cmdlets. Use Python one-liners "
    '(python -c "...") or the built-in tools instead of shell pipes - never '
    "construct commands that mix Unix and Windows syntax in a single invocation.\n"
    f"- The shell for the run tool is: {detect_shell()}. On Windows there "
    "one-liners (python -c \"...\") or the built-in tools; run output is "
    "truncated to 5000 chars automatically. The run tool reports non-zero "
    "exit codes as [EXIT CODE: N] at the end of its output - read it from "
    "there, never chain '...; echo $?' (the model cannot see $? afterwards).\n"
    "- Python's sys.exitcode is only set at interpreter exit - use "
    "sys.exit(). Shell test tools ship with pytest (scripts exit non-zero "
    "on failure).\n"
    "- You have an effectively unlimited tool budget: do not rush, do not "
    "stop early, and do not plan around a budget. Take as many tool calls "
    "as the task needs - but make steady progress: when exploration is "
    "done, actually edit/write/fix instead of only reading.\n"
    "- read returns up to 5000 chars per call by default - never request "
    "tiny slices (limit < 1000). Read big chunks and continue with the "
    "offset hint; avoid many small read calls.\n"
    "- For 'where is X used/defined?' use references (one call) instead of "
    "search + reads. To orient inside a large file, call definitions first, "
    "then read only the line window you need - do not page blindly.\n"
    "- Prefer targeted edits over rewriting whole files.\n"
    "- If a tool fails, read the error and try a different approach.\n"
    "- When the request is ambiguous, make a reasonable assumption and state it, "
    "or ask one clarifying question before acting.\n"
    "- Never assert facts about this repo that you have not verified with a "
    "tool. project_plan.md / project_tasks.md are HISTORICAL phase docs and "
    "may be outdated - verify claims against the actual code.\n"
    f"- You are currently on Git branch: {current_git_branch()}.\n"
    "- Git: call the git tool with subcommand='status' FIRST, before staging or "
    "committing. Stage everything with subcommand='add', args='-A' - a bare "
    "'-' is NOT a pathspec and will fail. Commit with subcommand='commit', "
    "args='-m \"message\"'. A request to \"commit changes\" means the FULL "
    "sequence status -> add -A -> commit -m -> (only if asked) push; never "
    "push alone, because `git push` only sends commits and would silently "
    "leave staged work behind. Never invent flags.\n"
    "- Verify numbers (e.g. how many tests exist) with the tests tool or git "
    "log before claiming them.\n"
    "- NEVER claim a repository state you have not observed with a tool this "
    "turn. Output the user saw in the terminal is NOT in your context - `git "
    "status` typed as a chat command is executed by the REPL and printed to "
    "stdout, so you never see it. If you called no tool, you do not know what "
    "is staged, committed or pushed. Call the git tool before stating it; an "
    "unverified \"already committed and pushed\" is a fabrication.\n"
    "- If a search finds nothing in source files, state that the symbol does "
    "not exist in the current code - never repeat the same search.\n"
    "- A failing test is a real signal: fix the implementation - never weaken "
    "or delete an assertion just to make it pass. Only change a test if the "
    "test itself is demonstrably wrong, and say why.\n"
    "- Every bug fix ships a permanent regression test in tests/ (pytest). "
    "Delete scratch scripts (_tmp_*.py) before finishing.\n"
    "- Verify fixes against the REAL code path (import the actual function), "
    "not a copied simulation of it.\n"
    "- Be concise. Answer in the user's language.\n"
    "- IMPORTANT: If the user message is a greeting (hi, hello, hey, etc.) "
    "or small talk, reply with a short text greeting. Do NOT call any tools."
)
