"""Signal handling, git branch detection, and interruptible input.

Extracted from agent.py as the first t00 concern (step t01).
Kept as a separate module so the signal handler has a stable import path
regardless of how agent.py is refactored.
"""
from __future__ import annotations

import signal
import subprocess
import sys
from typing import Any, Callable, Optional

# ── State ────────────────────────────────────────────────────────────────────

_shutdown_flag: bool = False
_input_executor: Optional[Callable[[], str]] = None


def _signal_break() -> None:
    """Called on SIGINT/SIGBREAK before the interpreter exits."""
    global _shutdown_flag
    _shutdown_flag = True


def _signal_break_with_cleanup(signum: int, frame) -> None:
    """Signal handler that also saves memory and chat history."""
    global _shutdown_flag
    _shutdown_flag = True
    try:
        import agent

        agent._shutdown_save_memory()  # type: ignore[attr-defined]
        agent._shutdown_trace_close()  # type: ignore[attr-defined]
    except Exception:
        pass
    sys.exit(128 + signum)


def _shutdown_save_memory() -> None:
    """Save agent memory (chat history, semantic memory)."""
    try:
        import agent

        ag: Optional[agent.Agent] = getattr(agent, "_shared_agent", None)  # type: ignore[attr-defined]
        if ag is not None:
            ag._save_memory()  # type: ignore[attr-defined]
            ag._save_chat_history()  # type: ignore[attr-defined]
    except Exception:
        pass


def _shutdown_trace_close() -> None:
    """Close any open trace writer."""
    try:
        from agent_core.monitoring.trace_writer import close_trace_writer

        close_trace_writer()
    except Exception:
        pass


def install(agent: Optional[Any] = None) -> None:
    """Install SIGINT/SIGBREAK handlers with cleanup on Ctrl-C.

    When an *agent* instance is provided, its cleanup callbacks (``_save_memory``
    and closing the active trace writer) are wired onto the ``agent`` module as
    ``_shutdown_save_memory`` / ``_shutdown_trace_close`` — the attributes
    ``_signal_break_with_cleanup`` invokes before exiting.
    """
    if agent is not None:
        import agent as agent_module

        def _close_active_trace() -> None:
            tw = getattr(agent, "_active_trace_writer", None)
            if tw is not None:
                tw.close()

        setattr(agent_module, "_shutdown_save_memory", agent._save_memory)
        setattr(agent_module, "_shutdown_trace_close", _close_active_trace)

    if sys.platform == "win32":
        signal.signal(signal.SIGBREAK, _signal_break_with_cleanup)
    else:
        signal.signal(signal.SIGINT, _signal_break_with_cleanup)
        # Ctrl-Z on POSIX sends SIGTSTP, whose default action suspends the
        # whole REPL (the shell then drops back to its prompt and the chat is
        # lost). Swallow it with a Python-level no-op handler: unlike SIG_IGN,
        # a handler is reset to default on exec(), so child shell commands
        # started by the agent keep normal job-control behaviour.
        signal.signal(signal.SIGTSTP, _ignore_signal)


def _ignore_signal(signum: int, frame) -> None:
    """No-op signal handler (used for SIGTSTP)."""


# ── Git branch ───────────────────────────────────────────────────────────────


def current_git_branch(cwd: str | None = None) -> str:
    """Return the name of the current Git branch, or 'HEAD' if detached."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            cwd=cwd,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "HEAD"


# ── Interruptible input ─────────────────────────────────────────────────────


def interruptible_input(prompt: str = "", executor: Callable[[], str] | None = None) -> str:
    """Read a line of input, supporting cancellation via Ctrl-C.

    On Windows the default input() is used; the executor parameter allows
    tests or custom run loops to supply an alternate source.
    """
    global _input_executor
    if executor is not None:
        _input_executor = executor
    if _input_executor is not None:
        return _input_executor()
    try:
        return input(prompt)  # type: ignore[func-returns-value]
    except (KeyboardInterrupt, EOFError):
        raise
