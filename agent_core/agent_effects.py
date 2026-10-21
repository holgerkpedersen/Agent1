"""File-effects helpers extracted from Agent (refactor step r03).

Each former ``Agent`` method is a plain module-level function that takes the
owning :class:`Agent` instance as its first parameter.  Agent-instance
attributes (``workspace``, ``_pending_effects``, ...) are read through the
instance at call time so monkeypatching and subclass overrides keep working.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING, Any

from agent_core.commands.base import save_file_py
from agent_core.path_utils import to_windows_path

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agent import Agent


def _resolve_nlp_path(agent: "Agent", path: str) -> str:
    """Resolve a path for NLP tool use, honouring _nlp_workspace if set.

    Relative paths are resolved against the effective workspace (never the
    process CWD), so tools stay scoped to the agent's workspace.
    """
    base = agent._nlp_workspace or agent.workspace
    if not os.path.isabs(path):
        return os.path.normpath(os.path.join(base, path))
    return path


async def _save_verify_note(
    agent: "Agent", path: str, content: str, ok_msg: str
) -> str:
    """Persist *content*, py_compile-verify .py files, note the effect.

    Shared tail of the NLP ``write`` and ``edit`` tool handlers so both
    produce byte-identical success output (message + verification line)
    and identical trace-effect bookkeeping.  Returns the "Skipped …
    (no changes)" message when ``save_file_py`` finds nothing to do.
    """
    if not save_file_py(path, content, auto_yes=True):
        return f"Skipped {path} (no changes)"
    verify = await getattr(agent, "_verify_file")(path) if path.endswith(".py") else ""
    getattr(agent, "_note_effect")(path)
    return f"{ok_msg}\n{verify}".strip()


async def _verify_file(agent: "Agent", path: str) -> str:
    """Run py_compile on *path* and return a short verification summary."""
    try:
        cwd = os.path.abspath(to_windows_path(agent._nlp_workspace or agent.workspace))
        if not os.path.isdir(cwd):
            cwd = agent.workspace
        r = subprocess.run(
            [sys.executable, "-m", "py_compile", path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd,
        )
    except OSError as e:
        return f"[verify] py_compile could not run: {e}"
    if r.returncode == 0:
        return "[verify] py_compile ✓"
    return f"[verify] py_compile ✗: {r.stderr.strip()[:300]}"


def _note_effect(agent: "Agent", path: str) -> None:
    """Record one file affected by the current tool call.

    Only active while ``_pending_effects`` is armed (i.e. inside a traced
    chat_nlp run, decision #048) — otherwise this is a no-op cost-free
    guard, so untraced agent behaviour is unchanged.
    """
    if agent._pending_effects is None:
        return
    abs_path = os.path.abspath(to_windows_path(str(path)))
    if abs_path not in agent._pending_effects:
        agent._pending_effects.append(abs_path)


def _take_trace_effects(agent: "Agent", name: str, args: dict[str, Any]) -> list[str]:
    """``effects_fn`` for ToolLoopRunner: drain this tool call's effects.

    Returns the files affected by the just-executed tool call and resets
    the buffer for the next one.  Never raises (guarded by the loop too).
    """
    effects = list(agent._pending_effects or [])
    if agent._pending_effects is not None:
        agent._pending_effects.clear()
    return effects
