"""File-effects helpers extracted from Agent (refactor step r03).

Each former ``Agent`` method is a plain module-level function that takes the
owning :class:`Agent` instance as its first parameter.  Agent-instance
attributes (``workspace``, ``_pending_effects``, ...) are read through the
instance at call time so monkeypatching and subclass overrides keep working.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
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
    """Run py_compile on *path* and return a short verification summary.

    The ``[verify] py_compile`` marker is load-bearing: ``turn_loop``
    scans for it to drive the post-mutation self-review note, so it must
    stay the FIRST line of the returned summary.  The ruff result (see
    :func:`_lint_file`) is appended after it for that reason.
    """
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
        summary = "[verify] py_compile ✓"
    else:
        summary = f"[verify] py_compile ✗: {r.stderr.strip()[:300]}"
    lint = _lint_file(path)
    return f"{summary}\n{lint}".strip() if lint else summary


#: Wall-clock cap for one ruff child.  The pre-commit hook and CI block on
#: ruff, so a lint violation the agent writes is a red build later; running
#: ruff beside py_compile catches it while the file is still in hand.  The
#: child is capped for the same reason fix_cmd's children are (see
#: tests/test_fix_subprocess_guard.py): an unbounded child can wedge a turn.
RUFF_TIMEOUT_S = 30


def _ruff_binary() -> str | None:
    """Return the ruff executable path, or None when ruff is not installed.

    Ruff is a ``[dev]`` extra, so an end-user install legitimately lacks
    it: absent ruff means "cannot lint", never a failed verification.
    """
    found = shutil.which("ruff")
    if found:
        return found
    return None


def _ruff_command(path: str) -> list[str] | None:
    """Return the argv to lint *path*, or None when ruff is unavailable."""
    binary = _ruff_binary()
    if binary:
        base = [binary]
    elif importlib.util.find_spec("ruff") is not None:
        base = [sys.executable, "-m", "ruff"]
    else:
        return None
    # Explicit rule set so the result does not depend on an ambient
    # [tool.ruff] block, and so test fixtures lacking pyproject.toml still
    # get the same E/F/W gate CI enforces.
    return [
        *base,
        "check",
        "--no-cache",
        "--output-format=concise",
        "--select",
        "E,F,W",
        path,
    ]


def _format_ruff_violations(stdout: str) -> list[str]:
    """Turn ``--output-format=concise`` output into short ``file:line`` items."""
    items: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.startswith(("Found ", "All checks")):
            continue
        # concise format: path:line:col: CODE message
        parts = line.split(":", 3)
        if len(parts) < 4:
            continue
        items.append(f"{os.path.basename(parts[0])}:{parts[1]}: {parts[3].strip()}")
    return items


def _lint_file(path: str) -> str:
    """Return a ``[verify] ruff …`` summary line for *path*, or "".

    Never raises.  An empty string means "nothing to report": ruff is not
    installed, the file is not Python, or the lint run could not start.
    """
    if not str(path).endswith(".py"):
        return ""
    cmd = _ruff_command(path)
    if cmd is None:
        return ""
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=RUFF_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return f"[verify] ruff could not run: {e}"
    if r.returncode == 0:
        return "[verify] ruff ✓"
    violations = _format_ruff_violations(r.stdout or r.stderr)
    if not violations:
        return f"[verify] ruff ✗: {(r.stderr or r.stdout).strip()[:200]}"
    shown = "; ".join(violations[:5])
    more = f" (+{len(violations) - 5} more)" if len(violations) > 5 else ""
    return f"[verify] ruff ✗ {len(violations)}: {shown}{more}"


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
