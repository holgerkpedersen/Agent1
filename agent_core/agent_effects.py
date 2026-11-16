"""File-effects helpers extracted from Agent (refactor step r03).

Each former ``Agent`` method is a plain module-level function that takes the
owning :class:`Agent` instance as its first parameter.  Agent-instance
attributes (``workspace``, ``_pending_effects``, ...) are read through the
instance at call time so monkeypatching and subclass overrides keep working.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tomllib
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
        # The model re-sent text it already wrote (2026-11-16: two of six
        # edits in one turn were byte-identical).  Say WHY and what to do
        # next, instead of a bare "no changes" it just retries.
        return (
            f"Skipped {path} (no changes)\n"
            "The new text is identical to what is already on disk, so nothing "
            "was written. Re-read the exact region (read with offset/limit) "
            "before editing again - or stop editing and run the code / answer."
        )
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


def _ruff_config_root(path: str) -> str | None:
    """Return the directory whose ruff config governs *path*, or None.

    Ruff resolves ``[tool.ruff]`` from the process CWD, not from the linted
    file — so linting a Kanban file while the agent's CWD is the Agent1
    checkout applied AGENT1's line-length (88) and reported twenty E501s the
    Kanban CLI run happily passed (2026-11-07 session, intents 37/44/56).
    When the file's own tree carries a ruff config (``ruff.toml``,
    ``.ruff.toml`` or a ``pyproject.toml`` with a ``[tool.ruff]`` table), the
    child runs from that directory so the TARGET repo's rules apply; only a
    config-less tree (test fixtures) falls back to the explicit rule set.
    """
    current = os.path.dirname(os.path.abspath(path))
    while True:
        for name in ("ruff.toml", ".ruff.toml"):
            if os.path.isfile(os.path.join(current, name)):
                return current
        pyproject = os.path.join(current, "pyproject.toml")
        if os.path.isfile(pyproject):
            try:
                # tomllib needs BYTES (UTF-8 invariant: never decode first).
                with open(pyproject, "rb") as fh:
                    data = tomllib.load(fh)
            except (OSError, tomllib.TOMLDecodeError):
                return None
            if isinstance(data.get("tool"), dict) and "ruff" in data["tool"]:
                return current
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def _ruff_command(path: str) -> list[str] | None:
    """Return the argv to lint *path*, or None when ruff is unavailable."""
    binary = _ruff_binary()
    if binary:
        base = [binary]
    elif importlib.util.find_spec("ruff") is not None:
        base = [sys.executable, "-m", "ruff"]
    else:
        return None
    # A target repo with its OWN [tool.ruff] is linted by its own rules:
    # _lint_file runs the child from the config root (see _ruff_config_root)
    # so ruff resolves the repo's real settings — an explicit --select there
    # would override settings like line-length the repo deliberately chose.
    root = _ruff_config_root(path)
    if root is not None:
        return [*base, "check", "--no-cache", "--output-format=concise", path]
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


#: One line of ruff ``--output-format=concise``: path:line:col: CODE message.
#: Greedy path so ``C:`` never becomes the first split point on Windows.
_CONCISE_LINE_RE = re.compile(r"^(?P<path>.+):(?P<line>\d+):(?P<col>\d+): (?P<msg>.+)$")

#: Rule codes that are real defects: parse errors (E9xx) and undefined names
#: (F821) — the ONLY findings this repo's CI blocks on top of parsing
#: (AGENTS.md: blocking ``ruff check --select F821`` + advisory full lint).
#: Everything else in the E/F/W set (E501, W293, W292, F401, ...) is style.
#:
#: Reporting style as a red ``✗`` cost a whole turn on 2026-11-16: a 119B
#: model wrote a Playwright script, then spent six edits chasing E501/W293 in
#: it and never ran it.  A verification line the model reads as "the write
#: FAILED" must mean a defect, not a formatting nit.
_BLOCKING_RUFF_CODES_RE = re.compile(r"^(?:E9\d+|F821)$")

#: Machine hint appended to advisory-only findings.  Read by the model, not a
#: log: it says explicitly that another edit is not the expected response.
_ADVISORY_HINT = "advisory - do not edit again just for formatting"

#: Extract the rule code from one ``_format_ruff_violations`` item
#: (``x.py:3: F401 `os` imported but unused``).  The position prefix is
#: matched first, so a message that happens to quote a code cannot win.
_ITEM_CODE_RE = re.compile(r":\d+: (?P<code>[A-Z]+\d+)\b")


def _ruff_code(item: str) -> str:
    """Rule code of one violation item ("" when it cannot be parsed)."""
    match = _ITEM_CODE_RE.search(item)
    return match.group("code") if match else ""


def _is_blocking_ruff_violation(item: str) -> bool:
    """True when *item* is a defect worth another edit, not formatting.

    Unknown/unparseable items count as blocking on purpose: an unreadable
    finding must never be silently downgraded to advice.
    """
    code = _ruff_code(item)
    if not code:
        return True
    return bool(_BLOCKING_RUFF_CODES_RE.match(code))


def _format_ruff_violations(stdout: str) -> list[str]:
    """Turn ``--output-format=concise`` output into short ``file:line`` items."""
    items: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.startswith(("Found ", "All checks")):
            continue
        # concise format: path:line:col: CODE message.  The path is matched
        # greedily so a Windows drive-letter colon (``C:\dev\x.py:8:5: E501``)
        # is NOT the split point — a plain split(":", 3) cut at ``C:`` and
        # rendered every finding as ``C: ...\x.py: <col>: <code>``.
        m = _CONCISE_LINE_RE.match(line)
        if m is None:
            continue
        items.append(
            f"{os.path.basename(m.group('path'))}:{m.group('line')}: "
            f"{m.group('msg').strip()}"
        )
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
        # cwd = the file's ruff config root (None = inherit): ruff resolves
        # [tool.ruff] from the CHILD'S cwd, so a cross-repo write is linted
        # by the repo it lands in, not by the agent's checkout (Kanban
        # 2026-11-07: 20 phantom E501s per patch, all Agent1's 88-col rules).
        r = subprocess.run(
            cmd,
            cwd=_ruff_config_root(path),
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
    blocking = [v for v in violations if _is_blocking_ruff_violation(v)]
    style = [v for v in violations if not _is_blocking_ruff_violation(v)]
    if not blocking:
        # Style only: report it once, WITHOUT the ✗ marker, so neither the
        # model nor the colouriser reads it as a failed write.
        shown = "; ".join(style[:5])
        more = f" (+{len(style) - 5} more)" if len(style) > 5 else ""
        return (
            f"[verify] ruff style {len(style)} ({_ADVISORY_HINT}): {shown}{more}"
        )
    shown = "; ".join(blocking[:5])
    more = f" (+{len(blocking) - 5} more blocking)" if len(blocking) > 5 else ""
    summary = f"[verify] ruff ✗ {len(blocking)} blocking: {shown}{more}"
    if style:
        codes = ",".join(sorted({_ruff_code(v) for v in style if _ruff_code(v)}))
        style_note = f"+{len(style)} style"
        if codes:
            style_note += f" ({codes})"
        summary += f"; {style_note} {_ADVISORY_HINT}"
    return summary


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
