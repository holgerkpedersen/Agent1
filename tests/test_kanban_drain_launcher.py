"""Regression: the scheduled drain launcher must match the REAL CLI.

The queue only drains while an interactive agent session runs its boot poller
(``agent_core/run_utils.py`` -> ``start_inbound_processor``).  With no session
open, ``queue-kanban-to-agent1`` is nobody's job, so messages pile up and the
offset stops moving.  ``scripts/run_kanban_queue_drain.cmd`` exists to let
Task Scheduler be the single handler in that gap.

A launcher is the easiest thing in a repo to rot silently: it is not imported
by anything, so a renamed flag or a hard-coded interpreter path fails only at
03:00 on the machine that matters.  These tests pin the two halves that would
break that way:

1. every flag the script passes is still accepted by
   ``harnessfix.kanban_bridge.main()`` — asserted by running the real
   ``main()`` with the script's own argument list;
2. the script path-qualifies the interpreter and the working directory,
   because a task action that trusts PATH or "Start in" dies with
   ``0x80070002`` (ERROR_FILE_NOT_FOUND).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path, PureWindowsPath

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "run_kanban_queue_drain.cmd"
HIDDEN_WRAPPER = REPO_ROOT / "scripts" / "run_hidden.vbs"
TASK_XML = REPO_ROOT / "scripts" / "kanban_queue_drain_task.xml"


def _script_text() -> str:
    assert SCRIPT.is_file(), f"launcher is missing: {SCRIPT}"
    # cp1252 fallback: a future editor saving this file as ANSI must not turn
    # the assertion failure into a UnicodeDecodeError.
    return SCRIPT.read_text(encoding="utf-8", errors="replace")


def _script_vars() -> dict[str, str]:
    """Every ``set "NAME=value"`` the launcher declares, for %VAR% expansion."""
    return dict(re.findall(r'(?m)^\s*set "([A-Z_][A-Z0-9_]*)=([^"]*)"', _script_text()))


def _expand(token: str, variables: dict[str, str]) -> str:
    """Expand ``%VAR%`` the way cmd does, so argparse sees the real value."""
    return re.sub(
        r"%([A-Z_][A-Z0-9_]*)%",
        lambda m: variables.get(m.group(1), m.group(0)),
        token,
    )


def _cli_args() -> list[str]:
    """The ``%PYTHON_EXE% -m harnessfix.kanban_bridge ...`` argument list."""
    text = _script_text()
    match = re.search(
        r"\"%PYTHON_EXE%\"\s+-m\s+harnessfix\.kanban_bridge\s+([^\r\n]+)",
        text,
    )
    assert match, "launcher must invoke harnessfix.kanban_bridge via -m"
    # Strip the ``>> "%LOG_FILE%" 2>&1`` redirection tail.
    tail = match.group(1).split(">>")[0]
    variables = _script_vars()
    return [_expand(tok, variables) for tok in tail.split() if tok]


def test_launcher_exists_and_is_invoked_as_a_module():
    """``-m`` from the repo root is what makes the package importable."""
    assert "-m harnessfix.kanban_bridge" in _script_text()
    assert "cd /d \"%REPO_ROOT%\"" in _script_text(), (
        "the launcher must set the working directory explicitly; the task's "
        "'Start in' is not guaranteed"
    )


def test_launcher_uses_an_absolute_interpreter_path():
    """PATH is per-account; a scheduled task gets a different PATH."""
    text = _script_text()
    match = re.search(r'set "PYTHON_EXE=([^"]+)"', text)
    assert match, "PYTHON_EXE must be set explicitly"
    assert PureWindowsPath(match.group(1)).is_absolute(), (
        f"PYTHON_EXE must be absolute, got {match.group(1)!r}"
    )


def test_launcher_waits_for_the_lock_instead_of_skipping(monkeypatch):
    """A timer that gives up the moment a poller holds the lock is useless.

    ``process_inbound`` refuses to drain a queue another process owns; that is
    the guard which stopped the duplicated ids.  So the scheduled run must
    WAIT (``--timeout`` > 0) rather than report "Processed 0" forever.
    """
    args = _cli_args()
    assert "--process-in" in args
    assert "--timeout" in args, "a scheduled drain must be able to wait for the lock"

    timeout_idx = args.index("--timeout")
    raw = args[timeout_idx + 1]
    # Either a literal or the %LOCK_WAIT_SECONDS% variable declared in-script.
    if raw.startswith("%"):
        var = raw.strip("%")
        match = re.search(rf'set "{var}=(\d+)"', _script_text())
        assert match, f"{var} must be set to a number in the launcher"
        raw = match.group(1)
    assert float(raw) > 0, (
        "timeout 0 makes the scheduled run a no-op while the agent runs"
    )


def test_every_launcher_flag_is_accepted_by_the_real_cli(monkeypatch, tmp_path):
    """Run the ACTUAL argparse of ``main()`` with the script's argument list."""
    import harnessfix.kanban_bridge as kb

    args = _cli_args()
    seen: list = []
    monkeypatch.setattr(
        kb, "process_inbound", lambda qdir, **kwargs: seen.append(kwargs) or 0
    )
    monkeypatch.setattr(sys, "argv", ["kanban_bridge", *args])

    # argparse raises SystemExit(2) on an unknown flag, which is exactly the
    # silent rot these tests exist to catch.
    assert kb.main() == 0, f"launcher flags were rejected by the real CLI: {args}"
    assert seen, "the launcher must reach process_inbound"


def test_launcher_does_not_start_a_second_daemon(monkeypatch):
    """One-shot drain only: no background poller, no competing handler."""
    import harnessfix.kanban_bridge as kb

    started: list = []
    monkeypatch.setattr(
        kb, "start_inbound_processor", lambda *a, **k: started.append(1)
    )
    monkeypatch.setattr(kb, "process_inbound", lambda qdir, **kwargs: 0)
    monkeypatch.setattr(sys, "argv", ["kanban_bridge", *_cli_args()])

    assert kb.main() == 0
    assert not started, (
        "the scheduled drain must not start a poller thread — it IS the handler"
    )


# ---------------------------------------------------------------------------
# Hidden window.
#
# The task fires every 5 minutes under LogonType=InteractiveToken, so a console
# launcher action creates a real cmd.exe window in the user's session: it pops
# up, steals focus from whatever is being typed, and vanishes.  Task Scheduler
# has NO "run hidden" flag for a console action, so the only way to keep the
# window off the desktop is to launch the .cmd through a script host that asks
# Windows to create the process with a hidden window.
# ---------------------------------------------------------------------------


def _wrapper_text() -> str:
    assert HIDDEN_WRAPPER.is_file(), f"hidden wrapper is missing: {HIDDEN_WRAPPER}"
    return HIDDEN_WRAPPER.read_text(encoding="utf-8", errors="replace")


def test_hidden_wrapper_runs_the_launcher_with_window_style_zero():
    """``Run(cmd, 0, ...)`` is the ONLY thing that hides the console."""
    text = _wrapper_text()
    # Match the ASSIGNMENT, not the prose in the header comment (which also
    # mentions ``Run(..., 0, ...)``) — an earlier loose pattern matched the
    # comment and asserted against the wrong text.
    match = re.search(r"rc\s*=\s*shell\.Run\(([^)]*)\)", text)
    assert match, "wrapper must call WScript.Shell.Run and capture its result"
    args = [part.strip() for part in match.group(1).split(",")]
    assert len(args) >= 2, f"Run() needs a window style, got {args!r}"
    # Position 1 is intWindowStyle; 0 == hidden.
    assert args[1] == "HIDDEN_WINDOW", (
        f"window style must be the named HIDDEN_WINDOW constant, got {args[1]!r}"
    )
    assert re.search(r"^\s*Const HIDDEN_WINDOW\s*=\s*0\b", text, re.M), (
        "HIDDEN_WINDOW must be 0 — that is SW_HIDE; 1 would SHOW the window"
    )


def test_hidden_wrapper_waits_so_the_exit_code_survives():
    """Detaching immediately would report 0 for every tick, hiding failures."""
    text = _wrapper_text()
    assert re.search(r"^\s*Const WAIT_FOR_LAUNCHER\s*=\s*True\b", text, re.M), (
        "bWaitOnReturn must be True, otherwise Task Scheduler's Last Result is "
        "always 0 and a broken drain looks healthy"
    )
    assert "WScript.Quit rc" in text, (
        "the wrapper must propagate the launcher's exit code"
    )


def test_hidden_wrapper_points_at_the_real_launcher_absolutely():
    """A task gets a different PATH and cwd, so the default must be absolute."""
    text = _wrapper_text()
    match = re.search(r'Const DEFAULT_LAUNCHER\s*=\s*"([^"]+)"', text)
    assert match, "DEFAULT_LAUNCHER must be declared"
    launcher = PureWindowsPath(match.group(1))
    assert launcher.is_absolute(), f"launcher path must be absolute: {launcher}"
    assert launcher == SCRIPT, (
        f"wrapper must default to the drain launcher ({SCRIPT}), got {launcher}"
    )


def test_task_definition_runs_the_wrapper_not_the_cmd_directly():
    """Pin the registered action: ``wscript.exe`` + the wrapper, never the .cmd.

    The .cmd action is what showed the console; a future re-registration from
    a stale XML would bring the flashing window straight back.
    """
    assert TASK_XML.is_file(), f"task definition is missing: {TASK_XML}"
    text = TASK_XML.read_text(encoding="utf-8", errors="replace")

    command = re.search(r"<Command>([^<]+)</Command>", text)
    arguments = re.search(r"<Arguments>([^<]+)</Arguments>", text)
    assert command, "task XML must declare a <Command>"
    assert arguments, "task XML must declare <Arguments>"

    assert command.group(1).lower() == "wscript.exe", (
        f"task action must be wscript.exe, got {command.group(1)!r} — pointing "
        "it at the .cmd is what shows the console window"
    )
    assert "run_hidden.vbs" in arguments.group(1), (
        f"task arguments must invoke the hidden wrapper, got {arguments.group(1)!r}"
    )
    assert SCRIPT.name not in arguments.group(1), (
        "the .cmd must be reached THROUGH the wrapper, not directly"
    )
    assert "//B" in arguments.group(1), (
        "//B suppresses script-error dialogs; a modal box from a hidden script "
        "would block the task forever with nothing on screen to explain it"
    )
    # The schedule is what makes this a repeated focus-steal; keep it pinned.
    assert re.search(r"<Interval>PT5M</Interval>", text), "schedule must stay 5 min"

