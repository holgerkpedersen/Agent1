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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "run_kanban_queue_drain.cmd"


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
    assert Path(match.group(1)).is_absolute(), (
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
