"""Regression tests for `review` argument handling (agent_core/commands/review_cmd.py).

Two real defects are pinned here:

1. QUOTES.  The REPL tokenizes input with
   ``shlex.split(user_input, posix=False)`` (agent.py), which KEEPS the
   literal quotes on a quoted value.  ``review_cmd`` used to consume the raw
   tokens, so ``review refresh --trace-dir "other traces"`` looked for a
   directory literally named ``"other traces"`` (quotes included) and
   ``--note "two words"`` stored the note WITH its quotes.  Repo convention
   (analyze/fix/implement/jev commands) is to strip them in the command.

2. --note ON THE AUTO PATH.  ``review auto --note "..."`` (and
   ``review label <task> auto --note "..."``) silently dropped the note:
   ``_cmd_auto`` never parsed the flag and passed ``note=""``.
"""

import asyncio

import pytest

from agent_core.commands.review_cmd import ReviewCommand
from harnessfix.review import REVIEWS_RELPATH, ReviewRecord, load_reviews, save_reviews
from harnessfix.tracing import TraceWriter


class _Agent:
    """Minimal stand-in for Agent: the command only needs ``workspace``."""

    def __init__(self, workspace):
        self.workspace = str(workspace)


def _failed_trace(directory, task_id: str) -> None:
    writer = TraceWriter(task_id=task_id, directory=directory)
    writer.emit({"kind": "task_begin", "layer": "context",
                 "user_input": "fix the flaky gate",
                 "model": "qwen3.8-27b", "profile": "deep-analysis"})
    writer.emit({"kind": "tool_call", "layer": "tool_interface",
                 "tool": "write", "args_hash": "w"})
    writer.emit({"kind": "tool_error", "layer": "tool_interface",
                 "exception": "PermissionError", "message": "denied"})
    writer.emit({"kind": "loop_end", "layer": "lifecycle", "outcome": "error"})
    writer.close()


def _ledger(ws):
    """The review ledger path the command actually writes."""
    return ws / REVIEWS_RELPATH


@pytest.fixture()
def ws(tmp_path):
    (tmp_path / "reports" / "traces").mkdir(parents=True)
    return tmp_path


def _run(cmd, args, agent):
    return asyncio.run(cmd.execute(args, agent))


# ── 1. quotes must be stripped ──────────────────────────────────────────

def test_refresh_trace_dir_with_spaces_needs_unquoted_token(ws, capsys):
    """The REPL hands us '"other traces"' — the command must strip the quotes
    and find the real directory."""
    other = ws / "other traces"
    other.mkdir()
    _failed_trace(other, "tspace")
    agent = _Agent(ws)

    _run(ReviewCommand(), ["refresh", "--trace-dir", '"other traces"'], agent)

    out = capsys.readouterr().out
    assert "Reviewed 1 failed task(s)" in out
    assert '"other traces"' not in out
    assert "tspace" in load_reviews(_ledger(ws))


def test_refresh_trace_dir_single_quotes_also_stripped(ws, capsys):
    other = ws / "other traces"
    other.mkdir()
    _failed_trace(other, "tspace")
    agent = _Agent(ws)

    _run(ReviewCommand(), ["refresh", "--trace-dir", "'other traces'"], agent)

    assert "Reviewed 1 failed task(s)" in capsys.readouterr().out


def test_refresh_missing_flag_value_errors_instead_of_using_default(ws, capsys):
    """`review refresh --trace-dir` (no value) used to silently scan the
    DEFAULT dir and report success."""
    _failed_trace(ws / "reports" / "traces", "tdefault")
    agent = _Agent(ws)

    _run(ReviewCommand(), ["refresh", "--trace-dir"], agent)

    out = capsys.readouterr().out
    assert "Error:" in out and "--trace-dir" in out
    assert "Reviewed" not in out


def test_label_note_quotes_stripped(ws, capsys):
    _failed_trace(ws / "reports" / "traces", "tnote")
    agent = _Agent(ws)
    _run(ReviewCommand(), ["refresh"], agent)

    _run(ReviewCommand(), ["label", "tnote", "bug", "--note", '"two words"'], agent)

    rec = load_reviews(_ledger(ws))["tnote"]
    assert rec.note == "two words"
    assert rec.disposition == "bug"


def test_label_note_equals_form_supported(ws, capsys):
    """`--note=...` used to be silently ignored (no note stored at all)."""
    _failed_trace(ws / "reports" / "traces", "tnote")
    agent = _Agent(ws)
    _run(ReviewCommand(), ["refresh"], agent)

    _run(ReviewCommand(), ["label", "tnote", "noise", "--note=quoted note"], agent)

    assert load_reviews(_ledger(ws))["tnote"].note == "quoted note"


# ── 2. --note must survive the auto path ────────────────────────────────

def test_auto_all_keeps_note(ws, capsys):
    """`review auto --note "..."` used to drop the note entirely."""
    _failed_trace(ws / "reports" / "traces", "tauto")
    agent = _Agent(ws)
    _run(ReviewCommand(), ["refresh"], agent)

    _run(ReviewCommand(), ["auto", "--note", '"human prose"'], agent)

    rec = load_reviews(_ledger(ws))["tauto"]
    assert rec.note == "human prose"
    assert rec.source == "agent"


def test_auto_single_task_keeps_note(ws, capsys):
    _failed_trace(ws / "reports" / "traces", "tauto")
    agent = _Agent(ws)
    _run(ReviewCommand(), ["refresh"], agent)

    _run(ReviewCommand(), ["auto", "tauto", "--note", "single note"], agent)

    assert load_reviews(_ledger(ws))["tauto"].note == "single note"


def test_label_auto_keeps_note(ws, capsys):
    """`review label <task> auto --note "..."` (documented in the usage
    string) used to drop the note too."""
    _failed_trace(ws / "reports" / "traces", "tauto")
    agent = _Agent(ws)
    _run(ReviewCommand(), ["refresh"], agent)

    _run(ReviewCommand(), ["label", "tauto", "auto", "--note", '"kept note"'], agent)

    assert load_reviews(_ledger(ws))["tauto"].note == "kept note"


def test_auto_without_note_still_uses_verdict_note(ws, capsys):
    """No --note must keep the agent verdict's own note (no regression)."""
    _failed_trace(ws / "reports" / "traces", "tauto")
    agent = _Agent(ws)
    _run(ReviewCommand(), ["refresh"], agent)

    _run(ReviewCommand(), ["auto", "tauto"], agent)

    rec = load_reviews(_ledger(ws))["tauto"]
    assert rec.note  # verdict note, not empty
    assert rec.is_labeled()


# ── 3. quoted task ids ──────────────────────────────────────────────────

def test_show_and_export_accept_quoted_task_id(ws, capsys):
    _failed_trace(ws / "reports" / "traces", "tq")
    agent = _Agent(ws)
    _run(ReviewCommand(), ["refresh"], agent)
    _run(ReviewCommand(), ["label", "tq", "bug"], agent)

    _run(ReviewCommand(), ["show", '"tq"'], agent)
    assert "task_id: tq" in capsys.readouterr().out

    _run(ReviewCommand(), ["export", '"tq"'], agent)
    assert "Exported" in capsys.readouterr().out


def test_label_quoted_task_id(ws, capsys):
    _failed_trace(ws / "reports" / "traces", "tq")
    agent = _Agent(ws)
    _run(ReviewCommand(), ["refresh"], agent)

    _run(ReviewCommand(), ["label", '"tq"', "ok"], agent)

    assert load_reviews(_ledger(ws))["tq"].disposition == "ok"


def test_existing_ledger_roundtrip_unaffected(ws):
    """Sanity: the helper changes must not break plain records."""
    reviews = {"t": ReviewRecord(task_id="t", disposition="bug", note="n")}
    save_reviews(reviews, _ledger(ws))
    assert load_reviews(_ledger(ws))["t"].note == "n"
