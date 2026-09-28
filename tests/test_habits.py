"""Regression tests for the workspace habit feature (plan Part A, item C1).

Covers the five contracts of ``agent_core.habits`` plus its two wiring
points:

1. :func:`mine_habits` — promotion at ``min_count``, recency decay,
   corrections rendered as negative habits, per-category caps.
2. :func:`habits_block` — empty input renders nothing; a populated list
   renders ``HABITS_MARKER`` exactly once with at most 8 lines.
3. Prompt integration — the marker survives two ``_refresh_system_message``
   calls exactly once and strips back to ``_SYSTEM_PROMPT``
   (mirrors ``tests/test_vendored_skills.py:119``).
4. An empty workspace keeps the prompt byte-identical to the pre-habits
   baseline.
5. A corrupt ``.habits.json`` is quarantined to ``.habits.json.bad-*`` and
   :func:`load_habits` returns ``[]``.
6. The ``habits`` command: ``pin`` / ``forget`` / ``mine`` / ``off``.
7. The A3 turn hook: one ``_finish_turn`` call appends a
   ``turn_log.jsonl`` line and inserts an ``experiences`` row.

Every test writes only into ``tmp_path`` workspaces.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

import agent as agent_mod
from agent_core.commands.habits_cmd import HabitsCommand
from agent_core.config import AgentDisplayMode
from agent_core.habits import (
    HABITS_MARKER,
    MAX_BLOCK_LINES,
    MAX_PER_CATEGORY,
    habits_block,
    load_habits,
    mine_habits,
    save_habits,
)


def _u(text: str) -> dict:
    """One chat-history user message."""
    return {"role": "user", "content": text}


def _make_agent(workspace: Path, tmp_path: Path, monkeypatch) -> "agent_mod.Agent":
    """Build a real Agent whose state files land in *tmp_path*."""
    workspace.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        agent_mod, "CHAT_HISTORY_JSON_PATH", str(tmp_path / "chat_history.json")
    )
    monkeypatch.setattr(
        agent_mod, "AGENT_MEMORY_JSON_PATH", str(tmp_path / "agent_memory.json")
    )
    return agent_mod.Agent(workspace=str(workspace))


# ---------------------------------------------------------------------------
# 1. mine_habits: promotion / decay / corrections / caps
# ---------------------------------------------------------------------------

def test_miner_promotes_only_at_min_count() -> None:
    two = [_u("run the tests")] * 2
    three = [_u("run the tests")] * 3

    assert mine_habits(two, []) == [], "count 2 must not promote (min_count=3)"

    out = mine_habits(three, [])
    assert out, "count 3 must promote at the default min_count"
    assert out[0]["category"] == "command"
    assert out[0]["count"] == 3
    # An explicit higher bar suppresses the same candidate.
    assert mine_habits(three, [], min_count=4) == []


def test_miner_applies_recency_decay() -> None:
    filler = [_u("thanks")] * 8  # neutral text: classified by nothing
    old_first = [_u("run the tests")] * 3 + filler
    recent_last = filler + [_u("run the tests")] * 3

    old = next(h for h in mine_habits(old_first, []) if h["category"] == "command")
    recent = next(h for h in mine_habits(recent_last, []) if h["category"] == "command")

    assert old["count"] == recent["count"] == 3
    # Identical raw counts, but occurrences far back in the window weigh less.
    assert old["score"] < recent["score"]
    assert old["score"] < old["count"], "aged occurrences must decay below raw count"


def test_miner_renders_corrections_as_negative_habits() -> None:
    history = [_u("no, use the other function")] * 3

    out = mine_habits(history, [])
    corrections = [h for h in out if h["category"] == "corrections"]
    assert corrections, "repeated corrections must promote"
    assert corrections[0]["text"].startswith("avoid ")
    assert corrections[0]["count"] == 3


def test_miner_caps_habits_per_category() -> None:
    # Four distinct file paths, each repeated 3x; the OLDEST group (omega)
    # has the lowest decayed score and must be the one dropped by the cap.
    order = (
        ["src/omega.py"] * 3
        + ["src/alpha.py"] * 3
        + ["src/beta.py"] * 3
        + ["src/gamma.py"] * 3
    )
    history = [_u(f"check {p}") for p in order]

    out = mine_habits(history, [])
    files = [h for h in out if h["category"] == "files"]
    assert len(files) == MAX_PER_CATEGORY, "files category must be capped at 3"
    texts = {h["text"] for h in files}
    assert "often mentions src/omega.py" not in texts, "lowest-scored must be dropped"
    for kept in ("src/alpha.py", "src/beta.py", "src/gamma.py"):
        assert f"often mentions {kept}" in texts


# ---------------------------------------------------------------------------
# 2. habits_block: empty stays empty, populated renders the marker once
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("empty", [None, "", []])
def test_habits_block_empty_inputs_render_nothing(empty) -> None:
    assert habits_block(empty) == ""


def test_habits_block_renders_marker_once_and_caps_lines() -> None:
    habits = [
        {"text": f"preference number {i}", "count": i, "category": "style"}
        for i in range(1, 13)
    ]
    block = habits_block(habits)

    assert block.startswith(HABITS_MARKER)
    assert block.count("USER HABITS") == 1
    rendered = [ln for ln in block.splitlines() if ln.startswith("- ")]
    assert len(rendered) == MAX_BLOCK_LINES, "block must hard-cap at 8 lines"
    assert "(seen " in rendered[0]


# ---------------------------------------------------------------------------
# 3. Prompt integration: rebuilt every turn, strips back to the base prompt
# ---------------------------------------------------------------------------

def test_habits_marker_survives_double_refresh(
    tmp_path: Path, monkeypatch
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    save_habits(
        ws, [{"text": "keep answers short", "count": 5, "category": "style"}]
    )

    bot = _make_agent(ws, tmp_path, monkeypatch)
    bot._refresh_system_message()
    first = bot._chat_history[0]["content"]
    bot._refresh_system_message()
    second = bot._chat_history[0]["content"]

    assert HABITS_MARKER in first, "habits block missing from the system prompt"
    assert second == first, "dynamic blocks accumulated across refreshes"
    assert second.count("USER HABITS") == 1, "marker must appear exactly once"
    assert (
        agent_mod._strip_dynamic_system_blocks(second) == agent_mod._SYSTEM_PROMPT
    ), "base prompt must survive stripping with no habits leakage"


def test_empty_workspace_prompt_stays_byte_identical(
    tmp_path: Path, monkeypatch
) -> None:
    ws = tmp_path / "ws_empty"
    assert not (ws / ".habits.json").exists()

    bot = _make_agent(ws, tmp_path, monkeypatch)
    bot._refresh_system_message()
    content = bot._chat_history[0]["content"]

    assert HABITS_MARKER not in content
    assert "USER HABITS" not in content
    assert content == agent_mod._SYSTEM_PROMPT, (
        "an empty-workspace prompt must stay byte-identical to the pre-habits"
        " baseline"
    )


# ---------------------------------------------------------------------------
# 5. Corrupt ledger quarantine
# ---------------------------------------------------------------------------

def test_corrupt_habits_ledger_is_quarantined(tmp_path: Path) -> None:
    ws = tmp_path / "ws_corrupt"
    ws.mkdir()
    (ws / ".habits.json").write_bytes(b"{ this is not json")

    assert load_habits(ws) == [], "corrupt ledger must load as empty, never raise"

    quarantined = list(ws.glob(".habits.json.bad-*"))
    assert quarantined, "corrupt bytes must be preserved in .habits.json.bad-*"
    assert quarantined[0].read_bytes() == b"{ this is not json"
    assert not (ws / ".habits.json").exists(), "corrupt file must be moved aside"


# ---------------------------------------------------------------------------
# 6. The habits command: pin / forget / mine / off
# ---------------------------------------------------------------------------

def _run(args: list[str], bot) -> None:
    asyncio.run(HabitsCommand().execute(args, bot))


def test_habits_command_pin_and_forget(tmp_path: Path, monkeypatch) -> None:
    ws = tmp_path / "ws_cmd"
    bot = _make_agent(ws, tmp_path, monkeypatch)

    _run(["pin", "always use pytest -q"], bot)
    _run(["pin", "prefer short answers"], bot)

    habits = load_habits(ws)
    assert [h["text"] for h in habits] == ["prefer short answers", "always use pytest -q"]
    assert all(h["category"] == "pinned" for h in habits)

    _run(["forget", "0"], bot)
    habits = load_habits(ws)
    assert [h["text"] for h in habits] == ["always use pytest -q"]

    _run(["forget", "99"], bot)  # out of range: prints usage, changes nothing
    assert len(load_habits(ws)) == 1


def test_habits_command_mine_keeps_pinned(tmp_path: Path, monkeypatch) -> None:
    ws = tmp_path / "ws_mine"
    bot = _make_agent(ws, tmp_path, monkeypatch)

    _run(["pin", "always use pytest -q"], bot)
    bot._chat_history = [_u("run the tests")] * 3

    _run(["mine"], bot)

    habits = load_habits(ws)
    texts = [h["text"] for h in habits]
    assert texts[0] == "always use pytest -q", "pinned habits must survive mining"
    assert "run the test suite (pytest)" in texts
    mined = next(h for h in habits if h["text"] == "run the test suite (pytest)")
    assert mined["count"] == 3


def test_habits_command_off_suppresses_ledger(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    ws = tmp_path / "ws_off"
    bot = _make_agent(ws, tmp_path, monkeypatch)
    save_habits(ws, [{"text": "keep answers short", "count": 3, "category": "style"}])
    assert load_habits(ws), "sanity: ledger loads before switching off"

    _run(["off"], bot)
    capsys.readouterr()
    assert load_habits(ws) == [], "'habits off' must suppress the ledger"
    # The habits are not deleted — only the preference hides them.
    assert (ws / ".habits.json").exists()

    _run(["on"], bot)
    capsys.readouterr()
    assert load_habits(ws), "'habits on' must restore the ledger"


# ---------------------------------------------------------------------------
# 7. The A3 turn hook: turn_log.jsonl line + experiences row
# ---------------------------------------------------------------------------

def test_finish_turn_writes_turn_log_and_experience_row(
    tmp_path: Path, monkeypatch
) -> None:
    from agent_core.monitoring import metrics_file

    ws = tmp_path / "ws_turn"
    bot = _make_agent(ws, tmp_path, monkeypatch)
    # The turn log must land in tmp_path, not the live state dir.
    monkeypatch.setattr(
        agent_mod, "TURN_LOG_PATH", str(tmp_path / "turn_log.jsonl")
    )
    # Isolate the two sibling observability hooks (covered by C2 tests):
    # avoid mutating global METRICS state and appending to the repo-root
    # .metrics_events.jsonl during this run.
    monkeypatch.setattr(agent_mod, "record_turn_outcome", lambda **kwargs: None)
    quality_events: list = []
    monkeypatch.setattr(
        metrics_file,
        "append_event",
        lambda kind, name, value: quality_events.append((kind, name, value)),
    )

    bot._chat_history = [_u("make it work"), {"role": "assistant", "content": "done."}]
    if not hasattr(bot, "_turn_start_index"):
        bot._turn_start_index = 0
    bot._last_user_input = "make it work"
    bot._turn_started_at = datetime.now()

    bot._finish_turn("All finished.", None, object(), AgentDisplayMode.QUIET)

    # (a) exactly one bounded turn-log line with the A3 record shape
    log = tmp_path / "turn_log.jsonl"
    assert log.exists(), "turn_log.jsonl was not written"
    lines = [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1, f"expected exactly 1 turn-log line, got {len(lines)}"
    rec = json.loads(lines[0])
    assert rec["user_input_tail"] == "make it work"
    assert rec["llm_error"] is False
    assert rec["mutated_files"] == []
    assert rec["duration"] is None or rec["duration"] >= 0

    # (b) the experiences row, asserted through real sqlite3
    db = tmp_path / "agent_memory.db"
    assert db.exists(), "experiences DB was not created"
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            "SELECT action, outcome, success FROM experiences"
        ).fetchall()
    finally:
        conn.close()
    assert rows, "no experiences row inserted by _finish_turn"
    action, outcome, success = rows[0]
    assert action == "chat_turn"
    assert outcome == pytest.approx(0.7), "clean turn with no mutations scores 0.7"
    assert success == 1
