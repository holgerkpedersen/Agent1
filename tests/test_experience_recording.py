"""Regression tests for LLM-decision experience recording + runtime-state isolation.

Two related contracts are pinned here:

1. ``Agent._record_llm_experience`` persists one row per traced LLM decision
   into the SQLite ``experiences`` table that the Memory MCP server reads
   (``mcp_servers/memory.py`` ``record_experience``).  The write must be
   byte/schema-compatible with that server, and must NEVER raise (decision
   #014: it is harness-layer observability only, never a behavioural lever).

2. ``chat_nlp`` records exactly one experience per ``run()`` invocation, with
   the outcome/success derived from the loop verdict and provider errors.

Both used to write into the LIVE repo files (``agent_memory.json`` /
``agent_memory.db``) during test runs, because ``chat_nlp`` tests sandboxed
only ``CHAT_HISTORY_JSON_PATH``.  A single suite run appended 21 phantom
``llm_decision`` rows to the live DB.  ``conftest._isolate_from_real_tree_and_beacons``
now redirects the runtime-state paths; the last class pins that redirection so
it can never silently regress.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

import agent
import agent_core.constants as constants
from agent import Agent

#: The experiences table exactly as the Memory MCP server expects it
#: (column order matters for byte-compatibility of the DDL).
_MCP_EXPERIENCE_COLUMNS = [
    ("timestamp", "TEXT"),
    ("action", "TEXT"),
    ("outcome", "REAL"),
    ("context", "TEXT"),
    ("success", "INTEGER"),
]


def _db_path_for(memory_json: Path) -> Path:
    """The DB path ``_record_llm_experience`` derives from the JSON path."""
    return Path(str(memory_json).replace(".json", ".db"))


def _rows(db: Path) -> list[tuple]:
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(
            "SELECT timestamp, action, outcome, context, success FROM experiences"
        ).fetchall()
    finally:
        conn.close()


class TestRecordLlmExperienceDirect:
    """The helper writes MCP-compatible rows and never raises."""

    def test_inserts_row_and_returns_rowid(self, tmp_path: Path, monkeypatch) -> None:
        mem = tmp_path / "agent_memory.json"
        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem))
        bot = Agent(workspace=str(tmp_path))

        rowid = bot._record_llm_experience(
            action="llm_decision", outcome=1.0,
            context={"model": "m", "verdict": "answer"}, success=True,
        )

        assert rowid == 1
        db = _db_path_for(mem)
        assert db.exists()
        ts, action, outcome, ctx, success = _rows(db)[0]
        assert action == "llm_decision"
        assert outcome == 1.0
        assert success == 1
        assert json.loads(ctx) == {"model": "m", "verdict": "answer"}
        # Timestamp is the ISO-ish form the rest of the memory DB uses.
        assert len(ts) == 19 and ts[10] == "T"

    def test_table_schema_matches_mcp_memory_server(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """The created DDL must match the table the MCP server reads/writes."""
        mem = tmp_path / "agent_memory.json"
        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem))
        bot = Agent(workspace=str(tmp_path))
        bot._record_llm_experience(action="llm_decision", outcome=1.0)

        conn = sqlite3.connect(str(_db_path_for(mem)))
        try:
            info = conn.execute("PRAGMA table_info(experiences)").fetchall()
        finally:
            conn.close()
        # (cid, name, type, notnull, dflt_value, pk) -> name/type, in order.
        assert [(r[1], r[2]) for r in info] == _MCP_EXPERIENCE_COLUMNS

    def test_pre_existing_table_is_reused_not_clobbered(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """A DB already holding MCP-written rows must gain an append, not a reset."""
        mem = tmp_path / "agent_memory.json"
        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem))
        db = _db_path_for(mem)
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE experiences (timestamp TEXT, action TEXT, outcome REAL, "
            "context TEXT, success INTEGER)"
        )
        conn.execute(
            "INSERT INTO experiences VALUES (?, ?, ?, ?, ?)",
            ("2026-01-01T00:00:00", "mcp_written", 0.7, "{}", 1),
        )
        conn.commit()
        conn.close()

        bot = Agent(workspace=str(tmp_path))
        bot._record_llm_experience(action="llm_decision", outcome=1.0)

        rows = _rows(db)
        assert len(rows) == 2
        assert rows[0][1] == "mcp_written"
        assert rows[1][1] == "llm_decision"

    def test_out_of_range_outcome_is_rejected_without_writing(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        mem = tmp_path / "agent_memory.json"
        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem))
        bot = Agent(workspace=str(tmp_path))

        assert bot._record_llm_experience(action="a", outcome=1.5) is None
        assert bot._record_llm_experience(action="a", outcome=-0.1) is None
        assert not _db_path_for(mem).exists()

    def test_success_defaults_from_outcome(self, tmp_path: Path, monkeypatch) -> None:
        mem = tmp_path / "agent_memory.json"
        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem))
        bot = Agent(workspace=str(tmp_path))

        bot._record_llm_experience(action="low", outcome=0.2)
        bot._record_llm_experience(action="high", outcome=0.5)

        rows = _rows(_db_path_for(mem))
        assert [(r[1], r[4]) for r in rows] == [("low", 0), ("high", 1)]

    def test_context_is_sorted_json_and_defaults_to_empty_object(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        mem = tmp_path / "agent_memory.json"
        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem))
        bot = Agent(workspace=str(tmp_path))

        bot._record_llm_experience(action="a", outcome=1.0, context={"b": 1, "a": 2})
        bot._record_llm_experience(action="b", outcome=1.0)

        rows = _rows(_db_path_for(mem))
        assert rows[0][3] == '{"a": 2, "b": 1}'
        assert rows[1][3] == "{}"

    def test_never_raises_on_unwritable_location(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """A bad path must silently no-op (decision #014 / #048 contract)."""
        # Occupy the DERIVED db path with a directory: sqlite3.connect then
        # fails with "unable to open database file".  (Blocking the .json path
        # would not help — the DB path is derived by string replacement.)
        mem = tmp_path / "agent_memory.json"
        _db_path_for(mem).mkdir()
        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem))
        bot = Agent(workspace=str(tmp_path))

        assert bot._record_llm_experience(action="a", outcome=1.0) is None

    def test_never_raises_on_non_numeric_outcome(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        mem = tmp_path / "agent_memory.json"
        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem))
        bot = Agent(workspace=str(tmp_path))

        assert bot._record_llm_experience(action="a", outcome="bad") is None  # type: ignore[arg-type]
        assert not _db_path_for(mem).exists()


class _FakeLLM:
    """Minimal provider stub: returns a scripted reply, never touches network."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    async def chat(self, messages, tools=None, **kwargs):
        self.calls += 1
        return self.reply


class TestChatNlpRecordsExperience:
    """``chat_nlp`` records the run verdict through the REAL ToolLoopRunner."""

    def _run_turn(self, tmp_path: Path, reply: str) -> tuple[Agent, Path]:
        mem = tmp_path / "agent_memory.json"
        hist = tmp_path / "chat_history.json"
        with patch("agent.AGENT_MEMORY_JSON_PATH", str(mem)), \
             patch("agent.CHAT_HISTORY_JSON_PATH", str(hist)):
            bot = Agent(workspace=str(tmp_path))
            bot.llm = _FakeLLM(reply)
            asyncio.run(bot.chat_nlp("hello"))
        return bot, _db_path_for(mem)

    def test_completed_answer_records_success_row(self, tmp_path: Path) -> None:
        # "All done" is a strong completion marker -> no auto-continue, so
        # exactly one run (and one row) happens deterministically.
        bot, db = self._run_turn(tmp_path, "All done.")
        print(f"# model={bot.model_name} db={db.name}")

        rows = _rows(db)
        assert len(rows) == 1
        _ts, action, outcome, ctx, success = rows[0]
        assert action == "llm_decision"
        assert outcome == 1.0
        assert success == 1
        payload = json.loads(ctx)
        assert payload["verdict"] == "answer"
        assert payload["continuations"] == 0
        assert payload["model"] == bot.model_name
        assert payload["final_text_len"] == len("All done.")

    def test_provider_error_records_failure_row(self, tmp_path: Path) -> None:
        """An "[Error..." reply is a provider failure, not an answer."""
        _bot, db = self._run_turn(tmp_path, "[Error] provider exploded")

        rows = _rows(db)
        assert len(rows) == 1
        _ts, action, outcome, ctx, success = rows[0]
        assert action == "llm_decision"
        assert outcome == 0.0
        assert success == 0
        assert json.loads(ctx)["verdict"] in ("answer", "cap", "stuck", "no_progress")

    def test_context_records_the_model_that_answered(self, tmp_path: Path) -> None:
        """The row must be self-describing for cross-model comparison."""
        bot, db = self._run_turn(tmp_path, "All done.")
        payload = json.loads(_rows(db)[0][3])
        assert payload["model"] == bot.model_name
        assert "profile" in payload


class TestRuntimeStateIsolation:
    """The suite must never write the LIVE memory DB / history of the checkout.

    These tests fail if ``conftest._isolate_from_real_tree_and_beacons`` stops
    redirecting the runtime-state paths (the bug that appended 21 phantom
    ``llm_decision`` rows to the real ``agent_memory.db``).
    """

    def test_module_paths_are_redirected_away_from_the_live_files(self) -> None:
        repo_root = Path(agent.__file__).resolve().parent

        live_history = Path(constants.CHAT_HISTORY_JSON_PATH).resolve()
        live_memory = Path(constants.AGENT_MEMORY_JSON_PATH).resolve()
        for live in (live_history, live_memory, _db_path_for(live_memory)):
            assert repo_root in live.parents, (
                f"precondition: {live} is expected to be the live in-repo path"
            )

        for patched, live in (
            (Path(agent.CHAT_HISTORY_JSON_PATH).resolve(), live_history),
            (Path(agent.AGENT_MEMORY_JSON_PATH).resolve(), live_memory),
        ):
            assert patched != live, (
                "runtime-state path was NOT redirected during tests; the suite "
                "would pollute the live memory/history files"
            )
            assert repo_root not in patched.parents, (
                f"runtime-state path {patched} still lives inside the repo"
            )

    def test_derived_db_path_is_also_isolated(self) -> None:
        """The DB is derived from the JSON path, so redirecting it must cover both."""
        repo_root = Path(agent.__file__).resolve().parent
        db = _db_path_for(Path(agent.AGENT_MEMORY_JSON_PATH))
        assert db.name == "agent_memory.db"
        assert db != _db_path_for(Path(constants.AGENT_MEMORY_JSON_PATH))
        assert repo_root not in db.resolve().parents

    def test_memory_path_is_resolved_at_call_time(self, tmp_path: Path) -> None:
        """A patched ``AGENT_MEMORY_JSON_PATH`` must decide where the row lands.

        ``_record_llm_experience`` derives the DB from the module global on
        every call, NOT at import time.  This is what makes the conftest
        redirection (and every test that sandboxes the path itself) effective;
        if the path were captured at import, rows would go to the live DB no
        matter what a test patches.
        """
        live_db = _db_path_for(Path(constants.AGENT_MEMORY_JSON_PATH))
        before = live_db.stat().st_mtime_ns if live_db.exists() else None

        sandbox_mem = tmp_path / "nested" / "agent_memory.json"
        sandbox_mem.parent.mkdir(parents=True)
        bot = Agent(workspace=str(tmp_path))
        with patch("agent.AGENT_MEMORY_JSON_PATH", str(sandbox_mem)):
            assert bot._record_llm_experience(action="call_time", outcome=1.0) == 1

        sandbox_db = _db_path_for(sandbox_mem)
        assert sandbox_db.exists(), (
            "row did not follow the patched AGENT_MEMORY_JSON_PATH - the DB path "
            "is not resolved at call time"
        )
        assert _rows(sandbox_db)[0][1] == "call_time"

        if before is None:
            assert not live_db.exists(), "test run created the LIVE agent_memory.db"
        else:
            assert live_db.stat().st_mtime_ns == before, (
                "test run modified the LIVE agent_memory.db"
            )


@pytest.mark.parametrize(
    "outcome,expected_success",
    [(0.0, 0), (0.49, 0), (0.5, 1), (1.0, 1)],
)
def test_success_threshold_boundaries(
    tmp_path: Path, monkeypatch, outcome: float, expected_success: int,
) -> None:
    mem = tmp_path / "agent_memory.json"
    monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem))
    bot = Agent(workspace=str(tmp_path))
    bot._record_llm_experience(action="a", outcome=outcome)
    assert _rows(_db_path_for(mem))[0][4] == expected_success
