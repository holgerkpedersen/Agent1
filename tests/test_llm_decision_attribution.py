"""Regression tests for LLM decision attribution in the memory store.

Defects/features these guard
----------------------------
``experiences`` recorded *what* happened but not *which model decided it*, so
a model swap could not be measured against ground-truth memory outcomes.

The implementation (``agent_core/memory/attribution.py`` +
``Agent._record_llm_experience``) must satisfy, permanently:

1. The MCP memory server's five columns stay byte-identical AND FIRST; the
   attribution columns are APPENDED and nullable, so a pre-existing
   (MCP-authored) table is extended in place, never clobbered or reordered.
2. Every attributed row is stamped with the deciding model, and appends exactly
   ONE ``llm_decisions`` provenance row — the invariant is exact.
3. A row that cannot be attributed (no model identity) records NO author and
   writes NO provenance row, instead of inventing one.
4. ``prompt_sha256`` is canonical and tag-insensitive, so a transient
   loop-injected tag cannot change the identity of the prompt answered.
5. The whole path NEVER raises — a bad/locked DB degrades to a silent no-op.
6. The analytics helpers answer "which model's memories are better?".
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

import agent
from agent import Agent
from agent_core.memory import (
    ATTRIBUTION_COLUMNS,
    EXPERIENCE_COLUMNS,
    attribution_summary,
    decision_latency_histogram,
    ensure_attribution_schema,
    experiences_by_llm,
    prompt_sha256,
    resolve_decision_llm,
    success_rate_by_model,
)

#: The experiences columns exactly as the Memory MCP server expects them.
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


def _connect(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(db))


def _columns(db: Path, table: str = "experiences") -> list[tuple[str, str]]:
    conn = _connect(db)
    try:
        return [(r[1], r[2]) for r in conn.execute(f"PRAGMA table_info({table})")]
    finally:
        conn.close()


def _attributed_rows(db: Path) -> list[tuple]:
    """(action, decision_llm, decision_prompt_hash, decision_timestamp) rows."""
    conn = _connect(db)
    try:
        return conn.execute(
            "SELECT action, decision_llm, decision_prompt_hash, decision_timestamp "
            "FROM experiences ORDER BY rowid ASC"
        ).fetchall()
    finally:
        conn.close()


def _decisions(db: Path) -> list[tuple]:
    """(run_id, model_name, prompt_sha256, outcome_experience_id, latency, tokens)."""
    conn = _connect(db)
    try:
        return conn.execute(
            "SELECT run_id, model_name, prompt_sha256, outcome_experience_id, "
            "latency_ms, token_usage FROM llm_decisions ORDER BY id ASC"
        ).fetchall()
    finally:
        conn.close()


def _table_exists(db: Path, table: str) -> bool:
    conn = _connect(db)
    try:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone() is not None
    finally:
        conn.close()


# --------------------------------------------------------------------------
# prompt_sha256 — canonical, tag-insensitive identity of the decision input
# --------------------------------------------------------------------------

class TestPromptHash:
    def test_is_deterministic_and_hex_sha256(self) -> None:
        first = prompt_sha256("explain the loop")
        assert first == prompt_sha256("explain the loop")
        assert len(first) == 64
        assert all(c in "0123456789abcdef" for c in first)

    def test_distinguishes_different_prompts(self) -> None:
        assert prompt_sha256("a") != prompt_sha256("b")

    def test_message_list_is_canonical_across_key_order(self) -> None:
        """Same conversation, differently-built dicts -> same hash."""
        a = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
        b = [{"content": "hi", "role": "user"}, {"content": "yo", "role": "assistant"}]
        assert prompt_sha256(a) == prompt_sha256(b)

    def test_loop_injected_tags_do_not_change_the_hash(self) -> None:
        """A transient loop note tag must not forge a new prompt identity."""
        clean = [{"role": "user", "content": "do it"}]
        tagged = [{"role": "user", "content": "do it", "_continue_note_tag": "auto"}]
        assert prompt_sha256(clean) == prompt_sha256(tagged)

    def test_empty_prompt_differs_from_a_stringified_none(self) -> None:
        """The hash of "" must not be reused for a missing prompt.

        ``_record_llm_experience`` stores NULL for "no prompt"; it must never
        store the hash of the empty string, which would claim the empty prompt
        was the decision input.
        """
        assert prompt_sha256("") != prompt_sha256("None")

    def test_unserialisable_input_still_hashes(self) -> None:
        """Never raises on exotic input — falls back to ``str()``."""
        assert len(prompt_sha256(object())) == 64


# --------------------------------------------------------------------------
# Schema — additive, MCP-compatible, idempotent
# --------------------------------------------------------------------------

class TestAttributionSchema:
    def test_mcp_columns_are_first_and_appended_columns_are_nullable(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        mem = tmp_path / "agent_memory.json"
        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem))
        Agent(workspace=str(tmp_path))._record_llm_experience(
            action="llm_decision", outcome=1.0)

        cols = _columns(_db_path_for(mem))
        assert cols[:len(_MCP_EXPERIENCE_COLUMNS)] == _MCP_EXPERIENCE_COLUMNS
        assert [c[0] for c in cols[len(_MCP_EXPERIENCE_COLUMNS):]] == [
            name for name, _ in ATTRIBUTION_COLUMNS
        ]

        conn = _connect(_db_path_for(mem))
        try:
            info = conn.execute("PRAGMA table_info(experiences)").fetchall()
        finally:
            conn.close()
        for row in info[len(_MCP_EXPERIENCE_COLUMNS):]:
            assert row[3] == 0, f"{row[1]} must be nullable for pre-existing rows"

    def test_declared_column_tuples_match_the_created_table(self) -> None:
        """The exported column contract must not drift from the real DDL."""
        assert [n for n, _ in EXPERIENCE_COLUMNS] == [
            n for n, _ in _MCP_EXPERIENCE_COLUMNS
        ]

    def test_llm_decisions_table_holds_the_provenance_columns(
        self, tmp_path: Path
    ) -> None:
        db = tmp_path / "m.db"
        conn = _connect(db)
        try:
            ensure_attribution_schema(conn)
            conn.commit()
        finally:
            conn.close()
        assert [n for n, _ in _columns(db, "llm_decisions")] == [
            "id", "run_id", "model_name", "prompt_sha256",
            "outcome_experience_id", "latency_ms", "token_usage", "timestamp",
        ]

    def test_migration_is_idempotent_and_preserves_existing_rows(
        self, tmp_path: Path
    ) -> None:
        db = tmp_path / "m.db"
        conn = _connect(db)
        try:
            conn.execute(
                "CREATE TABLE experiences (timestamp TEXT, action TEXT, outcome REAL, "
                "context TEXT, success INTEGER)"
            )
            conn.execute(
                "INSERT INTO experiences VALUES (?, ?, ?, ?, ?)",
                ("2026-01-01T00:00:00", "mcp_written", 0.7, "{}", 1),
            )
            conn.commit()
            ensure_attribution_schema(conn)
            ensure_attribution_schema(conn)  # second call must be a no-op
            conn.commit()
            rows = conn.execute(
                "SELECT action, decision_llm, decision_prompt_hash FROM experiences"
            ).fetchall()
        finally:
            conn.close()
        # The MCP row survives untouched and reads back NULL for the new columns.
        assert rows == [("mcp_written", None, None)]

    def test_preexisting_agent_row_gains_attribution_columns(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """An existing agent-written DB is migrated in place, not reset."""
        mem = tmp_path / "agent_memory.json"
        db = _db_path_for(mem)
        conn = _connect(db)
        try:
            conn.execute(
                "CREATE TABLE experiences (timestamp TEXT, action TEXT, outcome REAL, "
                "context TEXT, success INTEGER)"
            )
            conn.execute(
                "INSERT INTO experiences VALUES (?, ?, ?, ?, ?)",
                ("2026-01-01T00:00:00", "old_row", 1.0, "{}", 1),
            )
            conn.commit()
        finally:
            conn.close()

        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem))
        Agent(workspace=str(tmp_path))._record_llm_experience(
            action="new_row", outcome=1.0)

        rows = _attributed_rows(db)
        assert rows[0] == ("old_row", None, None, None)
        assert rows[1][0] == "new_row"
        assert rows[1][1]  # the new row carries its deciding model


# --------------------------------------------------------------------------
# Stamping — the real write path
# --------------------------------------------------------------------------

class TestDecisionStamping:
    def test_row_is_stamped_with_the_agents_model(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        mem = tmp_path / "agent_memory.json"
        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem))
        bot = Agent(workspace=str(tmp_path))

        bot._record_llm_experience(action="llm_decision", outcome=1.0)

        action, decision_llm, _hash, ts = _attributed_rows(_db_path_for(mem))[0]
        assert action == "llm_decision"
        assert decision_llm == bot.model_name
        assert ts and len(ts) == 19 and ts[10] == "T"

    def test_explicit_decision_llm_overrides_the_agent_default(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        mem = tmp_path / "agent_memory.json"
        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem))
        bot = Agent(workspace=str(tmp_path))

        bot._record_llm_experience(
            action="a", outcome=1.0, decision_llm="some/other-model")

        assert _attributed_rows(_db_path_for(mem))[0][1] == "some/other-model"

    def test_prompt_hash_is_stored_and_matches_prompt_sha256(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        mem = tmp_path / "agent_memory.json"
        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem))
        bot = Agent(workspace=str(tmp_path))

        bot._record_llm_experience(
            action="a", outcome=1.0, decision_prompt="the question asked")

        assert _attributed_rows(_db_path_for(mem))[0][2] == prompt_sha256(
            "the question asked")

    def test_no_prompt_stores_null_hash_not_the_empty_string_hash(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        mem = tmp_path / "agent_memory.json"
        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem))
        bot = Agent(workspace=str(tmp_path))

        bot._record_llm_experience(action="a", outcome=1.0)

        stored = _attributed_rows(_db_path_for(mem))[0][2]
        assert stored is None
        assert stored != prompt_sha256("")

    def test_one_provenance_row_per_attributed_experience(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """The invariant: |llm_decisions| == |experiences with decision_llm|."""
        mem = tmp_path / "agent_memory.json"
        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem))
        bot = Agent(workspace=str(tmp_path))

        first = bot._record_llm_experience(
            action="llm_decision", outcome=1.0, decision_prompt="p1",
            run_id="run-1", latency_ms=12.5, token_usage=99)
        second = bot._record_llm_experience(
            action="chat_turn", outcome=0.7, run_id="run-1")

        decisions = _decisions(_db_path_for(mem))
        assert len(decisions) == 2
        assert decisions[0] == (
            "run-1", bot.model_name, prompt_sha256("p1"), first, 12.5, 99)
        assert decisions[1] == ("run-1", bot.model_name, None, second, None, None)

    def test_provenance_row_links_back_to_its_experience_rowid(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        mem = tmp_path / "agent_memory.json"
        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem))
        bot = Agent(workspace=str(tmp_path))

        rowid = bot._record_llm_experience(action="a", outcome=1.0)

        assert _decisions(_db_path_for(mem))[0][3] == rowid


class TestUnattributableRows:
    """No model identity -> record NO author and NO provenance row."""

    def test_missing_model_identity_leaves_columns_null(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        mem = tmp_path / "agent_memory.json"
        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem))
        bot = Agent(workspace=str(tmp_path))

        with patch("agent_core.memory.attribution.resolve_decision_llm",
                   return_value=None):
            rowid = bot._record_llm_experience(action="a", outcome=1.0)

        assert rowid is not None
        assert _attributed_rows(_db_path_for(mem))[0][1:] == (None, None, None)
        assert _decisions(_db_path_for(mem)) == []

    def test_unattributable_row_keeps_the_mcp_column_shape(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """An unattributed insert must still be readable by the MCP server."""
        mem = tmp_path / "agent_memory.json"
        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem))
        bot = Agent(workspace=str(tmp_path))

        with patch("agent_core.memory.attribution.resolve_decision_llm",
                   return_value=None):
            bot._record_llm_experience(
                action="a", outcome=0.9, context={"k": "v"}, success=True)

        conn = _connect(_db_path_for(mem))
        try:
            row = conn.execute(
                "SELECT timestamp, action, outcome, context, success "
                "FROM experiences"
            ).fetchone()
        finally:
            conn.close()
        assert row[1] == "a"
        assert row[2] == 0.9
        assert json.loads(row[3]) == {"k": "v"}
        assert row[4] == 1


# --------------------------------------------------------------------------
# resolve_decision_llm — identity fallback chain
# --------------------------------------------------------------------------

class TestResolveDecisionLlm:
    def test_prefers_the_agents_own_model_name(self) -> None:
        class _Stub:
            model_name = "agent/model"
            llm = type("L", (), {"model_name": "provider/model"})()

        assert resolve_decision_llm(_Stub()) == "agent/model"

    def test_falls_back_to_provider_model_name(self) -> None:
        class _Stub:
            model_name = None
            llm = type("L", (), {"model_name": "provider/model"})()

        assert resolve_decision_llm(_Stub()) == "provider/model"

    def test_returns_none_when_nothing_is_known(self) -> None:
        class _Stub:
            model_name = ""
            llm = None

        assert resolve_decision_llm(_Stub()) is None


# --------------------------------------------------------------------------
# Never raises — the observability contract
# --------------------------------------------------------------------------

class TestNeverRaises:
    def test_unwritable_db_path_is_a_silent_no_op(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """A directory in place of the DB file must not break the turn."""
        mem = tmp_path / "agent_memory.json"
        db = _db_path_for(mem)
        db.mkdir()  # sqlite cannot open a directory as a DB

        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem))
        bot = Agent(workspace=str(tmp_path))

        assert bot._record_llm_experience(action="a", outcome=1.0) is None

    def test_out_of_range_outcome_is_rejected_without_writing(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        mem = tmp_path / "agent_memory.json"
        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem))
        bot = Agent(workspace=str(tmp_path))

        assert bot._record_llm_experience(action="a", outcome=1.5) is None
        assert not _db_path_for(mem).exists()

    def test_analytics_helpers_return_empty_for_a_missing_db(
        self, tmp_path: Path
    ) -> None:
        missing = str(tmp_path / "nope.db")
        assert experiences_by_llm(missing) == []
        assert success_rate_by_model(missing) == []
        assert decision_latency_histogram(missing) == []
        summary = attribution_summary(missing)
        assert summary["experiences"] == 0
        assert summary["llm_decisions"] == 0

    def test_latency_histogram_rejects_a_nonpositive_bucket(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError):
            decision_latency_histogram(str(tmp_path / "m.db"), bucket_ms=0)


# --------------------------------------------------------------------------
# Analytics — "which model's memories are better?"
# --------------------------------------------------------------------------

def _seed(db: Path, rows: list[tuple]) -> None:
    """rows: (action, outcome, success, decision_llm)."""
    conn = _connect(db)
    try:
        ensure_attribution_schema(conn)
        for action, outcome, success, llm in rows:
            conn.execute(
                "INSERT INTO experiences (timestamp, action, outcome, context, "
                "success, decision_llm) VALUES (?, ?, ?, '{}', ?, ?)",
                ("2026-01-01T00:00:00", action, outcome, success, llm),
            )
        conn.commit()
    finally:
        conn.close()


class TestAttributionAnalytics:
    def test_experiences_by_llm_counts_only_attributed_rows(
        self, tmp_path: Path
    ) -> None:
        db = tmp_path / "m.db"
        _seed(db, [
            ("a", 1.0, 1, "model-a"),
            ("a", 0.0, 0, "model-a"),
            ("a", 1.0, 1, "model-b"),
            ("a", 1.0, 1, None),  # unattributed: credited to nobody
        ])

        by_llm = experiences_by_llm(db)
        assert by_llm == [
            {"decision_llm": "model-a", "experiences": 2, "avg_outcome": 0.5},
            {"decision_llm": "model-b", "experiences": 1, "avg_outcome": 1.0},
        ]

    def test_success_rate_by_model(self, tmp_path: Path) -> None:
        db = tmp_path / "m.db"
        _seed(db, [
            ("a", 1.0, 1, "good"),
            ("a", 1.0, 1, "good"),
            ("a", 0.0, 0, "bad"),
        ])

        rates = {r["decision_llm"]: r for r in success_rate_by_model(db)}
        assert rates["good"]["success_rate"] == pytest.approx(1.0)
        assert rates["good"]["successes"] == 2
        assert rates["bad"]["success_rate"] == pytest.approx(0.0)

    def test_latency_histogram_buckets_to_the_lower_edge(self, tmp_path: Path) -> None:
        db = tmp_path / "m.db"
        conn = _connect(db)
        try:
            ensure_attribution_schema(conn)
            for latency in (0.0, 50.0, 199.9, 200.0, 1000.0):
                conn.execute(
                    "INSERT INTO llm_decisions (latency_ms) VALUES (?)", (latency,))
            conn.commit()
        finally:
            conn.close()

        assert decision_latency_histogram(db, bucket_ms=100.0) == [
            {"bucket_ms": 0, "count": 2},      # 0.0 and 50.0
            {"bucket_ms": 100, "count": 1},    # 199.9
            {"bucket_ms": 200, "count": 1},    # 200.0
            {"bucket_ms": 1000, "count": 1},   # 1000.0
        ]

    def test_latency_histogram_ignores_null_latencies(self, tmp_path: Path) -> None:
        db = tmp_path / "m.db"
        conn = _connect(db)
        try:
            ensure_attribution_schema(conn)
            conn.execute("INSERT INTO llm_decisions (latency_ms) VALUES (NULL)")
            conn.execute("INSERT INTO llm_decisions (latency_ms) VALUES (10.0)")
            conn.commit()
        finally:
            conn.close()

        assert decision_latency_histogram(db) == [{"bucket_ms": 0, "count": 1}]

    def test_summary_separates_attributed_from_unattributed(
        self, tmp_path: Path
    ) -> None:
        db = tmp_path / "m.db"
        _seed(db, [
            ("a", 1.0, 1, "model-a"),
            ("a", 0.7, 1, None),
        ])

        summary = attribution_summary(db)
        assert summary["experiences"] == 2
        assert summary["attributed"] == 1
        assert summary["unattributed"] == 1
        assert summary["llm_decisions"] == 0
        assert summary["by_llm"][0]["decision_llm"] == "model-a"


# --------------------------------------------------------------------------
# End-to-end: the real chat_turn write is attributed and shares the run id
# --------------------------------------------------------------------------

class TestTurnRowIsAttributed:
    def test_finish_turn_row_carries_model_and_run_id(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """The real ``_finish_turn`` path stamps the turn roll-up row."""
        from agent_core.config import AgentDisplayMode

        mem = tmp_path / "agent_memory.json"
        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem))
        bot = Agent(workspace=str(tmp_path))
        bot._turn_run_id = "turn-cid-123"
        bot._last_user_input = "what changed?"
        bot._turn_started_at = None

        bot._finish_turn("All done.", None, object(), AgentDisplayMode.QUIET)

        db = _db_path_for(mem)
        rows = _attributed_rows(db)
        turn_rows = [r for r in rows if r[0] == "chat_turn"]
        assert turn_rows, f"no chat_turn experience row in {rows!r}"
        _action, llm, prompt_hash, _ts = turn_rows[-1]
        assert llm == bot.model_name
        assert prompt_hash == prompt_sha256("what changed?")

        decisions = _decisions(db)
        assert any(d[0] == "turn-cid-123" and d[1] == bot.model_name
                   for d in decisions), decisions


# --------------------------------------------------------------------------
# Cross-module: the Memory MCP server must still read an attributed table
# --------------------------------------------------------------------------

class TestMemoryServerCompatibility:
    """The real server, driven for real, over a table this code wrote.

    The attribution columns are only safe because the server reads with
    ``SELECT *`` and formats rows by column NAME.  Asserting that in a comment
    is not evidence, so these drive ``mcp_servers.memory._execute_tool`` against
    a DB produced by ``Agent._record_llm_experience``.
    """

    @pytest.fixture()
    def wired_db(self, tmp_path: Path, monkeypatch) -> Path:
        from mcp_servers import memory

        mem = tmp_path / "agent_memory.json"
        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem))
        db = _db_path_for(mem)
        bot = Agent(workspace=str(tmp_path))
        bot._record_llm_experience(
            action="llm_decision", outcome=1.0, context={"note": "attributed row"},
            decision_prompt="the question")
        bot._record_llm_experience(action="chat_turn", outcome=0.0, success=False)
        monkeypatch.setattr(memory, "_db_path", lambda: str(db))
        return db

    def test_list_experiences_still_renders_every_row(self, wired_db: Path) -> None:
        from mcp_servers import memory

        out = memory._execute_tool("list_experiences", {})

        assert "No matching experiences found." not in out
        assert "llm_decision" in out
        assert "chat_turn" in out
        # The appended columns are visible but did not displace the MCP ones.
        for mcp_col in ("timestamp", "action", "outcome", "context", "success"):
            assert mcp_col in out

    def test_get_experience_round_trips_by_rowid(self, wired_db: Path) -> None:
        from mcp_servers import memory

        out = memory._execute_tool("get_experience", {"id": 1})

        assert "Experience:" in out
        assert "llm_decision" in out

    def test_server_search_finds_a_harness_written_attributed_row(
        self, wired_db: Path,
    ) -> None:
        """The FTS index self-heals over the widened table (no triggers)."""
        from mcp_servers import memory

        out = memory._execute_tool("search_experiences", {"query": "attributed"})

        assert "No experiences matching" not in out
        assert "llm_decision" in out

    def test_server_can_append_its_own_row_after_attribution_columns_exist(
        self, wired_db: Path,
    ) -> None:
        """The server's column-named INSERT keeps working on the widened table."""
        from mcp_servers import memory

        out = memory._execute_tool(
            "record_experience",
            {"action": "mcp_written", "outcome": 0.9, "context": "{}"},
        )

        assert "Recorded experience" in out
        rows = _attributed_rows(wired_db)
        # The server's own row carries no attribution — it is not a model
        # decision, so it must not be credited to any LLM.
        assert rows[-1] == ("mcp_written", None, None, None)
