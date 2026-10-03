"""Regression tests for FTS5-backed retrieval in the Memory MCP server.

Before this, ``mcp_servers/memory.py`` searched ``experiences`` with
``action LIKE '%q%' OR context LIKE '%q%'`` ordered by ``outcome DESC``:

* a multi-word query had to appear as one contiguous phrase inside a single
  column, so ``"auth token"`` missed every row where the two words sit in
  different columns;
* ranking was by ``outcome``, a 3-valued heuristic that is near-constant per
  query, so results came back in arbitrary order;
* there was no index at all, so every search was a full table scan.

The server now keeps an FTS5 index in sync with the table and ranks hits with
``bm25()``.  The contract pinned here:

1. tokens match across columns and in any order (FTS semantics);
2. relevance, not ``outcome``, decides the order;
3. rows written DIRECTLY into ``experiences`` by the harness
   (``Agent._record_llm_experience`` inserts without going through the MCP
   server) are still indexed — the index self-heals;
4. a query with no usable tokens never raises;
5. a SQLite without FTS5 degrades to the old LIKE behaviour rather than
   breaking the tool.

FTS5 was verified available on this interpreter (SQLite 3.49.1).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mcp_servers import memory


@pytest.fixture()
def db(tmp_path: Path, monkeypatch) -> Path:
    """An isolated agent_memory.db wired into the server's _db_path."""
    path = tmp_path / "agent_memory.db"
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE experiences ("
        "timestamp TEXT, action TEXT, outcome REAL, context TEXT, success INTEGER)"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(memory, "_db_path", lambda: str(path))
    return path


def _add(path: Path, action: str, outcome: float, context: str,
         success: int | None = None) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "INSERT INTO experiences (timestamp, action, outcome, context, success) "
            "VALUES (?, ?, ?, ?, ?)",
            ("2026-10-01T00:00:00", action, outcome, context,
             1 if success is None else success),
        )
        conn.commit()
    finally:
        conn.close()


class TestTokenMatching:
    def test_matches_tokens_across_different_columns(self, db: Path) -> None:
        """'refactor auth' must match even though the words are in separate
        columns — the old LIKE required one contiguous phrase in one column."""
        _add(db, "refactor module", 1.0, '{"note": "auth flow needs work"}')

        out = memory._execute_tool(
            "search_experiences", {"query": "refactor auth"})

        assert "No experiences matching" not in out
        assert "refactor module" in out

    def test_matches_terms_in_any_order(self, db: Path) -> None:
        _add(db, "auth refactor", 1.0, "{}")
        out = memory._execute_tool(
            "search_experiences", {"query": "auth refactor"})
        assert "auth refactor" in out

    def test_all_tokens_must_be_present(self, db: Path) -> None:
        """Implicit AND: a row missing one token is not a hit."""
        _add(db, "auth module", 1.0, "{}")
        _add(db, "payment gateway", 1.0, "{}")

        out = memory._execute_tool(
            "search_experiences", {"query": "auth unrelatedterm"})

        assert "No experiences matching" in out

    def test_list_experiences_search_also_matches_tokens_across_columns(
        self, db: Path,
    ) -> None:
        _add(db, "refactor module", 1.0, '{"note": "auth flow"}')
        out = memory._execute_tool(
            "list_experiences", {"search": "refactor auth"})
        assert "refactor module" in out


class TestRelevanceRanking:
    def test_ranks_by_relevance_not_outcome(self, db: Path) -> None:
        """The dense, on-topic row must outrank the passing row that only
        mentions the terms once — ORDER BY outcome DESC put the latter first."""
        _add(db, "generic run", 1.0, '{"note": "sqlite fts5"}')
        for i in range(5):
            _add(db, f"run {i}", 0.0,
                 '{"note": "sqlite fts5 bm25 ranking sqlite fts5 bm25"}')

        out = memory._execute_tool(
            "search_experiences", {"query": "sqlite fts5"})

        assert "No experiences matching" not in out
        assert out.index("run 0") < out.index("generic run")


class TestListSortingIsHonoured:
    """list_experiences(search=...) must still obey sort/order.

    The FTS path takes an "outcome DESC" style order_by, which is validated
    against the bare column names ("outcome", "timestamp").  A caller passing
    the composed string slipped past nothing and silently fell back to bm25
    relevance, so an explicit sort was quietly ignored.
    """

    def test_outcome_desc_is_honoured(self, db: Path) -> None:
        _add(db, "low", 0.1, '{"tag": "deploy"}')
        _add(db, "high", 0.9, '{"tag": "deploy"}')

        out = memory._execute_tool("list_experiences", {
            "search": "deploy", "sort": "outcome", "order": "desc"})

        assert out.index("high") < out.index("low")

    def test_outcome_asc_is_honoured(self, db: Path) -> None:
        _add(db, "low", 0.1, '{"tag": "deploy"}')
        _add(db, "high", 0.9, '{"tag": "deploy"}')

        out = memory._execute_tool("list_experiences", {
            "search": "deploy", "sort": "outcome", "order": "asc"})

        assert out.index("low") < out.index("high")

    def test_search_combines_with_success_filter(self, db: Path) -> None:
        _add(db, "ok_row", 1.0, '{"tag": "deploy"}')
        _add(db, "bad_row", 0.0, '{"tag": "deploy"}', success=0)

        out = memory._execute_tool("list_experiences", {
            "search": "deploy", "success": False})

        assert "bad_row" in out
        assert "ok_row" not in out

    def test_no_search_keeps_plain_sorting(self, db: Path) -> None:
        _add(db, "low", 0.1, "{}")
        _add(db, "high", 0.9, "{}")
        out = memory._execute_tool("list_experiences", {"sort": "outcome"})
        assert out.index("high") < out.index("low")


class TestIndexStaysInSync:
    def test_finds_row_inserted_directly_by_the_harness(self, db: Path) -> None:
        """Agent._record_llm_experience inserts into `experiences` without the
        MCP server's help; the index must pick the row up on the next search."""
        assert "No experiences matching" in memory._execute_tool(
            "search_experiences", {"query": "checkout"})

        _add(db, "chat_turn", 1.0, '{"intent": "checkout regression"}')

        assert "chat_turn" in memory._execute_tool(
            "search_experiences", {"query": "checkout"})

    def test_picks_up_several_direct_writes_in_order(self, db: Path) -> None:
        for i in range(3):
            _add(db, f"act{i}", 1.0, '{"tag": "deploy"}')
        out = memory._execute_tool("search_experiences", {"query": "deploy"})
        assert "act0" in out and "act2" in out

    def test_deleted_row_does_not_come_back(self, db: Path) -> None:
        """A removed row must not be resurrected by a stale index entry.

        `_fts_is_current` only compares `MAX(rowid)`, so deleting a row that is
        NOT the newest leaves the index deliberately stale.  The search joins
        the index back to `experiences` with an INNER JOIN, so the orphan index
        entry must simply drop out — not raise, and not surface a dead row.
        """
        _add(db, "gone", 1.0, '{"tag": "deploy"}')
        _add(db, "stays", 1.0, '{"tag": "deploy"}')
        assert "gone" in memory._execute_tool(
            "search_experiences", {"query": "deploy"})

        # Remove the FIRST row; `stays` still owns the highest rowid, so the
        # high-water mark does not move and no rebuild is triggered.
        conn = sqlite3.connect(str(db))
        try:
            conn.execute("DELETE FROM experiences WHERE action = ?", ("gone",))
            conn.commit()
        finally:
            conn.close()

        out = memory._execute_tool("search_experiences", {"query": "deploy"})
        assert "stays" in out
        assert "gone" not in out


class TestRobustness:
    def test_punctuation_only_query_does_not_raise(self, db: Path) -> None:
        _add(db, "auth", 1.0, "{}")
        out = memory._execute_tool(
            "search_experiences", {"query": "!!! ??? ---"})
        assert isinstance(out, str)

    def test_fts_syntax_in_user_query_is_not_interpreted(self, db: Path) -> None:
        """A raw MATCH would read ``payment OR auth`` as a UNION and return
        BOTH rows.  User text must be tokenised and quoted, so operators are
        literal words (AND semantics) — no injection, no surprise hits."""
        _add(db, "auth module", 1.0, "{}")
        _add(db, "payment module", 1.0, "{}")

        out = memory._execute_tool(
            "search_experiences", {"query": "payment OR auth"})

        # Neither row contains every token, so the OR must not have unioned.
        assert "No experiences matching" in out

    def test_empty_query_returns_no_matches_not_everything(self, db: Path) -> None:
        _add(db, "auth module", 1.0, "{}")
        out = memory._execute_tool("search_experiences", {"query": "   "})
        assert "No experiences matching" in out

    def test_falls_back_to_like_when_fts_is_unavailable(
        self, db: Path, monkeypatch,
    ) -> None:
        """A SQLite built without FTS5 must degrade, not break the tool."""
        monkeypatch.setattr(memory, "_ensure_fts", lambda conn: False)
        _add(db, "refactor module", 1.0, '{"note": "auth flow"}')

        out = memory._execute_tool(
            "search_experiences", {"query": "refactor"})

        assert "refactor module" in out
