"""LLM decision attribution for the ``experiences`` memory store.

Why this exists
---------------
``experiences`` records *what* happened (action / outcome / success), but not
*which model decided it*.  Without that, swapping in a better model and
measuring whether memory quality improved is impossible: the stored outcomes
have no ground-truth author.

This module adds exactly that, and nothing behavioural.  It is harness-layer
observability only (decision #014) — no model, prompt or tool-loop change.

Schema (additive, never destructive)
------------------------------------
``experiences`` gains three NULLABLE columns, appended so every existing
positional expectation still holds:

    decision_llm, decision_prompt_hash, decision_timestamp

The Memory MCP server (``mcp_servers/memory.py``) reads this table with
``SELECT *`` and formats rows **by column name** (``row[h]``), and both
writers name their columns explicitly in ``INSERT INTO experiences (...)``.
So appended columns are invisible to the server and to the FTS index
(``experiences_fts`` is declared over ``action, context`` only) — an old row
simply reads back ``NULL`` for the new columns.

``llm_decisions`` logs one row per *attributed* memory write:

    id, run_id, model_name, prompt_sha256, outcome_experience_id,
    latency_ms, token_usage, timestamp

The invariant is exact: ``llm_decisions`` holds one row per ``experiences``
row whose ``decision_llm`` is non-NULL.  Rows recorded without attribution
(untraced / non-model paths) leave both untouched, so such runs stay
byte-identical — the same contract ``Agent._record_llm_experience`` honours.

Nothing here ever raises: a missing, locked or read-only DB must degrade to a
silent no-op, never break a turn.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import datetime
from typing import Any, Sequence

logger = logging.getLogger(__name__)

#: Columns of the ``experiences`` table as the Memory MCP server expects them.
#: These five keep their exact order — the attribution columns below are
#: APPENDED after them.
EXPERIENCE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("timestamp", "TEXT"),
    ("action", "TEXT"),
    ("outcome", "REAL"),
    ("context", "TEXT"),
    ("success", "INTEGER"),
)

#: Attribution columns appended to ``experiences`` (all nullable).
ATTRIBUTION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("decision_llm", "TEXT"),
    ("decision_prompt_hash", "TEXT"),
    ("decision_timestamp", "TEXT"),
)

#: Full DDL used when this module has to create the table from scratch.
CREATE_EXPERIENCES_SQL = (
    "CREATE TABLE IF NOT EXISTS experiences ("
    + ", ".join(f"{name} {typ}" for name, typ in EXPERIENCE_COLUMNS + ATTRIBUTION_COLUMNS)
    + ")"
)

#: The provenance log of model invocations that produced a memory write.
CREATE_LLM_DECISIONS_SQL = (
    "CREATE TABLE IF NOT EXISTS llm_decisions ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
    "run_id TEXT, "
    "model_name TEXT, "
    "prompt_sha256 TEXT, "
    "outcome_experience_id INTEGER, "
    "latency_ms REAL, "
    "token_usage INTEGER, "
    "timestamp TEXT)"
)


def prompt_sha256(prompt: Any) -> str:
    """Stable SHA-256 of a prompt, for cross-run decision comparison.

    Accepts a plain string or a chat ``messages`` list.  For a list the
    serialisation is canonical (``sort_keys`` + compact separators) so the same
    conversation hashes identically across processes and Python versions; the
    ``messages`` markers are stripped so a transient loop tag cannot change the
    identity of the prompt that was answered.
    """
    if isinstance(prompt, str):
        canonical = prompt
    else:
        try:
            clean = [
                {k: v for k, v in m.items() if not str(k).startswith("_")}
                if isinstance(m, dict) else m
                for m in prompt
            ]
            canonical = json.dumps(clean, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":"), default=str)
        except (TypeError, ValueError):
            canonical = str(prompt)
    return hashlib.sha256(canonical.encode("utf-8", errors="replace")).hexdigest()


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    # PRAGMA rows are (cid, name, type, notnull, dflt_value, pk).
    return {str(r[1]) for r in rows}


def ensure_attribution_schema(conn: sqlite3.Connection) -> None:
    """Create the tables and additively migrate a pre-existing ``experiences``.

    Idempotent.  Existing rows keep their values and read back ``NULL`` for the
    appended attribution columns — no row is ever rewritten or dropped, so an
    MCP-authored table is extended in place rather than replaced.
    """
    conn.execute(CREATE_EXPERIENCES_SQL)
    conn.execute(CREATE_LLM_DECISIONS_SQL)
    existing = _table_columns(conn, "experiences")
    for name, typ in ATTRIBUTION_COLUMNS:
        if name not in existing:
            conn.execute(f"ALTER TABLE experiences ADD COLUMN {name} {typ}")


def record_llm_decision(
    conn: sqlite3.Connection,
    *,
    run_id: str | None,
    model_name: str | None,
    prompt_sha256: str | None = None,
    outcome_experience_id: int | None = None,
    latency_ms: float | None = None,
    token_usage: int | None = None,
) -> int | None:
    """Insert one provenance row and return its rowid (``None`` on failure)."""
    try:
        cur = conn.execute(
            "INSERT INTO llm_decisions (run_id, model_name, prompt_sha256, "
            "outcome_experience_id, latency_ms, token_usage, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                model_name,
                prompt_sha256,
                outcome_experience_id,
                None if latency_ms is None else float(latency_ms),
                None if token_usage is None else int(token_usage),
                datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            ),
        )
        return int(cur.lastrowid) if cur.lastrowid is not None else None
    except sqlite3.Error as exc:
        logger.debug("Could not record llm_decision (no-op): %s", exc)
        return None


def resolve_decision_llm(agent: Any) -> str | None:
    """Best-effort identity of the model behind *agent*'s decisions.

    Prefers the agent's own ``model_name``; falls back to the active provider's
    model when a caller swapped the provider under a stale name.  Returns
    ``None`` when nothing can be determined — callers then record without
    attribution, which is the safe (schema-compatible) outcome.
    """
    for candidate in (
        getattr(agent, "model_name", None),
        getattr(getattr(agent, "llm", None), "model_name", None),
        getattr(getattr(getattr(agent, "llm", None), "_provider", None),
                "model_name", None),
    ):
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _open(db_path: str) -> sqlite3.Connection | None:
    """Read-only-ish connection for the analytics helpers (``None`` on error)."""
    try:
        return sqlite3.connect(str(db_path), timeout=5.0)
    except sqlite3.Error as exc:
        logger.debug("Could not open memory DB for attribution query: %s", exc)
        return None


def _fetch(db_path: str, sql: str, params: Sequence[Any] = ()) -> list[tuple]:
    """Run *sql*, returning ``[]`` for a missing DB/table (never raises)."""
    conn = _open(db_path)
    if conn is None:
        return []
    try:
        return conn.execute(sql, tuple(params)).fetchall()
    except sqlite3.Error as exc:
        logger.debug("Attribution query failed (no-op): %s", exc)
        return []
    finally:
        conn.close()


def experiences_by_llm(db_path: str) -> list[dict[str, Any]]:
    """Per-model experience counts with mean outcome, best first.

    Only attributed rows are counted: an experience with no ``decision_llm``
    carries no provenance, so it must not be credited to any model.
    """
    rows = _fetch(
        db_path,
        "SELECT decision_llm, COUNT(*), AVG(outcome) FROM experiences "
        "WHERE decision_llm IS NOT NULL "
        "GROUP BY decision_llm ORDER BY COUNT(*) DESC, decision_llm ASC",
    )
    return [
        {"decision_llm": r[0], "experiences": int(r[1]),
         "avg_outcome": float(r[2]) if r[2] is not None else None}
        for r in rows
    ]


def success_rate_by_model(db_path: str) -> list[dict[str, Any]]:
    """Per-model success rate over attributed experiences."""
    rows = _fetch(
        db_path,
        "SELECT decision_llm, COUNT(*), SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) "
        "FROM experiences WHERE decision_llm IS NOT NULL "
        "GROUP BY decision_llm ORDER BY decision_llm ASC",
    )
    out: list[dict[str, Any]] = []
    for llm, total, successes in rows:
        total_i = int(total)
        ok = int(successes or 0)
        out.append({
            "decision_llm": llm,
            "total": total_i,
            "successes": ok,
            "success_rate": (ok / total_i) if total_i else None,
        })
    return out


def decision_latency_histogram(
    db_path: str, bucket_ms: float = 100.0,
) -> list[dict[str, Any]]:
    """Latency histogram of logged decisions, bucketed to *bucket_ms*.

    The bucket key is the inclusive lower edge in milliseconds, so
    ``bucket_ms=100`` yields ``0`` for 0-99.9 ms, ``100`` for 100-199.9 ms, ...
    """
    if bucket_ms <= 0:
        raise ValueError("bucket_ms must be positive")
    rows = _fetch(
        db_path,
        "SELECT CAST(latency_ms / ? AS INTEGER) AS bucket, COUNT(*) "
        "FROM llm_decisions WHERE latency_ms IS NOT NULL "
        "GROUP BY bucket ORDER BY bucket ASC",
        (float(bucket_ms),),
    )
    return [
        {"bucket_ms": int(r[0]) * bucket_ms, "count": int(r[1])} for r in rows
    ]


def attribution_summary(db_path: str) -> dict[str, Any]:
    """One-call roll-up: attributed vs unattributed experience rows."""
    rows = _fetch(
        db_path,
        "SELECT COUNT(*), SUM(CASE WHEN decision_llm IS NOT NULL THEN 1 ELSE 0 END) "
        "FROM experiences",
    )
    decisions = _fetch(db_path, "SELECT COUNT(*) FROM llm_decisions")
    total = int(rows[0][0]) if rows else 0
    attributed = int(rows[0][1] or 0) if rows else 0
    return {
        "experiences": total,
        "attributed": attributed,
        "unattributed": total - attributed,
        "llm_decisions": int(decisions[0][0]) if decisions else 0,
        "by_llm": experiences_by_llm(db_path),
        "success_by_model": success_rate_by_model(db_path),
    }
