"""Agent Memory MCP server over stdio.

Exposes tools for querying and managing the agent's persistent memory:
  list_experiences, get_experience, search_experiences,
  record_experience, evolution_summary, list_templates,
  prompt_metrics, memory_stats.

Reads from agent_memory.db (SQLite) at the workspace root.
Zero external dependencies — stdlib only.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# JSON-RPC helpers (inlined to keep this zero-dependency)
# ---------------------------------------------------------------------------

REQUEST_ID_COUNTER = 0


def _next_id() -> int:
    global REQUEST_ID_COUNTER
    REQUEST_ID_COUNTER += 1
    return REQUEST_ID_COUNTER


def _parse_message(raw: str) -> dict[str, Any]:
    msg = json.loads(raw)
    if not isinstance(msg, dict):
        raise ValueError("not a JSON object")
    return msg


def _make_response(req_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _make_error(req_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


def _write(msg: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(msg, separators=(",", ":")) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# MCP protocol constants
# ---------------------------------------------------------------------------

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INTERNAL_ERROR = -32603

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "agent-memory", "version": "1.0"}
SERVER_CAPABILITIES: dict[str, Any] = {"tools": {}}

# ---------------------------------------------------------------------------
# Database location
# ---------------------------------------------------------------------------

def _db_path() -> str:
    """Find agent_memory.db at the workspace root."""
    # Try CWD first, then walk up
    cwd = Path.cwd()
    for d in [cwd, *cwd.parents]:
        p = d / "agent_memory.db"
        if p.exists():
            return str(p)
    # Fallback: relative to this file (mcp_servers/..)
    p = Path(__file__).resolve().parent.parent / "agent_memory.db"
    if p.exists():
        return str(p)
    raise FileNotFoundError("agent_memory.db not found")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Full-text search index (FTS5)
# ---------------------------------------------------------------------------
#
# Retrieval used to be ``action LIKE '%q%' OR context LIKE '%q%'`` ordered by
# ``outcome DESC``.  That could not match a multi-word query whose words sit in
# different columns ("refactor auth" missed every row), ranked by a near-
# constant heuristic instead of relevance, and scanned the whole table on
# every call.  SQLite ships FTS5 (verified on this interpreter), so the
# server keeps a real index and ranks with bm25().
#
# The index is a CACHE, never a source of truth: `experiences` remains the
# table of record and `Agent._record_llm_experience` writes straight into it
# without going through this server.  `_ensure_fts` therefore self-heals on
# every search (rebuilds when the index is missing or behind) rather than
# depending on triggers, which would silently break for those direct writes.

_FTS_TABLE = "experiences_fts"
_FTS_STALE_KEY = "_fts_built_upto"  # table rowid high-water mark in a side table


def _tokenize(query: str) -> list[str]:
    """Split a user query into plain FTS5 tokens.

    Everything that is not alphanumeric (underscore included, so snake_case
    and dotted paths survive) is a separator.  Returning bare tokens — rather
    than passing user text to MATCH — means FTS operators typed by the user
    ("OR", "NEAR", "*", unbalanced quotes) are literal search words instead
    of syntax, so a query can never widen itself into a UNION or raise.
    """
    return [t for t in re.split(r"[^0-9A-Za-z_]+", query or "") if t]


def _fts_match_expr(tokens: list[str]) -> str:
    """Build an AND-ed, quoted MATCH expression from *tokens*."""
    return " AND ".join('"' + t.replace('"', '""') + '"' for t in tokens)


def _ensure_fts(conn: sqlite3.Connection) -> bool:
    """Make sure the FTS index exists and covers every row of `experiences`.

    Returns True when the index is usable, False when this SQLite lacks FTS5
    (or the experiences table is missing) so callers can fall back to LIKE.
    Idempotent and cheap when already current: it compares the indexed high-
    water mark against ``MAX(rowid)`` and rebuilds only when they differ.
    """
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (_FTS_TABLE,),
        ).fetchone()
        if exists is None:
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS experiences_fts "
                "USING fts5(action, context, content='experiences', "
                "content_rowid='rowid')"
            )
            _rebuild_fts(conn)
            conn.commit()
        elif not _fts_is_current(conn):
            _rebuild_fts(conn)
            conn.commit()
        return True
    except sqlite3.Error:
        # No FTS5 in this build, or a table we cannot index: the caller uses
        # the LIKE fallback.  Never raise out of a search tool.
        return False


def _fts_is_current(conn: sqlite3.Connection) -> bool:
    """True when the index already holds every row currently in `experiences`."""
    try:
        row = conn.execute("SELECT MAX(rowid) FROM experiences").fetchone()
    except sqlite3.Error:
        return False
    max_rowid = row[0] if row else None
    mark = conn.execute(
        "SELECT value FROM experiences_meta WHERE key=?", (_FTS_STALE_KEY,)
    ).fetchone()
    if mark is None:
        return False
    # A rebuild that found an empty table records -1; compare as ints so a
    # NULL max (empty table) and the marker still agree.
    try:
        return int(mark[0]) == int(max_rowid or -1)
    except (TypeError, ValueError):
        return False


def _rebuild_fts(conn: sqlite3.Connection) -> None:
    """Rebuild the whole index from `experiences` and record the mark."""
    conn.execute("CREATE TABLE IF NOT EXISTS experiences_meta "
                 "(key TEXT PRIMARY KEY, value TEXT)")
    max_rowid = conn.execute("SELECT MAX(rowid) FROM experiences").fetchone()[0]
    # An EXTERNAL-CONTENT index stores no copy of the text: it must be
    # maintained through its own commands.  A plain DELETE/INSERT against the
    # virtual table raises "database disk image is malformed"; 'rebuild'
    # reindexes the whole content table in a single pass (cheaper than
    # delete-then-copy, and atomic).
    conn.execute(
        "INSERT INTO experiences_fts (experiences_fts) VALUES('rebuild')"
    )
    conn.execute(
        "INSERT INTO experiences_meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (_FTS_STALE_KEY, str(int(max_rowid or -1))),
    )


def _order_clause(order_by: str | None) -> str:
    """Validate an ORDER BY request and render it as safe SQL.

    *order_by* is the caller's ``"<column> <ASC|DESC>"`` string.  It is parsed
    against a whitelist of real columns and two directions rather than
    interpolated whole, so an unexpected value degrades to bm25 relevance
    order instead of silently dropping the caller's sort.
    """
    if not order_by:
        return "ORDER BY bm25(experiences_fts)"
    parts = order_by.split()
    if len(parts) != 2:
        return "ORDER BY bm25(experiences_fts)"
    column, direction = parts[0].lower(), parts[1].upper()
    if column not in ("outcome", "timestamp") or direction not in ("ASC", "DESC"):
        return "ORDER BY bm25(experiences_fts)"
    return f"ORDER BY e.{column} {direction}"


def _search_experiences(
    conn: sqlite3.Connection, query: str, limit: int, *,
    success: bool | None = None,
    order_by: str | None = None,
) -> list:
    """Rows matching *query*. FTS5 + bm25 when available.

    *order_by* is an explicit ``"column ASC|DESC"`` chosen by the caller; when
    it is None the rows come back in bm25 relevance order.  *success* adds the
    same success filter ``list_experiences`` applies.

    Falls back to the historical LIKE scan when the query has no usable
    tokens or FTS5 is unavailable, so the tool keeps working everywhere.
    """
    tokens = _tokenize(query)
    if not tokens:
        return []
    if _ensure_fts(conn):
        try:
            clause = ""
            params: list[Any] = []
            if success is not None:
                clause = " AND e.success = ?"
                params.append(1 if success else 0)
            return conn.execute(
                "SELECT e.* FROM experiences_fts f "
                "JOIN experiences e ON e.rowid = f.rowid "
                f"WHERE experiences_fts MATCH ?{clause} "
                f"{_order_clause(order_by)} LIMIT ?",
                (_fts_match_expr(tokens), *params, limit),
            ).fetchall()
        except sqlite3.Error:
            pass  # corrupt index or bad expression: degrade, do not raise
    where = ""
    params = []
    if success is not None:
        where = " AND success = ?"
        params.append(1 if success else 0)
    like = f"%{query.strip()}%"
    params.extend([like, like, limit])
    return conn.execute(
        "SELECT * FROM experiences WHERE (action LIKE ? OR context LIKE ?)"
        f"{where} ORDER BY outcome DESC LIMIT ?",
        params,
    ).fetchall()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_table(rows: list[sqlite3.Row], max_rows: int = 50) -> str:
    """Format rows as a text table with aligned columns."""
    if not rows:
        return "(0 rows)"
    headers = rows[0].keys()
    display_rows = rows[:max_rows]
    col_widths = {h: len(h) for h in headers}
    str_rows = []
    for row in display_rows:
        sr = {}
        for h in headers:
            val = str(row[h]) if row[h] is not None else "NULL"
            if len(val) > 80:
                val = val[:77] + "..."
            sr[h] = val
            col_widths[h] = max(col_widths[h], len(val))
        str_rows.append(sr)
    lines = []
    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    lines.append(header_line)
    lines.append("-+-".join("-" * col_widths[h] for h in headers))
    for sr in str_rows:
        lines.append(" | ".join(sr[h].ljust(col_widths[h]) for h in headers))
    total = len(rows)
    if total > max_rows:
        lines.append(f"\n... ({total - max_rows} more rows, showing {max_rows} of {total})")
    else:
        lines.append(f"\n({total} row{'s' if total != 1 else ''})")
    return "\n".join(lines)


def _fmt_size(n: int) -> str:
    if n > 1024 * 1024:
        return f"{n / (1024*1024):.1f} MB"
    if n > 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_experiences",
        "description": (
            "List learned experiences (action → outcome records) from the agent's "
            "memory. Supports filtering by success/failure, text search, and "
            "sorting by outcome or timestamp."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "success": {
                    "type": "boolean",
                    "description": "Filter: true=only successful, false=only failed, omit=all",
                },
                "search": {
                    "type": "string",
                    "description": "Text to search in action/context fields",
                },
                "sort": {
                    "type": "string",
                    "enum": ["outcome", "timestamp"],
                    "description": "Sort field (default: timestamp)",
                },
                "order": {
                    "type": "string",
                    "enum": ["asc", "desc"],
                    "description": "Sort direction (default: desc)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max rows to return (default: 20)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_experience",
        "description": "Get a single experience by its row id (from list_experiences).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "Row id of the experience",
                },
            },
            "required": ["id"],
        },
    },
    {
        "name": "search_experiences",
        "description": (
            "Full-text search across experiences. Searches action, context, and "
            "outcome fields. Returns matching experiences with their scores."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (matches against action + context)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default: 10)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "record_experience",
        "description": (
            "Record a new experience in the agent's memory. "
            "Provides action, outcome score (0-1), and optional context JSON."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "What action was taken",
                },
                "outcome": {
                    "type": "number",
                    "description": "Outcome score (0.0 to 1.0)",
                },
                "context": {
                    "type": "string",
                    "description": "Optional JSON context string",
                },
            },
            "required": ["action", "outcome"],
        },
    },
    {
        "name": "evolution_summary",
        "description": (
            "Show the agent's evolution history: fitness scores across generations, "
            "changes made, and performance improvements. Includes a trend summary."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max generations to show (default: 20)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "list_templates",
        "description": (
            "List all prompt templates stored in memory, with their task type, "
            "profile, version, and creation/update timestamps."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_type": {
                    "type": "string",
                    "description": "Filter by task type (optional)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "prompt_metrics",
        "description": (
            "Show per-template performance metrics: success/failure counts, "
            "average latency, and update timestamps."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_type": {
                    "type": "string",
                    "description": "Filter by task type (optional)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "memory_stats",
        "description": (
            "Overview of the agent memory database: table row counts, file size, "
            "and a quick health summary."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def _execute_tool(name: str, args: dict[str, Any]) -> str:
    """Run a tool and return the result as a string."""

    if name == "list_experiences":
        where_clauses = []
        params: list[Any] = []
        if "success" in args:
            if args["success"]:
                where_clauses.append("success = 1")
            else:
                where_clauses.append("success = 0")
        if args.get("search"):
            # Text search goes through the same tokenised index as
            # search_experiences (was: a whole-string LIKE, which could not
            # match a query whose words are spread across columns).
            search = args["search"]
        else:
            search = None
        sort = args.get("sort", "timestamp")
        if sort not in ("outcome", "timestamp"):
            sort = "timestamp"
        order = args.get("order", "desc").upper()
        if order not in ("ASC", "DESC"):
            order = "DESC"
        limit = args.get("limit", 20)

        with _connect() as conn:
            if search:
                rows = _search_experiences(
                    conn, search, limit,
                    success=(bool(args["success"]) if "success" in args else None),
                    order_by=f"{sort} {order}",
                )
            else:
                where = f"WHERE {' AND '.join(where_clauses)}" \
                    if where_clauses else ""
                rows = conn.execute(
                    f"SELECT * FROM experiences {where} "
                    f"ORDER BY {sort} {order} LIMIT ?",
                    (*params, limit),
                ).fetchall()

        if not rows:
            return "No matching experiences found."
        return _format_table(rows, max_rows=limit)

    if name == "get_experience":
        exp_id = args["id"]
        with _connect() as conn:
            row = conn.execute(
                "SELECT * FROM experiences WHERE rowid = ?", (exp_id,)
            ).fetchone()
        if row is None:
            # Try by timestamp (first column)
            with _connect() as conn:
                row = conn.execute(
                    "SELECT * FROM experiences WHERE timestamp = ?", (exp_id,)
                ).fetchone()
        if row is None:
            raise ValueError(f"Experience with id={exp_id} not found")
        d = dict(row)
        lines = ["Experience:"]
        for k, v in d.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    if name == "search_experiences":
        query = args["query"]
        limit = args.get("limit", 10)
        with _connect() as conn:
            rows = _search_experiences(conn, query, limit)
        if not rows:
            return f"No experiences matching '{query}'."
        return _format_table(rows, max_rows=limit)

    if name == "record_experience":
        action = args["action"]
        outcome = args["outcome"]
        if not 0.0 <= outcome <= 1.0:
            raise ValueError("outcome must be between 0.0 and 1.0")
        context = args.get("context", "{}")
        import time
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        with _connect() as conn:
            cur = conn.execute(
                "INSERT INTO experiences (timestamp, action, outcome, context, success) "
                "VALUES (?, ?, ?, ?, ?)",
                (ts, action, outcome, context, 1 if outcome >= 0.5 else 0),
            )
            conn.commit()
            new_id = cur.lastrowid
        return f"Recorded experience (rowid={new_id}): action='{action}', outcome={outcome}"

    if name == "evolution_summary":
        limit = args.get("limit", 20)
        with _connect() as conn:
            rows = conn.execute(
                "SELECT * FROM evolution_records ORDER BY generation DESC LIMIT ?",
                (limit,),
            ).fetchall()
            total = conn.execute("SELECT COUNT(*) as c FROM evolution_records").fetchone()["c"]
        if not rows:
            return "No evolution records found."
        table = _format_table(rows, max_rows=limit)
        # Trend summary
        scores = [r["fitness_score"] for r in rows if r["fitness_score"] is not None]
        if len(scores) >= 2:
            # rows are DESC by generation, so first=latest, last=oldest
            trend = "improving" if scores[0] > scores[-1] else "declining" if scores[0] < scores[-1] else "stable"
            summary = (
                f"\nTrend: {trend}\n"
                f"  Latest fitness: {scores[0]:.3f}\n"
                f"  Oldest fitness: {scores[-1]:.3f}\n"
                f"  Delta: {scores[0] - scores[-1]:+.3f}\n"
                f"  Total generations: {total}"
            )
        else:
            summary = f"\nTotal generations: {total}"
        return table + summary

    if name == "list_templates":
        where = ""
        params: list[Any] = []
        if args.get("task_type"):
            where = "WHERE task_type = ?"
            params.append(args["task_type"])
        with _connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM prompt_templates {where} ORDER BY task_type, version",
                params,
            ).fetchall()
        if not rows:
            return "No prompt templates stored."
        return _format_table(rows, max_rows=50)

    if name == "prompt_metrics":
        where = ""
        params: list[Any] = []
        if args.get("task_type"):
            where = "WHERE task_type = ?"
            params.append(args["task_type"])
        with _connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM prompt_metrics {where} ORDER BY task_type, version",
                params,
            ).fetchall()
        if not rows:
            return "No prompt metrics recorded."
        return _format_table(rows, max_rows=50)

    if name == "memory_stats":
        db = _db_path()
        size = os.path.getsize(db)
        with _connect() as conn:
            tables = [
                r["name"]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            ]
            counts = {}
            for t in tables:
                counts[t] = conn.execute(f"SELECT COUNT(*) as c FROM [{t}]").fetchone()["c"]
        lines = [
            f"Memory database: {db}",
            f"File size: {_fmt_size(size)}",
            f"Tables: {len(tables)}",
            "",
        ]
        for t in tables:
            lines.append(f"  {t}: {counts[t]} row{'s' if counts[t] != 1 else ''}")
        # Quick health
        exp_total = counts.get("experiences", 0)
        exp_success = 0
        with _connect() as conn:
            exp_success = conn.execute("SELECT COUNT(*) FROM experiences WHERE success=1").fetchone()[0]
        if exp_total > 0:
            lines.append("")
            lines.append(f"Experience success rate: {exp_success}/{exp_total} ({100*exp_success/exp_total:.0f}%)")
        return "\n".join(lines)

    raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# Message routing
# ---------------------------------------------------------------------------

def _handle(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Route an incoming JSON-RPC message; return a response dict (or None for notifications)."""
    req_id = msg.get("id")
    method = msg.get("method")
    params = msg.get("params", {})

    if method == "notifications/initialized":
        return None

    if method == "initialize":
        return _make_response(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": SERVER_CAPABILITIES,
            "serverInfo": SERVER_INFO,
        })

    if method == "tools/list":
        return _make_response(req_id, {"tools": TOOLS})

    if method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        if not isinstance(tool_name, str):
            return _make_error(req_id, -32602, "missing tool name")
        try:
            text = _execute_tool(tool_name, arguments)
            return _make_response(req_id, {
                "content": [{"type": "text", "text": text}],
                "isError": False,
            })
        except (KeyError, ValueError, TypeError, FileNotFoundError, sqlite3.Error) as exc:
            return _make_response(req_id, {
                "content": [{"type": "text", "text": f"Error: {exc}"}],
                "isError": True,
            })

    if req_id is not None:
        return _make_error(req_id, METHOD_NOT_FOUND, f"Unknown method: {method}")

    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Read newline-delimited JSON-RPC from stdin, write responses to stdout."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = _parse_message(line)
        except (json.JSONDecodeError, ValueError) as exc:
            _write(_make_error(None, PARSE_ERROR, f"Parse error: {exc}"))
            continue
        response = _handle(msg)
        if response is not None:
            _write(response)


if __name__ == "__main__":
    main()
