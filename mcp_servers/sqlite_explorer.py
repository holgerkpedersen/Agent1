"""SQLite Explorer MCP server over stdio.

Exposes tools for querying and inspecting SQLite databases:
  list_databases, list_tables, describe_table, query, execute,
  count_rows, export_csv.

Zero external dependencies — stdlib only.
"""
from __future__ import annotations

import csv
import io
import os
import sqlite3
import sys
import json
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
SERVER_INFO = {"name": "sqlite-explorer", "version": "1.0"}
SERVER_CAPABILITIES: dict[str, Any] = {"tools": {}}

# Default search directories (relative to workspace root)
DEFAULT_SEARCH_DIRS = [".", "backups"]


# ---------------------------------------------------------------------------
# Database discovery
# ---------------------------------------------------------------------------

def _find_databases(search_dirs: list[str] | None = None) -> list[dict[str, str]]:
    """Walk search dirs and find .db / .sqlite / .sqlite3 files."""
    dirs = search_dirs or DEFAULT_SEARCH_DIRS
    found: dict[str, dict[str, str]] = {}
    for d in dirs:
        base = Path(d)
        if not base.is_dir():
            continue
        for p in base.rglob("*.db"):
            _add_db(found, p)
        for p in base.rglob("*.sqlite"):
            _add_db(found, p)
        for p in base.rglob("*.sqlite3"):
            _add_db(found, p)
    return sorted(found.values(), key=lambda x: x["path"])


def _add_db(found: dict[str, dict[str, str]], p: Path) -> None:
    key = str(p.resolve())
    if key not in found:
        found[key] = {"name": p.stem, "path": str(p), "size_bytes": str(p.stat().st_size)}


def _connect(db_path: str) -> sqlite3.Connection:
    """Open a SQLite database; raises if file doesn't exist."""
    p = Path(db_path)
    if not p.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_databases",
        "description": (
            "Find all SQLite database files (.db, .sqlite, .sqlite3) in the "
            "workspace. Returns name, path, and size for each."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "search_dirs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Directories to search (default: workspace root)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "list_tables",
        "description": "List all tables in a SQLite database.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "database": {
                    "type": "string",
                    "description": "Path to the SQLite database file",
                },
            },
            "required": ["database"],
        },
    },
    {
        "name": "describe_table",
        "description": (
            "Show column names, types, and constraints for a table. "
            "Also shows row count and index information."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "database": {
                    "type": "string",
                    "description": "Path to the SQLite database file",
                },
                "table": {
                    "type": "string",
                    "description": "Table name to describe",
                },
            },
            "required": ["database", "table"],
        },
    },
    {
        "name": "query",
        "description": (
            "Execute a read-only SQL query (SELECT). Results are returned as "
            "a formatted text table. Supports LIMIT/OFFSET for pagination."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "database": {
                    "type": "string",
                    "description": "Path to the SQLite database file",
                },
                "sql": {
                    "type": "string",
                    "description": "SQL SELECT query to execute",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max rows to return (default: 50)",
                },
            },
            "required": ["database", "sql"],
        },
    },
    {
        "name": "execute",
        "description": (
            "Execute a write SQL statement (INSERT, UPDATE, DELETE, CREATE, etc.). "
            "Returns the number of affected rows. Use with caution."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "database": {
                    "type": "string",
                    "description": "Path to the SQLite database file",
                },
                "sql": {
                    "type": "string",
                    "description": "SQL statement to execute",
                },
            },
            "required": ["database", "sql"],
        },
    },
    {
        "name": "count_rows",
        "description": "Count total rows in a table.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "database": {
                    "type": "string",
                    "description": "Path to the SQLite database file",
                },
                "table": {
                    "type": "string",
                    "description": "Table name",
                },
            },
            "required": ["database", "table"],
        },
    },
    {
        "name": "export_csv",
        "description": (
            "Run a SQL query and export results as CSV text. "
            "Useful for data export or piping to other tools."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "database": {
                    "type": "string",
                    "description": "Path to the SQLite database file",
                },
                "sql": {
                    "type": "string",
                    "description": "SQL SELECT query to export",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max rows to export (default: 1000)",
                },
            },
            "required": ["database", "sql"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def _format_table(rows: list[sqlite3.Row], max_rows: int = 50) -> str:
    """Format rows as a text table with aligned columns."""
    if not rows:
        return "(0 rows)"
    headers = rows[0].keys()
    # Truncate to max_rows
    display_rows = rows[:max_rows]
    # Calculate column widths
    col_widths = {h: len(h) for h in headers}
    str_rows = []
    for row in display_rows:
        sr = {}
        for h in headers:
            val = str(row[h]) if row[h] is not None else "NULL"
            # Truncate long values
            if len(val) > 80:
                val = val[:77] + "..."
            sr[h] = val
            col_widths[h] = max(col_widths[h], len(val))
        str_rows.append(sr)
    # Build output
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


def _execute_tool(name: str, args: dict[str, Any]) -> str:
    """Run a tool and return the result as a string."""
    if name == "list_databases":
        dbs = _find_databases(args.get("search_dirs"))
        if not dbs:
            return "No SQLite databases found in the workspace."
        lines = [f"Found {len(dbs)} database(s):"]
        for db in dbs:
            size = int(db["size_bytes"])
            if size > 1024 * 1024:
                size_str = f"{size / (1024*1024):.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            lines.append(f"  - {db['name']}: {db['path']} ({size_str})")
        return "\n".join(lines)

    if name == "list_tables":
        db_path = args["database"]
        with _connect(db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
            tables = [row["name"] for row in cursor.fetchall()]
        if not tables:
            return "No user tables found."
        lines = [f"Tables in {Path(db_path).name} ({len(tables)}):"]
        for t in tables:
            lines.append(f"  - {t}")
        return "\n".join(lines)

    if name == "describe_table":
        db_path = args["database"]
        table = args["table"]
        with _connect(db_path) as conn:
            # Check table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            )
            if not cursor.fetchone():
                raise ValueError(f"Table '{table}' not found")
            # Column info
            cursor = conn.execute(f"PRAGMA table_info({table})")
            cols = cursor.fetchall()
            # Row count
            cursor = conn.execute(f"SELECT COUNT(*) as cnt FROM [{table}]")
            row_count = cursor.fetchone()["cnt"]
            # Indexes
            cursor = conn.execute(f"PRAGMA index_list({table})")
            indexes = cursor.fetchall()

        lines = [f"Table: {table}  ({row_count} rows)", "", "Columns:"]
        for col in cols:
            pk = " [PK]" if col["pk"] else ""
            notnull = " NOT NULL" if col["notnull"] else ""
            default = f" DEFAULT {col['dflt_value']}" if col["dflt_value"] is not None else ""
            lines.append(f"  {col['name']}: {col['type']}{pk}{notnull}{default}")
        if indexes:
            lines.append("")
            lines.append(f"Indexes ({len(indexes)}):")
            for idx in indexes:
                cursor = conn.execute(f"PRAGMA index_info({idx['name']})")
                idx_cols = [r["name"] for r in cursor.fetchall()]
                unique = " (unique)" if idx["unique"] else ""
                lines.append(f"  {idx['name']}: ({', '.join(idx_cols)}){unique}")
        return "\n".join(lines)

    if name == "query":
        db_path = args["database"]
        sql = args["sql"].strip()
        limit = args.get("limit", 50)
        # Safety: enforce SELECT only
        upper = sql.upper().lstrip()
        if not upper.startswith("SELECT") and not upper.startswith("WITH"):
            raise ValueError("Only SELECT/WITH queries allowed in 'query'. Use 'execute' for writes.")
        # Auto-add LIMIT if missing
        if "LIMIT" not in upper:
            sql = f"{sql} LIMIT {limit}"
        with _connect(db_path) as conn:
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
        return _format_table(rows, max_rows=limit)

    if name == "execute":
        db_path = args["database"]
        sql = args["sql"].strip()
        with _connect(db_path) as conn:
            cursor = conn.execute(sql)
            conn.commit()
            affected = cursor.rowcount
        return f"OK — {affected} row{'s' if affected != 1 else ''} affected."

    if name == "count_rows":
        db_path = args["database"]
        table = args["table"]
        with _connect(db_path) as conn:
            cursor = conn.execute(f"SELECT COUNT(*) as cnt FROM [{table}]")
            cnt = cursor.fetchone()["cnt"]
        return f"{table}: {cnt} row{'s' if cnt != 1 else ''}"

    if name == "export_csv":
        db_path = args["database"]
        sql = args["sql"].strip()
        limit = args.get("limit", 1000)
        upper = sql.upper().lstrip()
        if not upper.startswith("SELECT") and not upper.startswith("WITH"):
            raise ValueError("Only SELECT/WITH queries allowed.")
        if "LIMIT" not in upper:
            sql = f"{sql} LIMIT {limit}"
        with _connect(db_path) as conn:
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
            if not rows:
                return "(no rows)"
            headers = rows[0].keys()
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(headers)
            for row in rows:
                writer.writerow([row[h] for h in headers])
        total = len(rows)
        trunc = ""
        if total >= limit:
            trunc = f"\n\n(export limited to {limit} rows)"
        return buf.getvalue().rstrip("\r\n") + trunc

    raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# Message routing
# ---------------------------------------------------------------------------

def _handle(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Route an incoming JSON-RPC message; return a response dict (or None for notifications)."""
    req_id = msg.get("id")
    method = msg.get("method")
    params = msg.get("params", {})

    # Notifications have no id — no response expected
    if method == "notifications/initialized":
        return None

    # ---- initialize ----
    if method == "initialize":
        return _make_response(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": SERVER_CAPABILITIES,
            "serverInfo": SERVER_INFO,
        })

    # ---- tools/list ----
    if method == "tools/list":
        return _make_response(req_id, {"tools": TOOLS})

    # ---- tools/call ----
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

    # ---- unknown method ----
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
