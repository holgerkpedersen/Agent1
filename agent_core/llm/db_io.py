from __future__ import annotations
import sqlite3
from typing import Any
from pathlib import Path

DB_FILE = "agent_memory.db"


def init_db() -> None:
    path = Path(DB_FILE)
    conn = sqlite3.connect(str(path))
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS prompt_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_type TEXT NOT NULL,
            profile_type TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            template_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS prompt_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_type TEXT NOT NULL,
            profile_type TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            success_count INTEGER NOT NULL DEFAULT 0,
            failure_count INTEGER NOT NULL DEFAULT 0,
            avg_latency_ms REAL NOT NULL DEFAULT 0.0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def save_template(task_type: str, profile_type: str, version: int, template_text: str) -> None:
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO prompt_templates (task_type, profile_type, version, template_text)
        VALUES (?, ?, ?, ?)
        """,
        (task_type, profile_type, version, template_text),
    )
    conn.commit()
    conn.close()


def load_template(task_type: str, profile_type: str, version: int) -> None | str:
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT template_text FROM prompt_templates
        WHERE task_type = ? AND profile_type = ? AND version = ?
        ORDER BY updated_at DESC LIMIT 1
        """,
        (task_type, profile_type, version),
    )
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def update_template(template_id: int, template_text: str) -> bool:
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE prompt_templates SET template_text = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (template_text, template_id),
    )
    conn.commit()
    success = cursor.rowcount > 0
    conn.close()
    return success


def save_metrics(task_type: str, profile_type: str, version: int, metrics: dict[str, Any]) -> None:
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO prompt_metrics (task_type, profile_type, version, success_count, failure_count, avg_latency_ms)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            task_type,
            profile_type,
            version,
            metrics.get("success_count", 0),
            metrics.get("failure_count", 0),
            metrics.get("avg_latency_ms", 0.0),
        ),
    )
    conn.commit()
    conn.close()


def load_metrics(task_type: str, profile_type: str, version: int) -> None | dict[str, Any]:
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT success_count, failure_count, avg_latency_ms FROM prompt_metrics
        WHERE task_type = ? AND profile_type = ? AND version = ?
        ORDER BY updated_at DESC LIMIT 1
        """,
        (task_type, profile_type, version),
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "success_count": row[0],
            "failure_count": row[1],
            "avg_latency_ms": row[2],
        }
    return None


def update_metrics(metrics_id: int, metrics: dict[str, Any]) -> bool:
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE prompt_metrics SET success_count = ?, failure_count = ?, avg_latency_ms = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (
            metrics.get("success_count", 0),
            metrics.get("failure_count", 0),
            metrics.get("avg_latency_ms", 0.0),
            metrics_id,
        ),
    )
    conn.commit()
    success = cursor.rowcount > 0
    conn.close()
    return success


def get_latest_version(task_type: str, profile_type: str) -> int | None:
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT MAX(version) FROM prompt_templates
        WHERE task_type = ? AND profile_type = ?
        """,
        (task_type, profile_type),
    )
    row = cursor.fetchone()
    conn.close()
    return row[0] if row and row[0] is not None else None


def list_versions(task_type: str, profile_type: str) -> list[int]:
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT DISTINCT version FROM prompt_templates
        WHERE task_type = ? AND profile_type = ?
        ORDER BY version ASC
        """,
        (task_type, profile_type),
    )
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]


def delete_template(template_id: int) -> bool:
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        DELETE FROM prompt_templates WHERE id = ?
        """,
        (template_id,),
    )
    conn.commit()
    success = cursor.rowcount > 0
    conn.close()
    return success


def delete_metrics(metrics_id: int) -> bool:
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        DELETE FROM prompt_metrics WHERE id = ?
        """,
        (metrics_id,),
    )
    conn.commit()
    success = cursor.rowcount > 0
    conn.close()
    return success


def get_template_by_id(template_id: int) -> None | dict[str, Any]:
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, task_type, profile_type, version, template_text FROM prompt_templates WHERE id = ?
        """,
        (template_id,),
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "id": row[0],
            "task_type": row[1],
            "profile_type": row[2],
            "version": row[3],
            "template_text": row[4],
        }
    return None


def get_metrics_by_id(metrics_id: int) -> None | dict[str, Any]:
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, task_type, profile_type, version, success_count, failure_count, avg_latency_ms FROM prompt_metrics WHERE id = ?
        """,
        (metrics_id,),
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "id": row[0],
            "task_type": row[1],
            "profile_type": row[2],
            "version": row[3],
            "success_count": row[4],
            "failure_count": row[5],
            "avg_latency_ms": row[6],
        }
    return None


def list_templates(task_type: str | None = None, profile_type: str | None = None) -> list[dict[str, Any]]:
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    if task_type and profile_type:
        cursor.execute(
            """
            SELECT id, task_type, profile_type, version, template_text FROM prompt_templates
            WHERE task_type = ? AND profile_type = ? ORDER BY version ASC
            """,
            (task_type, profile_type),
        )
    elif task_type:
        cursor.execute(
            """
            SELECT id, task_type, profile_type, version, template_text FROM prompt_templates
            WHERE task_type = ? ORDER BY version ASC
            """,
            (task_type,),
        )
    else:
        cursor.execute(
            """
            SELECT id, task_type, profile_type, version, template_text FROM prompt_templates ORDER BY version ASC
            """
        )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "id": row[0],
            "task_type": row[1],
            "profile_type": row[2],
            "version": row[3],
            "template_text": row[4],
        }
        for row in rows
    ]


def list_metrics(task_type: str | None = None, profile_type: str | None = None) -> list[dict[str, Any]]:
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    if task_type and profile_type:
        cursor.execute(
            """
            SELECT id, task_type, profile_type, version, success_count, failure_count, avg_latency_ms FROM prompt_metrics
            WHERE task_type = ? AND profile_type = ? ORDER BY version ASC
            """,
            (task_type, profile_type),
        )
    elif task_type:
        cursor.execute(
            """
            SELECT id, task_type, profile_type, version, success_count, failure_count, avg_latency_ms FROM prompt_metrics
            WHERE task_type = ? ORDER BY version ASC
            """,
            (task_type,),
        )
    else:
        cursor.execute(
            """
            SELECT id, task_type, profile_type, version, success_count, failure_count, avg_latency_ms FROM prompt_metrics ORDER BY version ASC
            """
        )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "id": row[0],
            "task_type": row[1],
            "profile_type": row[2],
            "version": row[3],
            "success_count": row[4],
            "failure_count": row[5],
            "avg_latency_ms": row[6],
        }
        for row in rows
    ]


def delete_all_templates(task_type: str | None = None, profile_type: str | None = None) -> int:
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    if task_type and profile_type:
        cursor.execute(
            """
            DELETE FROM prompt_templates WHERE task_type = ? AND profile_type = ?
            """,
            (task_type, profile_type),
        )
    elif task_type:
        cursor.execute(
            """
            DELETE FROM prompt_templates WHERE task_type = ?
            """,
            (task_type,),
        )
    else:
        cursor.execute("""DELETE FROM prompt_templates""")
    conn.commit()
    deleted = cursor.rowcount
    conn.close()
    return deleted


def delete_all_metrics(task_type: str | None = None, profile_type: str | None = None) -> int:
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    if task_type and profile_type:
        cursor.execute(
            """
            DELETE FROM prompt_metrics WHERE task_type = ? AND profile_type = ?
            """,
            (task_type, profile_type),
        )
    elif task_type:
        cursor.execute(
            """
            DELETE FROM prompt_metrics WHERE task_type = ?
            """,
            (task_type,),
        )
    else:
        cursor.execute("""DELETE FROM prompt_metrics""")
    conn.commit()
    deleted = cursor.rowcount
    conn.close()
    return deleted


def close_db(conn: sqlite3.Connection) -> None:
    conn.close()