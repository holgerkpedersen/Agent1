"""Memory subsystem data structures: storage backends and vector search.

Moved verbatim from the retired ``src/agent1.core`` namespace so the memory
stack is self-contained inside ``agent_core``.
"""
import logging  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class StorageBackend(Protocol):
    def store(self, key: str, data: Dict[str, Any]) -> None: ...
    def retrieve(self, key: str) -> Optional[Dict[str, Any]]: ...
    def delete(self, key: str) -> bool: ...


import json  # noqa: E402  (kept next to its only user, as in the original)
import os  # noqa: E402  (used by EmbeddingService for AGENT_EMBEDDING_MODEL)
import sqlite3  # noqa: E402
import time  # noqa: E402


class SQLiteStorage:
    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn: sqlite3.Connection = sqlite3.connect(db_path, check_same_thread=False)
        self._setup_schema()

    def _setup_schema(self) -> None:
        cursor = self._conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_entries (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                data TEXT NOT NULL,
                tags TEXT
            )
        """)
        self._conn.commit()

    def store(self, key: str, data: Dict[str, Any]) -> None:
        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO memory_entries (id, agent_id, timestamp, data, tags)
            VALUES (?, ?, ?, ?, ?)
        """, (key, str(data.get("agent_id", "unknown")),
              time.strftime("%Y-%m-%dT%H:%M:%S"), json.dumps(data), ""))
        self._conn.commit()

    def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        cursor = self._conn.cursor()
        cursor.execute("SELECT data FROM memory_entries WHERE id = ?", (key,))
        result = cursor.fetchone()
        if result is None:
            return None
        return json.loads(result[0])

    def delete(self, key: str) -> bool:
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM memory_entries WHERE id = ?", (key,))
        deleted = cursor.rowcount > 0
        self._conn.commit()
        return deleted


class VectorEmbeddingModel(Protocol):
    def encode(self, texts: List[str]) -> np.ndarray: ...


class EmbeddingService:
    """Real semantic embedding service via LM Studio /v1/embeddings.

    Uses the optional backend from :mod:`agent_core.utils.module_similarity`,
    which probes LM Studio for an embedding-capable model and then calls
    /v1/embeddings.  Returns 384-dim float vectors (all-MiniLM-L6-v2).
    """

    def __init__(self) -> None:
        # Model name from AGENT_EMBEDDING_MODEL env var (or "" for unavailable)
        self._model_name: str = os.environ.get("AGENT_EMBEDDING_MODEL", "")
        self._backend = None  # lazily created

    def _ensure_backend(self) -> Any:
        """Create the LM Studio embedding backend on first use."""
        if self._backend is None:
            try:
                from agent_core.utils.module_similarity import _EmbeddingBackend
                self._backend = _EmbeddingBackend()
            except Exception:
                # Fail-open: backend unavailable but service still works
                self._backend = None
        return self._backend

    def embed_text(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts into 384-dim vectors using LM Studio."""
        backend = self._ensure_backend()
        if backend is None or not backend.available:
            # Stub fallback: zeros (never raises, fail-open)
            return np.zeros((len(texts), 384))
        try:
            return backend.embed(texts)
        except Exception:
            # Fail-open: fallback to zeros on any backend error
            return np.zeros((len(texts), 384))


class VectorDatabase:
    def __init__(self, dimension: int = 384) -> None:
        self._dimension: int = dimension
        self._vectors: Dict[int, np.ndarray] = {}
        self._id_to_metadata: Dict[int, Dict[str, Any]] = {}
        self._next_id: int = 0

    def add_vector(self, vector: np.ndarray, metadata: Dict[str, Any]) -> int:
        if len(vector) != self._dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self._dimension}, got {len(vector)}")
        normalized_vector = vector / (np.linalg.norm(vector) + 1e-8)
        vector_id = self._next_id
        self._vectors[vector_id] = normalized_vector
        self._id_to_metadata[vector_id] = metadata
        self._next_id += 1
        return vector_id

    def search_similar(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        normalized_query = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        results: List[tuple[float, int]] = []
        for vid, vec in self._vectors.items():
            similarity = float(np.dot(normalized_query, vec))
            results.append((similarity, vid))
        results.sort(key=lambda x: x[0], reverse=True)
        top_results: List[Dict[str, Any]] = []
        for i in range(min(k, len(results))):
            similarity, idx = results[i]
            if idx >= 0 and idx in self._id_to_metadata:
                result_item = {
                    "metadata": self._id_to_metadata[idx],
                    "similarity_score": similarity,
                    "vector_id": idx
                }
                top_results.append(result_item)
        return top_results


# ---------------------------------------------------------------------------
# System-prompt injection helper (used by agent._strip_dynamic_system_blocks)
# ---------------------------------------------------------------------------
#: Marker prefixing the injected semantic-memory block.  ``agent.py`` strips
#: everything from this marker onward before re-injecting a fresh block, so a
#: long-lived session never accumulates stale copies (same contract as
#: ``HABITS_MARKER`` / ``SKILL_INDEX_MARKER``).
SEMANTIC_MEMORY_MARKER = "\n\nSEMANTIC MEMORY"

#: Workspace-local ledger holding indexed semantic memories.
SEMANTIC_MEMORY_FILENAME = ".semantic_memory.json"

#: Hard cap on rendered lines so the block cannot bloat the system prompt.
MAX_BLOCK_LINES = 8

#: Marker prefixing the pinned "original task" block.  ``agent.py`` strips
#: everything from this marker onward before re-injecting a fresh block, so a
#: long-lived session never accumulates stale copies (same contract as
#: ``SEMANTIC_MEMORY_MARKER`` / ``HABITS_MARKER`` / ``SKILL_INDEX_MARKER``).
ORIGINAL_GOAL_MARKER = "\n\nORIGINAL TASK"

#: Closing boilerplate of the ORIGINAL TASK block (see
#: :func:`original_goal_block`).  Extracted as a constant so
#: :func:`original_goal_from_block` can recover the goal text from a restored
#: system prompt without the two ever drifting apart.
ORIGINAL_GOAL_SUFFIX = (
    "\n\nThis is the user's original request for this session. Do not "
    "lose sight of it - every answer must stay aligned with it.\n"
)

#: How much of the user's first prompt is kept as the pinned goal.  A first
#: prompt that is itself a file dump would otherwise swallow the context
#: budget it is supposed to protect.
MAX_GOAL_CHARS = 4000


def _semantic_memory_block(
    memories: Sequence[Any] | None,
    query: str | None = None,
    *,
    k: int = 3,
    line_cap: int = 10,
) -> str:
    """Inject a SEMANTIC MEMORY block into the system prompt (empty when none).

    Empty input (``[]``, ``""``, ``None``) returns the empty string so a fresh
    workspace prompt stays byte-identical to the pre-KG baseline.
    Accepts dicts with ``metadata`` and ``similarity_score`` (from
    ``VectorDatabase.search_similar``) or plain strings (e.g. wiki notes).

    When a query is provided, the block is prefixed with ``QUERY: ...`` for
    context-aware relevance.
    """
    if not memories:
        return ""
    if isinstance(memories, str):
        memories = [memories]
    lines: list[str] = []
    if query:
        lines.append(f"QUERY: {query[:140]}")
    for mem in memories[:k]:
        if isinstance(mem, Mapping):
            md = mem.get("metadata") or {}
            score = mem.get("similarity_score", 0.0)
            text = str(md.get("text") or "").strip()[:120]
            when = md.get("timestamp", "?")
            lines.append(f"  - {text} (score {score:.2f}, when {when})")
        else:
            text = str(mem).strip()[:120]
            lines.append(f"  - {text}")
        if len(lines) >= line_cap:
            break
    if not lines:
        return ""
    block = SEMANTIC_MEMORY_MARKER + "\n" + "\n".join(lines) + "\n"
    assert block.startswith(SEMANTIC_MEMORY_MARKER)  # marker contract
    return block


def semantic_memory_block(
    memories: Sequence[Any] | None,
    query: str | None = None,
    *,
    k: int = MAX_BLOCK_LINES,
) -> str:
    """Public wrapper around :func:`_semantic_memory_block`.

    ``agent.Agent._semantic_memory_block`` calls this with the ledger loaded by
    :func:`load_semantic_memory`.  ``k`` is clamped to ``MAX_BLOCK_LINES`` so a
    caller cannot accidentally bloat the system prompt.
    """
    return _semantic_memory_block(
        memories,
        query,
        k=min(k, MAX_BLOCK_LINES),
        line_cap=MAX_BLOCK_LINES + 1,
    )


def original_goal_block(
    goal: str | None, *, max_chars: int = MAX_GOAL_CHARS
) -> str:
    """Inject the session's ORIGINAL TASK block into the system prompt.

    The chat-history projection trims the OLDEST body messages
    (``_MAX_CHAT_MESSAGES`` / ``_HISTORY_CHAR_BUDGET``) — exactly where the
    user's first prompt lives — so without this block the agent silently
    forgets the task it was given.  Keeping the goal in the system prompt
    (position 0, always kept and never trimmed) makes it survive compaction
    and session restarts.

    The text is whitespace-normalised and capped at *max_chars*, so a first
    prompt that is itself a pasted file cannot blow up the system prompt.
    Returns the empty string when there is no goal, so a session without one
    gets a byte-identical prompt to the pre-goal baseline.
    """
    if not goal:
        return ""
    text = " ".join(str(goal).split())[:max_chars].strip()
    if not text:
        return ""
    block = (
        ORIGINAL_GOAL_MARKER
        + "\n"
        + text
        + ORIGINAL_GOAL_SUFFIX
    )
    assert block.startswith(ORIGINAL_GOAL_MARKER)  # marker contract
    return block


def original_goal_from_block(
    prompt: str | None, *, max_chars: int = MAX_GOAL_CHARS
) -> str:
    """Recover the pinned goal text from a system prompt ("" when absent).

    The inverse of :func:`original_goal_block`: the goal is persisted inside
    the system message of ``chat_history.json``, so when ``agent_memory.json``
    is lost or corrupt the ORIGINAL TASK block is the most faithful recovery
    source available (full text, unlike the 300-char compaction-note excerpt).

    The goal text is whitespace-normalised and single-line by construction, so
    the first line of the block body is the goal even when the closing
    boilerplate cannot be found.  Never raises.
    """
    try:
        text = str(prompt or "")
        if ORIGINAL_GOAL_MARKER not in text:
            return ""
        tail = text.split(ORIGINAL_GOAL_MARKER, 1)[1]
        idx = tail.find(ORIGINAL_GOAL_SUFFIX)
        body = tail[:idx] if idx >= 0 else tail
        first_line = body.strip().splitlines()[0].strip() if body.strip() else ""
        return first_line[:max_chars].strip()
    except Exception:  # noqa: BLE001 - recovery must not kill a session
        return ""


# ---------------------------------------------------------------------------
# Persistence (workspace-local .semantic_memory.json)
# ---------------------------------------------------------------------------

def _semantic_memory_path(workspace: str | Path) -> Path:
    """The workspace's ``.semantic_memory.json`` ledger path."""
    return Path(workspace) / SEMANTIC_MEMORY_FILENAME


def load_semantic_memory(workspace: str | Path) -> list[dict[str, Any]]:
    """Load indexed semantic memories from the workspace ledger.

    NEVER raises (``[]`` on any failure), so a broken ledger cannot kill a chat
    turn.  A corrupt file is quarantined to ``.semantic_memory.json.bad-<ts>``
    so the bytes stay inspectable (same rule as :func:`load_habits`).

    Returns a list of ``{"metadata": {...}, "similarity_score": float}`` dicts
    ready for :func:`semantic_memory_block`.
    """
    path = _semantic_memory_path(workspace)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return []
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quarantine = f"{path}.bad-{stamp}"
        try:
            os.replace(path, quarantine)
        except OSError:
            logger.warning("Failed to quarantine corrupt semantic memory at %s", path)
        else:
            logger.warning("Corrupt semantic memory moved to %s", quarantine)
        return []
    except Exception:  # noqa: BLE001 - contract: never raise
        logger.exception("Semantic memory load unavailable:\n")
        return []

    if not isinstance(data, list):
        return []
    memories: list[dict[str, Any]] = []
    for entry in data:
        if isinstance(entry, Mapping):
            metadata = entry.get("metadata")
            if not isinstance(metadata, Mapping):
                metadata = {"text": entry.get("text", "")}
            text = str(metadata.get("text") or "").strip()
            if not text:
                continue
            memories.append(
                {
                    "metadata": dict(metadata),
                    "similarity_score": float(entry.get("similarity_score") or 0.0),
                }
            )
        elif isinstance(entry, str) and entry.strip():
            memories.append(
                {"metadata": {"text": entry.strip()}, "similarity_score": 0.0}
            )
    return memories


__all__ = [
    "StorageBackend",
    "SQLiteStorage",
    "VectorEmbeddingModel",
    "EmbeddingService",
    "VectorDatabase",
    "SEMANTIC_MEMORY_MARKER",
    "SEMANTIC_MEMORY_FILENAME",
    "MAX_BLOCK_LINES",
    "ORIGINAL_GOAL_MARKER",
    "ORIGINAL_GOAL_SUFFIX",
    "MAX_GOAL_CHARS",
    "original_goal_block",
    "original_goal_from_block",
    "_semantic_memory_block",
    "semantic_memory_block",
    "load_semantic_memory",
]
