from typing import Dict, Any, Optional, List, Tuple
import time
from . import (
    AgentMessage, MessageType, StorageBackend, SQLiteStorage,
    EmbeddingService, VectorDatabase,
)


class ContextEntry:
    """A single shared context entry with provenance tracking."""

    def __init__(self, key: str, value: Any, agent_id: Optional[str] = None) -> None:
        self._key: str = key
        self._value: Any = value
        self._agent_id: Optional[str] = agent_id
        self._timestamp: float = time.time()

    @property
    def key(self) -> str:
        return self._key

    @property
    def value(self) -> Any:
        return self._value

    @property
    def agent_id(self) -> Optional[str]:
        return self._agent_id

    @property
    def timestamp(self) -> float:
        return self._timestamp


class SharedContext:
    """Manages shared state between collaborating agents.

    Provides locking to prevent concurrent writes, history tracking for
    auditing changes, and optional persistence through a storage backend.
    """

    def __init__(self, storage_backend: Optional[StorageBackend] = None, max_history: int = 1000) -> None:
        self._entries: Dict[str, ContextEntry] = {}
        self._locks: Dict[str, bool] = {}
        self._lock_owners: Dict[str, str] = {}
        self._history: List[Tuple[float, str, Optional[str], Any, Any]] = []
        self._backend: Optional[StorageBackend] = storage_backend
        self._max_history = max_history

    def clear(self) -> None:
        """Clear all context entries and history."""
        self._entries.clear()
        self._locks.clear()
        self._lock_owners.clear()
        self._history.clear()
        self._max_history: int = max_history

    def set(self, key: str, value: Any, agent_id: Optional[str] = None) -> bool:
        """Set a context value. Returns False if the key is locked."""
        if self._locks.get(key, False):
            return False
        old_value: Any = self._entries[key].value if key in self._entries else None
        entry = ContextEntry(key, value, agent_id)
        self._entries[key] = entry
        self._history.append((time.time(), key, agent_id, old_value, value))
        # Limit history size
        if len(self._history) > self._max_history:
            self._history.pop(0)
        if self._backend is not None:
            payload: Dict[str, Any] = {
                "key": key, "value": value, "agent_id": agent_id or SYSTEM_ID,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self._backend.store(key, payload)
        return True

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a context value."""
        entry = self._entries.get(key)
        if entry is None and self._backend is not None:
            record = self._backend.retrieve(key)
            if record is not None:
                return record.get("value")
        return entry.value if entry is not None else None

    def delete(self, key: str, agent_id: Optional[str] = None) -> bool:
        """Remove a context value."""
        if self._locks.get(key, False):
            return False
        existed = key in self._entries
        old_value: Any = self._entries[key].value if existed else None
        if existed:
            del self._entries[key]
        self._history.append((time.time(), key, agent_id, old_value, None))
        # Limit history size
        if len(self._history) > self._max_history:
            self._history.pop(0)
        if self._backend is not None and existed:
            self._backend.delete(key)
        return existed

    def lock(self, key: str, agent_id: str) -> bool:
        """Acquire an exclusive write lock on a context key."""
        if self._locks.get(key, False):
            return False
        self._locks[key] = True
        self._lock_owners[key] = agent_id
        return True

    def unlock(self, key: str, agent_id: Optional[str] = None) -> bool:
        """Release a lock on a context key."""
        if not self._locks.get(key, False):
            return False
        owner = self._lock_owners.get(key)
        if agent_id is not None and owner != agent_id:
            return False
        self._locks[key] = False
        self._lock_owners.pop(key, None)
        return True

    def get_history(self, key: Optional[str] = None) -> List[Tuple[float, str, Optional[str], Any, Any]]:
        """Return change history for a specific key or all keys."""
        if key is None:
            return list(self._history)
        return [h for h in self._history if h[1] == key]

    def snapshot(self) -> Dict[str, Any]:
        """Capture an immutable copy of current context values."""
        return {k: e.value for k, e in self._entries.items()}


class SemanticContextIndex:
    """Indexes shared context entries for semantic similarity search.

    Uses an embedding service and vector database to enable retrieval of
    context by meaning rather than exact key matching.
    """

    def __init__(self, embedding_service: EmbeddingService, vector_db: VectorDatabase) -> None:
        self._embedding_service: EmbeddingService = embedding_service
        self._vector_db: VectorDatabase = vector_db
        self._key_to_vector_id: Dict[str, int] = {}

    def index(self, context: SharedContext, keys: Optional[List[str]] = None) -> List[int]:
        """Index selected or all context entries into the vector database."""
        target_keys = keys if keys is not None else list(context.snapshot().keys())
        texts: List[str] = []
        metadata_list: List[Dict[str, Any]] = []
        for k in target_keys:
            value = context.get(k)
            text_repr: str = str(value) if value is not None else ""
            texts.append(text_repr)
            metadata_list.append({"key": k})
        embeddings = self._embedding_service.embed_text(texts)
        vector_ids: List[int] = []
        for i in range(len(embeddings)):
            vid = self._vector_db.add_vector(embeddings[i], metadata_list[i])
            if target_keys[i] not in self._key_to_vector_id:
                self._key_to_vector_id[target_keys[i]] = vid
            vector_ids.append(vid)
        return vector_ids

    def search(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search indexed context entries by semantic similarity."""
        embedding = self._embedding_service.embed_text([query_text])[0]
        results = self._vector_db.search_similar(embedding, k)
        return [r for r in results if "metadata" in r and "key" in r["metadata"]]


class ContextManager:
    """Top-level coordinator combining shared context with persistence.

    Wraps a SharedContext backed by SQLiteStorage and optionally provides
    semantic search via an EmbeddingService + VectorDatabase pair.
    """

    def __init__(self, db_path: str = ":memory:", max_history: int = 1000) -> None:
        self._storage: SQLiteStorage = SQLiteStorage(db_path)
        self._context: SharedContext = SharedContext(self._storage, max_history=max_history)
        self._semantic_index: Optional[SemanticContextIndex] = None

    @property
    def context(self) -> SharedContext:
        return self._context

    @property
    def storage(self) -> SQLiteStorage:
        return self._storage

    def enable_semantic_search(self, embedding_service: EmbeddingService, vector_db: VectorDatabase) -> None:
        """Enable semantic similarity search over shared context."""
        self._semantic_index = SemanticContextIndex(embedding_service, vector_db)

    def index_context(self, keys: Optional[List[str]] = None) -> List[int]:
        """Re-index context entries for semantic search."""
        if self._semantic_index is None:
            raise RuntimeError("Semantic search not enabled")
        return self._semantic_index.index(self._context, keys)

    def semantic_search(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search shared context by meaning."""
        if self._semantic_index is None:
            raise RuntimeError("Semantic search not enabled")
        return self._semantic_index.search(query_text, k)

    def broadcast(self, message_type: MessageType, content: Dict[str, Any], sender_id: str = "context_manager") -> AgentMessage:
        """Create an AgentMessage reflecting a context change."""
        return AgentMessage(
            sender_id=sender_id, receiver_id=None, message_type=message_type,
            content=content, timestamp=time.time(),
        )

    def cleanup(self) -> None:
        """Release internal caches held by the context manager."""
        self._context.clear()


SYSTEM_ID = "system"

__all__: List[str] = [
    "ContextEntry", "SharedContext", "SemanticContextIndex", "ContextManager",
    "SYSTEM_ID",
]