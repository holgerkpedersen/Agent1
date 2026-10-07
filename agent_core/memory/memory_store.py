from typing import Dict, Any, Mapping, Optional, List, Sequence
import json
import logging
import threading

from .types import (
    StorageBackend,
    SQLiteStorage,
    EmbeddingService,
    VectorDatabase,
)

logger = logging.getLogger(__name__)

# Cache of embedded ledger indexes, keyed by (embedding-service id, vector-db
# id, ledger fingerprint).  ``semantic_search_memories`` runs on EVERY chat
# turn, so re-embedding an unchanged ledger each turn would add needless LM
# Studio round-trips; the fingerprint entry ensures a ledger changed
# mid-session IS re-embedded on the next turn.
_SEMANTIC_INDEX_CACHE: Dict[tuple, "MemoryStore"] = {}
_SEMANTIC_CACHE_LOCK = threading.Lock()
_SEMANTIC_CACHE_MAX = 8

# Stable default embedding service: the cache key contains ``id(service)``, so
# the default must be a singleton — a fresh instance per call would never hit
# the cache (and each would re-probe LM Studio).  The vector database is the
# opposite: every cache entry OWNS a fresh one, or vectors from different
# ledgers would leak into each other's searches.
_DEFAULT_EMBEDDING_SERVICE: Optional[EmbeddingService] = None


class MemoryStore:
    """Persistent memory storage interface with caching and optional semantic search."""

    def __init__(
        self,
        backend: Optional[StorageBackend] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vector_db: Optional[VectorDatabase] = None,
    ) -> None:
        if backend is None:
            backend = SQLiteStorage()
        self._backend: StorageBackend = backend
        self._embedding_service: Optional[EmbeddingService] = embedding_service
        self._vector_db: Optional[VectorDatabase] = vector_db
        self._cache: Dict[str, Any] = {}
        self._lock: threading.Lock = threading.Lock()

    def save_memory(self, key: str, data: Dict[str, Any]) -> None:
        """Persist a memory entry and update the in-memory cache."""
        with self._lock:
            self._backend.store(key, data)
            self._cache[key] = data

    def load_memory(self, key: str) -> Optional[Any]:
        """Retrieve a memory entry from cache or backing storage."""
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            result = self._backend.retrieve(key)
            if result is not None:
                self._cache[key] = result
            return result

    def forget_memory(self, key: str) -> bool:
        """Delete a memory entry from storage and cache."""
        with self._lock:
            success = self._backend.delete(key)
            if key in self._cache:
                del self._cache[key]
            return success

    def configure_semantic_search(
        self, embedding_service: EmbeddingService, vector_db: VectorDatabase
    ) -> None:
        """Enable semantic search by wiring an embedding service and vector database."""
        with self._lock:
            self._embedding_service = embedding_service
            self._vector_db = vector_db

    def index_texts(
        self, texts: List[str], metadata_list: List[Dict[str, Any]]
    ) -> List[int]:
        """Embed and store a batch of texts for later semantic retrieval."""
        if len(texts) != len(metadata_list):
            raise ValueError("texts and metadata_list must have equal length")
        if self._embedding_service is None or self._vector_db is None:
            raise RuntimeError("Semantic search components not configured")
        embeddings = self._embedding_service.embed_text(texts)
        vector_ids: List[int] = []
        for i in range(len(embeddings)):
            vid = self._vector_db.add_vector(embeddings[i], metadata_list[i])
            vector_ids.append(vid)
        return vector_ids

    def semantic_search(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Find semantically similar indexed memories for a query."""
        if self._embedding_service is None or self._vector_db is None:
            raise RuntimeError("Semantic search components not configured")
        query_embedding = self._embedding_service.embed_text([query_text])[0]
        return self._vector_db.search_similar(query_embedding, k)

    def clear_cache(self) -> None:
        """Drop all cached entries without touching backing storage."""
        with self._lock:
            self._cache.clear()


def clear_semantic_index_cache() -> None:
    """Drop all cached embedding indexes (tests / memory hygiene)."""
    with _SEMANTIC_CACHE_LOCK:
        _SEMANTIC_INDEX_CACHE.clear()


def semantic_search_memories(
    memories: Sequence[Any],
    query: str,
    k: int = 5,
    *,
    embedding_service: Optional[EmbeddingService] = None,
    vector_db: Optional[VectorDatabase] = None,
) -> List[Dict[str, Any]]:
    """Rank *memories* by semantic similarity to *query* (most similar first).

    Production wrapper around :meth:`MemoryStore.semantic_search`: normalises
    ledger entries (``{"metadata": {...}}`` dicts or plain strings, as accepted
    by :func:`agent_core.memory.types.semantic_memory_block`), embeds them via
    ``index_texts`` and retrieves the top-*k* with the query embedding.  The
    result shape is ``{"metadata": {...}, "similarity_score": float, ...}`` —
    ready for the SEMANTIC MEMORY prompt block.

    Fail-open: returns ``[]`` whenever no meaningful ranking is available
    (embeddings backend down — every cosine score would be exactly ``0.0`` on
    zero vectors — broken ledger, backend error, ...), so callers can fall
    back to unranked ledger order instead of dying mid-turn.

    ``embedding_service`` / ``vector_db`` default to the real
    :class:`EmbeddingService` (LM Studio) and an in-memory
    :class:`VectorDatabase`; tests inject a deterministic embedder.  The
    resulting index is cached per ledger fingerprint (see
    :func:`clear_semantic_index_cache`).
    """
    try:
        if isinstance(memories, str):
            memories = [memories]
        texts: List[str] = []
        metadata_list: List[Dict[str, Any]] = []
        for mem in memories or []:
            if isinstance(mem, Mapping):
                metadata = dict(mem.get("metadata") or {})
                text = str(metadata.get("text") or "").strip()
                if not text:
                    continue
                metadata.setdefault("text", text)
                texts.append(text)
                metadata_list.append(metadata)
            else:
                text = str(mem).strip()
                if text:
                    texts.append(text)
                    metadata_list.append({"text": text})
        query_text = str(query or "").strip()
        if not query_text or not texts:
            return []
        global _DEFAULT_EMBEDDING_SERVICE
        if embedding_service is not None:
            service = embedding_service
        else:
            if _DEFAULT_EMBEDDING_SERVICE is None:
                _DEFAULT_EMBEDDING_SERVICE = EmbeddingService()
            service = _DEFAULT_EMBEDDING_SERVICE
        fingerprint = tuple(
            (text, json.dumps(md, sort_keys=True, default=str))
            for text, md in zip(texts, metadata_list)
        )
        # ``None`` marks the default case: each cache entry then owns a FRESH
        # VectorDatabase (a shared one would leak vectors across ledgers).
        key = (id(service), id(vector_db) if vector_db is not None else None, fingerprint)
        with _SEMANTIC_CACHE_LOCK:
            store = _SEMANTIC_INDEX_CACHE.get(key)
        if store is None:
            store = MemoryStore(SQLiteStorage(":memory:"))
            store.configure_semantic_search(
                service,
                vector_db if vector_db is not None else VectorDatabase(),
            )
            store.index_texts(texts, metadata_list)
        results = store.semantic_search(query_text, k)
        if not results or all(
            float(item.get("similarity_score") or 0.0) == 0.0 for item in results
        ):
            # Zero vectors (embeddings unavailable) score exactly 0.0 for every
            # entry — no meaningful ranking, so report "none" and let the
            # caller fall back to unranked ledger order.  Do NOT cache such an
            # index either: the backend may come back and must be retried.
            return []
        with _SEMANTIC_CACHE_LOCK:
            if key not in _SEMANTIC_INDEX_CACHE:
                if len(_SEMANTIC_INDEX_CACHE) >= _SEMANTIC_CACHE_MAX:
                    _SEMANTIC_INDEX_CACHE.pop(next(iter(_SEMANTIC_INDEX_CACHE)))
                _SEMANTIC_INDEX_CACHE[key] = store
        return results
    except Exception:  # noqa: BLE001 - fail-open contract: never raise
        logger.debug("semantic_search_memories unavailable", exc_info=True)
        return []


__all__: List[str] = [
    "MemoryStore",
    "semantic_search_memories",
    "clear_semantic_index_cache",
]
