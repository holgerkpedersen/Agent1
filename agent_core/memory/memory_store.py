from typing import Dict, Any, Mapping, Optional, List, Sequence
import logging
import threading

import numpy as np

from .types import (
    StorageBackend,
    SQLiteStorage,
    EmbeddingService,
    VectorDatabase,
)

logger = logging.getLogger(__name__)

#: Per-entry embedding cache.  Keyed by ``(id(service), text)``: an unchanged
#: ledger entry is embedded ONCE per process, so a turn that appends one line
#: costs one embedding instead of a full re-embed.  The previous whole-ledger
#: cache was keyed by a fingerprint of ALL entries, so one appended line
#: invalidated it and re-embedded the entire ledger through LM Studio.
_SEMANTIC_VECTOR_CACHE: Dict[tuple, Any] = {}
_SEMANTIC_VECTOR_CACHE_LOCK = threading.Lock()
_SEMANTIC_VECTOR_CACHE_MAX = 2048

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
        return self.index_vectors(embeddings, metadata_list)

    def index_vectors(
        self, embeddings: Any, metadata_list: List[Dict[str, Any]]
    ) -> List[int]:
        """Store already-computed *embeddings* for later semantic retrieval.

        The counterpart of :meth:`index_texts` for a caller that caches
        per-entry vectors: it re-uses each stored vector instead of paying for
        the embedding again.
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("embeddings and metadata_list must have equal length")
        if self._vector_db is None:
            raise RuntimeError("Semantic search components not configured")
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
    """Drop all cached embedding vectors (tests / memory hygiene).

    Kept under its original name so existing callers and tests keep working;
    the cache it clears is now per-entry rather than per-ledger.
    """
    with _SEMANTIC_VECTOR_CACHE_LOCK:
        _SEMANTIC_VECTOR_CACHE.clear()


def _vector_for(service: Any, text: str) -> Any:
    """The embedding vector for *text*, memoised per ``(service, text)``.

    ``semantic_search_memories`` runs on EVERY chat turn, and
    :func:`agent_core.memory.distill.record_turn` now writes the ledger every
    turn — so a whole-ledger cache keyed by a fingerprint of all entries was
    invalidated by a single appended line and re-embedded the entire ledger
    through LM Studio.  Keying the cache by entry TEXT means an unchanged
    entry is embedded once for the life of the process and a write costs
    exactly one new embedding.
    """
    key = (id(service), text)
    with _SEMANTIC_VECTOR_CACHE_LOCK:
        cached = _SEMANTIC_VECTOR_CACHE.get(key)
    if cached is not None:
        return cached
    vector = service.embed_text([text])[0]
    with _SEMANTIC_VECTOR_CACHE_LOCK:
        if len(_SEMANTIC_VECTOR_CACHE) >= _SEMANTIC_VECTOR_CACHE_MAX:
            _SEMANTIC_VECTOR_CACHE.pop(next(iter(_SEMANTIC_VECTOR_CACHE)))
        _SEMANTIC_VECTOR_CACHE[key] = vector
    return vector


def _is_zero_vector(vector: Any) -> bool:
    """True when *vector* is all zeros (the embedder's fail-open stub)."""
    try:
        return not bool(np.any(vector))
    except Exception:  # noqa: BLE001 - unknown shape counts as unusable
        return True


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
    by :func:`agent_core.memory.types.semantic_memory_block`), embeds them and
    retrieves the top-*k* with the query embedding.  The result shape is
    ``{"metadata": {...}, "similarity_score": float, ...}`` — ready for the
    SEMANTIC MEMORY prompt block.

    Embeddings are memoised per entry TEXT (:func:`_vector_for`), so a ledger
    that grows by one entry per turn costs one new embedding, not a full
    re-embed.

    Fail-open: returns ``[]`` whenever no meaningful ranking is available
    (embeddings backend down — every cosine score would be exactly ``0.0`` on
    zero vectors — broken ledger, backend error, ...), so callers can fall
    back to unranked ledger order instead of dying mid-turn.

    ``embedding_service`` / ``vector_db`` default to the real
    :class:`EmbeddingService` (LM Studio) and an in-memory
    :class:`VectorDatabase`; tests inject a deterministic embedder.
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
        store = MemoryStore(SQLiteStorage(":memory:"))
        store.configure_semantic_search(
            service, vector_db if vector_db is not None else VectorDatabase()
        )
        vectors = [_vector_for(service, text) for text in texts]
        if all(_is_zero_vector(vector) for vector in vectors):
            # Zero vectors (embeddings unavailable) score exactly 0.0 for every
            # entry — no meaningful ranking, so report "none" and let the
            # caller fall back to its lexical order.  Nothing is cached either:
            # the backend may come back and must be retried.
            return []
        store.index_vectors(vectors, metadata_list)
        results = store.semantic_search(query_text, k)
        if not results or all(
            float(item.get("similarity_score") or 0.0) == 0.0 for item in results
        ):
            return []
        return results
    except Exception:  # noqa: BLE001 - fail-open contract: never raise
        logger.debug("semantic_search_memories unavailable", exc_info=True)
        return []


__all__: List[str] = [
    "MemoryStore",
    "semantic_search_memories",
    "clear_semantic_index_cache",
]
