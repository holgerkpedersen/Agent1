from typing import Dict, Any, Optional, List
import threading

from .types import (
    StorageBackend,
    SQLiteStorage,
    EmbeddingService,
    VectorDatabase,
)


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


__all__: List[str] = ["MemoryStore"]
