from typing import Dict, Any, List, Optional

import numpy as np

from src.agent1.core import EmbeddingService, VectorDatabase


class SemanticSearchEngine:
    """Semantic search interface combining embedding service and vector database."""

    def __init__(self, embedding_service: EmbeddingService, vector_db: VectorDatabase) -> None:
        self._embedding_service: EmbeddingService = embedding_service
        self._vector_db: VectorDatabase = vector_db

    def index_content(self, texts: List[str], metadata_list: List[Dict[str, Any]]) -> List[int]:
        """Index content by generating embeddings and storing them in the vector database."""
        if len(texts) != len(metadata_list):
            raise ValueError("Texts and metadata must have the same length")
        embeddings = self._embedding_service.embed_text(texts)
        vector_ids: List[int] = []
        for i in range(len(embeddings)):
            vector_id = self._vector_db.add_vector(embeddings[i], metadata_list[i])
            vector_ids.append(vector_id)
        return vector_ids

    def search(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for semantically similar content to the given query text."""
        query_embedding = self._embedding_service.embed_text([query_text])[0]
        return self._vector_db.search_similar(query_embedding, k)

    def search_with_filter(self, query_text: str, k: int = 5, filter_fn: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Search for semantically similar content with an optional metadata filter."""
        results = self.search(query_text, k * 2 if filter_fn is not None else k)
        if filter_fn is None:
            return results[:k]
        filtered_results: List[Dict[str, Any]] = []
        for item in results:
            metadata = item.get("metadata", {})
            if filter_fn(metadata):
                filtered_results.append(item)
        return filtered_results[:k]

    def reindex(self, texts: List[str], metadata_list: List[Dict[str, Any]]) -> List[int]:
        """Clear existing vectors and index new content. Returns the new vector IDs."""
        self._vector_db.cleanup()
        return self.index_content(texts, metadata_list)


__all__: List[str] = ["SemanticSearchEngine"]