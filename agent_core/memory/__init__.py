"""Agent memory subsystem package.

Moved from the retired ``src/agent1.memory`` namespace into ``agent_core``.
"""

from typing import List

from .memory_store import MemoryStore
from .types import (
    EmbeddingService,
    SEMANTIC_MEMORY_MARKER,
    SQLiteStorage,
    StorageBackend,
    VectorDatabase,
    VectorEmbeddingModel,
    load_semantic_memory,
    semantic_memory_block,
)

__all__: List[str] = [
    "SQLiteStorage",
    "StorageBackend",
    "EmbeddingService",
    "VectorDatabase",
    "VectorEmbeddingModel",
    "MemoryStore",
    "SEMANTIC_MEMORY_MARKER",
    "semantic_memory_block",
    "load_semantic_memory",
]
