"""Agent memory subsystem package.

Moved from the retired ``src/agent1.memory`` namespace into ``agent_core``.
"""

from typing import List

from .memory_store import MemoryStore
from .types import (
    EmbeddingService,
    SQLiteStorage,
    StorageBackend,
    VectorDatabase,
    VectorEmbeddingModel,
)

__all__: List[str] = [
    "SQLiteStorage",
    "StorageBackend",
    "EmbeddingService",
    "VectorDatabase",
    "VectorEmbeddingModel",
    "MemoryStore",
]
