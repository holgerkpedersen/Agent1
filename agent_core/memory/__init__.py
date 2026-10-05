"""Agent memory subsystem package.

Moved from the retired ``src/agent1.memory`` namespace into ``agent_core``.
"""

from typing import List

from .attribution import (
    ATTRIBUTION_COLUMNS,
    CREATE_EXPERIENCES_SQL,
    CREATE_LLM_DECISIONS_SQL,
    EXPERIENCE_COLUMNS,
    attribution_summary,
    decision_latency_histogram,
    ensure_attribution_schema,
    experiences_by_llm,
    prompt_sha256,
    resolve_decision_llm,
    success_rate_by_model,
)
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
    # LLM decision attribution (observability only — decision #014).
    "EXPERIENCE_COLUMNS",
    "ATTRIBUTION_COLUMNS",
    "CREATE_EXPERIENCES_SQL",
    "CREATE_LLM_DECISIONS_SQL",
    "ensure_attribution_schema",
    "prompt_sha256",
    "resolve_decision_llm",
    "experiences_by_llm",
    "success_rate_by_model",
    "decision_latency_histogram",
    "attribution_summary",
]
