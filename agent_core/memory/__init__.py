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
from .memory_store import (
    MemoryStore,
    clear_semantic_index_cache,
    semantic_search_memories,
)
from .scoring import rank_memories, fuse
from .distill import consolidate, extract_signals, record_turn
from .types import (
    EmbeddingService,
    MAX_ENTRIES,
    MAX_ENTRY_CHARS,
    MAX_GOAL_CHARS,
    MEMORY_SCHEMA_VERSION,
    ORIGINAL_GOAL_MARKER,
    ORIGINAL_GOAL_SUFFIX,
    SEMANTIC_MEMORY_MARKER,
    SQLiteStorage,
    StorageBackend,
    VectorDatabase,
    VectorEmbeddingModel,
    load_semantic_memory,
    original_goal_block,
    original_goal_from_block,
    save_semantic_memory,
    semantic_memory_block,
    set_ledger_override,
)

__all__: List[str] = [
    "SQLiteStorage",
    "StorageBackend",
    "EmbeddingService",
    "VectorDatabase",
    "VectorEmbeddingModel",
    "MemoryStore",
    "semantic_search_memories",
    "clear_semantic_index_cache",
    "rank_memories",
    "fuse",
    "record_turn",
    "extract_signals",
    "consolidate",
    "SEMANTIC_MEMORY_MARKER",
    "semantic_memory_block",
    "load_semantic_memory",
    "save_semantic_memory",
    "set_ledger_override",
    "MAX_ENTRIES",
    "MAX_ENTRY_CHARS",
    "MEMORY_SCHEMA_VERSION",
    "ORIGINAL_GOAL_MARKER",
    "ORIGINAL_GOAL_SUFFIX",
    "MAX_GOAL_CHARS",
    "original_goal_block",
    "original_goal_from_block",
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
