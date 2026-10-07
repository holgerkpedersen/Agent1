"""Wiring tests: ``MemoryStore.semantic_search`` in the chat turn.

The semantic stack (``EmbeddingService`` + ``VectorDatabase`` +
``MemoryStore.semantic_search``) existed but was never called from production —
the SEMANTIC MEMORY block dumped the first ``k`` ledger entries unranked.
These tests pin the new query-aware path:

* :func:`agent_core.memory.memory_store.semantic_search_memories` ranks ledger
  entries by embedding similarity to the current user query (real
  ``MemoryStore.index_texts`` + ``MemoryStore.semantic_search`` code path).
* ``Agent._semantic_memory_block(query)`` injects the ranked top-k prefixed
  with a ``QUERY:`` line, and falls back to the old unranked block (no query
  line, byte-compatible) when embeddings are unavailable.

The embedding backend is the only test double (LM Studio is external); the
vector maths, ranking and prompt assembly are all production code.
"""
from __future__ import annotations

import json

import numpy as np

from agent_core.memory import EmbeddingService, SEMANTIC_MEMORY_MARKER
from agent_core.memory.memory_store import (
    clear_semantic_index_cache,
    semantic_search_memories,
)


class FakeEmbedder(EmbeddingService):
    """Deterministic bag-of-keywords embedder (stand-in for LM Studio).

    Maps a tiny vocabulary onto unit axes of a 384-dim vector; unknown words
    share a "misc" axis.  Cosine similarity then reflects keyword overlap, so
    rankings are exact and reproducible without any network.
    """

    VOCAB = ("pytest", "deploy", "database")

    def embed_text(self, texts):  # type: ignore[override]
        out = np.zeros((len(texts), 384))
        for i, text in enumerate(texts):
            for word in str(text).lower().split():
                axis = 383  # misc
                for j, key in enumerate(self.VOCAB):
                    if key in word:
                        axis = j
                        break
                out[i, axis] += 1.0
        return out


class ZeroEmbedder(EmbeddingService):
    """Embeddings unavailable (LM Studio down): fail-open zero vectors."""

    def embed_text(self, texts):  # type: ignore[override]
        return np.zeros((len(texts), 384))


class BrokenEmbedder(EmbeddingService):
    """Backend blow-up mid-call (must never kill the turn)."""

    def embed_text(self, texts):  # type: ignore[override]
        raise RuntimeError("embedding backend exploded")


def _memories() -> list[dict]:
    return [
        {"metadata": {"text": "user prefers pytest"}, "similarity_score": 0.0},
        {"metadata": {"text": "deploy uses blue-green"}, "similarity_score": 0.0},
        {"metadata": {"text": "database migrations run nightly"}, "similarity_score": 0.0},
    ]


def setup_function() -> None:
    clear_semantic_index_cache()


# ---------------------------------------------------------------------------
# semantic_search_memories (production helper around MemoryStore.semantic_search)
# ---------------------------------------------------------------------------


def test_search_ranks_most_similar_first() -> None:
    ranked = semantic_search_memories(
        _memories(), "pytest failures in the suite", k=3,
        embedding_service=FakeEmbedder(),
    )
    texts = [m["metadata"]["text"] for m in ranked]
    assert texts[0] == "user prefers pytest"
    assert set(texts) == {
        "user prefers pytest",
        "deploy uses blue-green",
        "database migrations run nightly",
    }
    assert ranked[0]["similarity_score"] > ranked[1]["similarity_score"]


def test_search_respects_k() -> None:
    ranked = semantic_search_memories(
        _memories(), "pytest", k=1, embedding_service=FakeEmbedder()
    )
    assert len(ranked) == 1


def test_search_returns_empty_when_embeddings_unavailable() -> None:
    """Zero vectors => no meaningful ranking => [] (caller falls back)."""
    assert semantic_search_memories(
        _memories(), "pytest", embedding_service=ZeroEmbedder()
    ) == []


def test_search_never_raises_on_broken_backend() -> None:
    assert semantic_search_memories(
        _memories(), "pytest", embedding_service=BrokenEmbedder()
    ) == []


def test_search_empty_without_query_or_memories() -> None:
    assert semantic_search_memories([], "pytest", embedding_service=FakeEmbedder()) == []
    assert semantic_search_memories(_memories(), "", embedding_service=FakeEmbedder()) == []
    assert semantic_search_memories(_memories(), "   ", embedding_service=FakeEmbedder()) == []


def test_search_accepts_plain_string_memories() -> None:
    ranked = semantic_search_memories(
        ["alpha pytest note", "deploy beta"], "pytest",
        embedding_service=FakeEmbedder(),
    )
    assert ranked[0]["metadata"]["text"] == "alpha pytest note"


def test_search_reindexes_after_memory_added() -> None:
    """A memory indexed mid-session must be found on the next turn.

    Pins the cache-invalidation contract: the per-workspace index is keyed on
    the ledger contents, so a changed ledger is re-embedded.
    """
    embedder = FakeEmbedder()
    first = semantic_search_memories(
        _memories(), "database rollback", embedding_service=embedder
    )
    # Baseline: the only database-related memory ranks first.
    assert first[0]["metadata"]["text"] == "database migrations run nightly"

    grown = _memories() + [
        {"metadata": {"text": "database rollback checklist"}, "similarity_score": 0.0}
    ]
    second = semantic_search_memories(
        grown, "database rollback", embedding_service=embedder
    )
    assert second[0]["metadata"]["text"] == "database rollback checklist"


def test_search_skips_blank_texts() -> None:
    ranked = semantic_search_memories(
        [{"metadata": {"text": "   "}}, {"metadata": {"text": "pytest rocks"}}],
        "pytest", embedding_service=FakeEmbedder(),
    )
    assert [m["metadata"]["text"] for m in ranked] == ["pytest rocks"]


# ---------------------------------------------------------------------------
# Agent wiring: query-aware SEMANTIC MEMORY block
# ---------------------------------------------------------------------------


def _bare_agent(tmp_path, embedder=None):
    from agent import Agent

    agent = Agent.__new__(Agent)  # avoid heavy __init__
    agent._effective_ws_dir = lambda: tmp_path  # type: ignore[method-assign]
    if embedder is not None:
        agent._semantic_embedding_service = embedder  # type: ignore[attr-defined]
    return agent


def _write_ledger(tmp_path, entries) -> None:
    (tmp_path / ".semantic_memory.json").write_text(
        json.dumps(entries), encoding="utf-8"
    )


def test_agent_block_query_ranks_and_prefixes_query(tmp_path) -> None:
    _write_ledger(tmp_path, _memories())
    agent = _bare_agent(tmp_path, FakeEmbedder())

    block = agent._semantic_memory_block("pytest failures in the suite")

    assert block.startswith(SEMANTIC_MEMORY_MARKER)
    assert "QUERY: pytest failures in the suite" in block
    body = block.split("QUERY:", 1)[1]
    assert body.index("user prefers pytest") < body.index("deploy uses blue-green")


def test_agent_block_query_falls_back_when_embeddings_down(tmp_path) -> None:
    """Embeddings unavailable => old unranked block, no QUERY line."""
    _write_ledger(tmp_path, _memories())
    agent = _bare_agent(tmp_path, ZeroEmbedder())

    block = agent._semantic_memory_block("pytest failures in the suite")

    assert block.startswith(SEMANTIC_MEMORY_MARKER)
    assert "QUERY:" not in block
    assert "user prefers pytest" in block
    assert "deploy uses blue-green" in block


def test_agent_block_without_query_is_unchanged(tmp_path) -> None:
    """No query => legacy unranked block even with embeddings available."""
    _write_ledger(tmp_path, _memories())
    agent = _bare_agent(tmp_path, FakeEmbedder())

    block = agent._semantic_memory_block()

    assert block.startswith(SEMANTIC_MEMORY_MARKER)
    assert "QUERY:" not in block
    assert "user prefers pytest" in block


def test_refresh_system_message_passes_query_to_block(tmp_path) -> None:
    """chat_nlp's pipeline step 1 must forward the user message as the query."""
    from agent import Agent

    _write_ledger(tmp_path, _memories())
    agent = _bare_agent(tmp_path, FakeEmbedder())
    agent._chat_history = []
    agent.is_plan_mode = lambda: False  # type: ignore[method-assign]
    agent._original_goal_block = lambda: ""  # type: ignore[method-assign]
    agent._decision_constraints_block = lambda: ""  # type: ignore[method-assign]
    agent._skill_index_block = lambda: ""  # type: ignore[method-assign]
    agent._habits_block = lambda: ""  # type: ignore[method-assign]

    agent._refresh_system_message("pytest failures in the suite")

    content = agent._chat_history[0]["content"]
    assert "QUERY: pytest failures in the suite" in content
    assert isinstance(agent, Agent)
