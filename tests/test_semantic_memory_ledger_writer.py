"""The ledger must be WRITTEN every turn, and the prompt block must rank it.

Before this suite the ``SEMANTIC MEMORY`` prompt block had a reader
(:func:`agent_core.memory.load_semantic_memory`), a ranker
(:func:`agent_core.memory.scoring.rank_memories`) and a renderer
(:func:`agent_core.memory.semantic_memory_block`) but NO writer — the ledger
was therefore permanently empty and the whole feature was dead weight.  These
tests pin the three links that were missing:

1. ``Agent._finish_turn`` distils the finished turn into the ledger (real
   ``_finish_turn`` code path, fake agent only for the surrounding hooks).
2. ``_semantic_memory_block(query)`` renders entries scored and WEIGHTED
   against the query/goal (relevance, goal alignment, recency, importance)
   instead of dumping the first five ledger lines.
3. Embeddings are memoised PER ENTRY, so a ledger that grows one line per
   turn costs one new embedding — not a full LM Studio re-embed of the whole
   ledger on every turn (the VRAM/latency requirement behind this feature).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np

from agent_core.memory import (
    EmbeddingService,
    SEMANTIC_MEMORY_MARKER,
    load_semantic_memory,
)
from agent_core.memory.distill import record_turn
from agent_core.memory.memory_store import (
    clear_semantic_index_cache,
    semantic_search_memories,
)
from agent_core.memory.scoring import rank_memories
from agent_core.memory.types import save_semantic_memory


class CountingEmbedder(EmbeddingService):
    """Deterministic embedder that COUNTS how many texts it was asked to embed."""

    VOCAB = ("pytest", "deploy", "database", "cache", "ranking")

    def __init__(self) -> None:
        super().__init__()
        self.texts_embedded: list[str] = []

    def embed_text(self, texts):  # type: ignore[override]
        self.texts_embedded.extend(str(t) for t in texts)
        out = np.zeros((len(texts), 384))
        for i, text in enumerate(texts):
            for word in str(text).lower().split():
                axis = 383
                for j, key in enumerate(self.VOCAB):
                    if key in word:
                        axis = j
                        break
                out[i, axis] += 1.0
        return out


def setup_function() -> None:
    clear_semantic_index_cache()


# ---------------------------------------------------------------------------
# 1. The writer: _finish_turn -> record_turn -> ledger
# ---------------------------------------------------------------------------


class _FakeLoop:
    """Minimal ``ToolLoopRunner`` surface read by ``_finish_turn``."""

    tool_calls_made = 1
    termination_reason = "answer"
    final_answer_streamed = True


class _FakeAgent:
    """Just enough ``Agent`` for the real ``_finish_turn`` to run end to end."""

    def __init__(self, ws) -> None:
        self._ws = ws
        self.workspace = str(ws)
        self._chat_history: list[dict] = []
        self._turn_start_index = 0
        self._original_goal = "make the memory ledger rank against the prompt"
        self.session_id = "test-session"
        self._last_user_input = (
            "from now on always keep the ranking weighted, we prefer pytest"
        )
        self._turn_started_at = datetime.now()
        self._turn_run_id = None
        self.llm = None
        self._nlp_workspace = None

    # --- hooks _finish_turn expects --------------------------------
    def _effective_ws_dir(self):
        return self._ws

    def _chat_messages_since_turn_start(self):
        return list(self._chat_history)

    def _mutating_files_this_turn(self):
        return ["agent_core/memory/scoring.py"]

    def _record_llm_experience(self, **_kwargs):
        return None

    def _collect_autonomous_decisions_from_turn(self, _messages):
        return None

    def _snapshot_session_meta(self, **_kwargs):
        return None

    def _save_chat_history(self):
        return None

    def _save_memory(self):
        return None

    def is_plan_mode(self):
        return False

    def _semantic_memory_block(self, query=None):
        """Delegate to the production renderer (real code path)."""
        from agent_core.prompts.context_blocks import _semantic_memory_block

        return _semantic_memory_block(self, query)


def _run_finish_turn(agent, store: dict | None = None) -> None:
    """Call the REAL ``_finish_turn`` (it prints; output is irrelevant here)."""
    from agent_core.config import AgentDisplayMode
    from agent_core.turn_loop import _finish_turn

    _finish_turn(agent, "", None, _FakeLoop(), AgentDisplayMode.QUIET)


def test_finish_turn_writes_the_semantic_memory_ledger(tmp_path) -> None:
    """The dead feature's core bug: nothing ever wrote the ledger."""
    agent = _FakeAgent(tmp_path)
    agent._chat_history = [
        {
            "role": "user",
            "content": "from now on always keep the ranking weighted, we prefer pytest",
        },
        {"role": "assistant", "content": "Done: ranking is weighted."},
        {
            "role": "tool",
            "content": (
                "Edited agent_core/memory/scoring.py\n[verify] py_compile ✓"
            ),
        },
    ]

    assert load_semantic_memory(tmp_path) == []

    _run_finish_turn(agent)

    ledger = load_semantic_memory(tmp_path)
    assert ledger, "_finish_turn did not write the semantic memory ledger"
    texts = " ".join(str(e["metadata"]["text"]) for e in ledger).lower()
    assert "scoring.py" in texts or "ranking" in texts


def test_finish_turn_keeps_ledger_when_turn_has_nothing_to_say(tmp_path) -> None:
    """An empty turn must not corrupt or truncate an existing ledger."""
    save_semantic_memory(tmp_path, ["Database migrations run nightly."])
    agent = _FakeAgent(tmp_path)
    agent._chat_history = [{"role": "assistant", "content": "ok"}]

    _run_finish_turn(agent)

    ledger = load_semantic_memory(tmp_path)
    assert any("migrations" in e["metadata"]["text"] for e in ledger)


def test_record_turn_deduplicates_and_counts_reuse(tmp_path) -> None:
    """Same fact in two turns => one entry with use_count bumped."""
    messages = [
        {"role": "user", "content": "we always deploy with blue-green"},
    ]
    first = record_turn(tmp_path, messages)
    assert first >= 1
    ledger = load_semantic_memory(tmp_path)
    assert ledger

    record_turn(tmp_path, messages)
    again = load_semantic_memory(tmp_path)
    preference_entries = [
        e for e in again if e["metadata"].get("kind") == "preference"
    ]
    assert len(preference_entries) == 1
    assert int(preference_entries[0]["metadata"].get("use_count") or 0) >= 2


# ---------------------------------------------------------------------------
# 2. The ranker: weighted against query and goal, bounded and stable
# ---------------------------------------------------------------------------


def _weighted_ledger(now: datetime) -> list[dict]:
    return [
        {
            "metadata": {
                "text": "Unrelated note about lunch.",
                "kind": "note",
                "created_at": (now - timedelta(days=400)).isoformat(),
            }
        },
        {
            "metadata": {
                "text": "Preference: always weight pytest ranking against prompt.",
                "kind": "preference",
                "created_at": (now - timedelta(days=2)).isoformat(),
            }
        },
        {
            "metadata": {
                "text": "Touched agent_core/memory/scoring.py",
                "kind": "file",
                "created_at": now.isoformat(),
            }
        },
    ]


def test_rank_memories_orders_by_weighted_relevance() -> None:
    now = datetime.now()
    ranked = rank_memories(
        _weighted_ledger(now),
        "weight pytest ranking against prompt",
        "make the memory ledger rank against the prompt",
        now=now,
    )
    texts = [item["metadata"]["text"] for item in ranked]
    assert "preference" in texts[0].lower()
    assert "lunch" in texts[-1].lower()


def test_rank_memories_semantic_scores_change_the_order() -> None:
    """An injected embedding score must actually move the ranking."""
    now = datetime.now()
    base = rank_memories(_weighted_ledger(now), None, None, now=now)
    assert "lunch" in base[-1]["metadata"]["text"].lower()

    boosted = rank_memories(
        _weighted_ledger(now),
        None,
        None,
        semantic_scores={"Unrelated note about lunch.": 1.0},
        now=now,
    )
    assert "lunch" in boosted[0]["metadata"]["text"].lower()


def test_rank_memories_is_bounded_and_keeps_metadata() -> None:
    now = datetime.now()
    ranked = rank_memories(_weighted_ledger(now), None, None, now=now, k=2)
    assert len(ranked) == 2
    for item in ranked:
        assert 0.0 <= float(item["similarity_score"]) <= 1.0
        assert item["metadata"].get("text")


# ---------------------------------------------------------------------------
# 3. Per-entry embedding cache (VRAM / latency requirement)
# ---------------------------------------------------------------------------


def test_ledger_growth_embeds_only_the_new_entry() -> None:
    """One appended entry => ONE new embedding, not a full re-embed."""
    embedder = CountingEmbedder()
    ledger = [
        {"metadata": {"text": "user prefers pytest"}},
        {"metadata": {"text": "deploy uses blue-green"}},
    ]

    semantic_search_memories(ledger, "pytest", embedding_service=embedder)
    first_pass = len(embedder.texts_embedded)
    # 2 entries + 1 query embedding
    assert first_pass == 3

    # Same ledger again: only the QUERY is embedded, no entry is re-embedded.
    # (The query legitimately changes every turn, so it is never cached.)
    semantic_search_memories(ledger, "pytest", embedding_service=embedder)
    assert len(embedder.texts_embedded) == first_pass + 1

    # Ledger grew by one entry: exactly one new entry embedding (+ the query).
    grown = ledger + [{"metadata": {"text": "database cache ranking"}}]
    semantic_search_memories(grown, "pytest", embedding_service=embedder)
    assert len(embedder.texts_embedded) == first_pass + 3

    # ... and the two unchanged entries were NOT embedded a second time.
    per_text = {
        t: embedder.texts_embedded.count(t) for t in set(embedder.texts_embedded)
    }
    assert per_text["user prefers pytest"] == 1
    assert per_text["deploy uses blue-green"] == 1


def test_turn_by_turn_growth_stays_linear_not_quadratic(tmp_path) -> None:
    """20 turns of ledger growth must not cost 20 full-ledger re-embeds."""
    embedder = CountingEmbedder()
    for i in range(20):
        save_semantic_memory(
            tmp_path,
            [{"metadata": {"text": f"fact number {i} about pytest"}}
             for i in range(i + 1)],
        )
        ledger = load_semantic_memory(tmp_path)
        semantic_search_memories(ledger, "pytest", embedding_service=embedder)

    # Quadratic behaviour would embed sum(1..20) = 210 entry texts; the
    # per-entry cache embeds each distinct entry exactly once.
    distinct_entries = 20
    query_calls = 20
    assert len(embedder.texts_embedded) <= distinct_entries + query_calls + 1


def test_agent_block_ranks_the_written_ledger(tmp_path) -> None:
    """End to end: a turn is recorded, then the block surfaces it by relevance."""
    messages = [
        {"role": "user", "content": "we always deploy with blue-green"},
    ]
    record_turn(tmp_path, messages, ["agent_core/memory/scoring.py"])

    agent = _FakeAgent(tmp_path)
    agent._semantic_embedding_service = CountingEmbedder()

    block = agent._semantic_memory_block("how do we deploy?")
    assert block.startswith(SEMANTIC_MEMORY_MARKER)
    assert "blue-green" in block


def test_corrupt_ledger_does_not_break_the_block(tmp_path) -> None:
    (tmp_path / ".semantic_memory.json").write_text("{not json", encoding="utf-8")
    agent = _FakeAgent(tmp_path)
    agent._semantic_embedding_service = CountingEmbedder()

    assert agent._semantic_memory_block("pytest") == ""


def test_ledger_entries_survive_a_json_round_trip(tmp_path) -> None:
    """Metadata written by the distiller must come back with its ranking fields."""
    record_turn(
        tmp_path,
        [{"role": "user", "content": "always prefer pytest for the suite"}],
    )
    raw = json.loads((tmp_path / ".semantic_memory.json").read_text(encoding="utf-8"))
    assert raw and "metadata" in raw[0]
    ledger = load_semantic_memory(tmp_path)
    meta = ledger[0]["metadata"]
    assert meta.get("text")
    assert meta.get("kind") in {"preference", "file", "error", "decision", "note"}
