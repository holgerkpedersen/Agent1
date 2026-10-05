"""Regression tests for the semantic-memory prompt block.

Guards a real outage: an uncommitted change added
``Agent._semantic_memory_block`` referencing ``load_semantic_memory`` /
``semantic_memory_block`` / ``SEMANTIC_MEMORY_MARKER`` that were defined
nowhere, and left ``agent_core/memory/types.py`` using ``Sequence``/``Mapping``
/``os`` without importing them.  The result was:

* ``import agent_core.memory`` raised ``NameError`` (whole package dead), and
* every chat turn logged "Semantic memory block unavailable" because the
  ``NameError`` was swallowed by a bare ``except Exception``.

These tests pin the public surface so the silent failure cannot come back.
"""
from __future__ import annotations

import json

from agent_core.memory import (
    SEMANTIC_MEMORY_MARKER,
    load_semantic_memory,
    semantic_memory_block,
)


def test_memory_package_imports_and_exports_public_surface() -> None:
    """The names agent.py imports must exist (they were previously NameError)."""
    assert SEMANTIC_MEMORY_MARKER
    assert callable(semantic_memory_block)
    assert callable(load_semantic_memory)


def test_block_is_empty_without_memories() -> None:
    """Empty input keeps the prompt byte-identical to the pre-block baseline."""
    assert semantic_memory_block(None) == ""
    assert semantic_memory_block([]) == ""
    assert semantic_memory_block("") == ""


def test_block_renders_marker_and_entries() -> None:
    block = semantic_memory_block(
        [{"metadata": {"text": "user prefers pytest"}, "similarity_score": 0.91}]
    )
    assert block.startswith(SEMANTIC_MEMORY_MARKER)
    assert "user prefers pytest" in block
    assert "0.91" in block


def test_block_accepts_plain_strings() -> None:
    block = semantic_memory_block(["alpha", "beta"])
    assert block.startswith(SEMANTIC_MEMORY_MARKER)
    assert "alpha" in block and "beta" in block


def test_block_respects_line_cap() -> None:
    from agent_core.memory.types import MAX_BLOCK_LINES

    block = semantic_memory_block([f"m{i}" for i in range(50)])
    # marker + at most MAX_BLOCK_LINES entry lines
    assert len(block.strip().splitlines()) <= MAX_BLOCK_LINES + 1


def test_load_missing_ledger_returns_empty(tmp_path) -> None:
    assert load_semantic_memory(tmp_path) == []


def test_load_round_trips_ledger(tmp_path) -> None:
    ledger = tmp_path / ".semantic_memory.json"
    ledger.write_text(
        json.dumps(
            [
                {"metadata": {"text": "fact one"}, "similarity_score": 0.5},
                "plain note",
                {"metadata": {"text": "   "}},  # blank -> dropped
            ]
        ),
        encoding="utf-8",
    )
    memories = load_semantic_memory(tmp_path)
    assert [m["metadata"]["text"] for m in memories] == ["fact one", "plain note"]
    assert memories[0]["similarity_score"] == 0.5


def test_corrupt_ledger_is_quarantined_and_never_raises(tmp_path) -> None:
    ledger = tmp_path / ".semantic_memory.json"
    ledger.write_text("{not json", encoding="utf-8")

    assert load_semantic_memory(tmp_path) == []  # never raises

    quarantined = list(tmp_path.glob(".semantic_memory.json.bad-*"))
    assert quarantined, "corrupt ledger should be quarantined for inspection"
    assert not ledger.exists()


def test_agent_semantic_memory_block_uses_real_ledger(tmp_path) -> None:
    """Exercise the actual Agent method, not a copy of its logic.

    This is the path that silently returned "" on every turn before the fix.
    """
    from agent import Agent

    (tmp_path / ".semantic_memory.json").write_text(
        json.dumps(
            [{"metadata": {"text": "deploy uses blue-green"}, "similarity_score": 0.8}]
        ),
        encoding="utf-8",
    )

    agent = Agent.__new__(Agent)  # avoid heavy __init__
    agent._effective_ws_dir = lambda: tmp_path  # type: ignore[method-assign]

    block = agent._semantic_memory_block()
    assert block.startswith(SEMANTIC_MEMORY_MARKER)
    assert "deploy uses blue-green" in block


def test_agent_semantic_memory_block_never_raises_on_broken_ledger(tmp_path) -> None:
    from agent import Agent

    (tmp_path / ".semantic_memory.json").write_text("<<<broken", encoding="utf-8")

    agent = Agent.__new__(Agent)
    agent._effective_ws_dir = lambda: tmp_path  # type: ignore[method-assign]

    assert agent._semantic_memory_block() == ""


def test_strip_dynamic_system_blocks_removes_semantic_block() -> None:
    """The marker must be registered so a stale block is stripped each turn."""
    from agent import _strip_dynamic_system_blocks

    text = "BASE PROMPT" + SEMANTIC_MEMORY_MARKER + "\n  - stale entry\n"
    assert _strip_dynamic_system_blocks(text) == "BASE PROMPT"
