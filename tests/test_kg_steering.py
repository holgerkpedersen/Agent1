"""Regression tests for `kg add` USE enforcement (prompts/context_blocks.py).

The knowledge-graph block has two jobs: show the model what is stored, and
make it use the command.  These tests pin the *use* half — the part that
makes sure `kg add fact` / `kg add file` are actually proposed to the model
when it reads files and works from prompts:

* a populated graph renders the marker once, names the `kg add` command and
  carries both steering lines (read a file -> `kg add file`, prompt teaches
  something -> `kg add fact`);
* an empty graph renders NOTHING, so a fresh workspace's system prompt stays
  byte-identical;
* a malformed entry must not kill the turn;
* rebuilds inject exactly one copy: ``strip_dynamic_system_blocks`` cuts at
  the marker exported by ``agent_core.run_utils`` (single source of truth).
"""

from agent_core.prompts.context_blocks import _knowledge_graph_block
from agent_core.run_utils import (
    KNOWLEDGE_GRAPH_MARKER,
    MAX_KG_BLOCK_LINES,
    strip_dynamic_system_blocks,
)


class _StubAgent:
    """Minimal stand-in exposing exactly what the block builder uses."""

    def __init__(self, graph) -> None:
        self._knowledge_graph = graph


def _populated() -> dict:
    return {
        "facts": {"theme": {"value": "dark", "ts": "2026-11-04T00:00:00"}},
        "files": {
            "agent_core/agent.py": {
                "value": "system-prompt assembly lives here",
                "ts": "2026-11-04T00:00:00",
            }
        },
    }


# ── content ───────────────────────────────────────────────────────────────


def test_empty_graph_renders_nothing() -> None:
    """A fresh workspace's prompt must stay byte-identical (no block)."""
    assert _knowledge_graph_block(_StubAgent({})) == ""


def test_non_dict_graph_is_ignored_not_raised() -> None:
    for bad in (None, "not a graph", ["list"]):
        assert _knowledge_graph_block(_StubAgent(bad)) == ""


def test_malformed_entry_does_not_kill_the_turn() -> None:
    """A broken entry degrades to the rest of the block, never raises."""
    graph = {"facts": {"good": {"value": "kept"}, "broken": object()}}
    block = _knowledge_graph_block(_StubAgent(graph))
    assert "facts/good: kept" in block


# ── use: the steering that makes the command get used ─────────────────────


def test_block_names_the_command_and_both_steering_rules() -> None:
    block = _knowledge_graph_block(_StubAgent(_populated()))
    assert block.startswith(KNOWLEDGE_GRAPH_MARKER)
    assert block.count("KNOWLEDGE GRAPH") == 1, "marker must appear exactly once"
    # The command is named with its full syntax, so the model can type it.
    assert "`kg add <fact|file|tag|symbol> <key> [value]`" in block
    # Reading a file -> record the file.
    assert "kg add file <path> <why it matters>" in block
    # A prompt teaching something durable -> record the fact.
    assert "kg add fact <key> <value>" in block
    # Timing: during work, not at the end.
    assert "as you work, not at the end" in block


def test_stored_entries_are_visible_to_the_model() -> None:
    block = _knowledge_graph_block(_StubAgent(_populated()))
    assert "facts/theme: dark" in block
    assert "files/agent_core/agent.py: system-prompt assembly lives here" in block


def test_overflow_is_capped_and_points_at_kg_show() -> None:
    graph = {
        "facts": {f"k{i}": {"value": f"v{i}"} for i in range(MAX_KG_BLOCK_LINES + 7)}
    }
    block = _knowledge_graph_block(_StubAgent(graph))
    assert "run 'kg show'" in block
    assert "facts/k0: v0" in block
    assert f"facts/k{MAX_KG_BLOCK_LINES + 6}" not in block


# ── single copy per rebuild ───────────────────────────────────────────────


def test_strip_removes_the_block_so_rebuilds_keep_one_copy() -> None:
    """Every turn rebuilds the prompt; the strip must cut the previous block."""
    system = "BASE SYSTEM PROMPT" + _knowledge_graph_block(_StubAgent(_populated()))
    stripped = strip_dynamic_system_blocks(system)
    assert stripped.strip() == "BASE SYSTEM PROMPT"
    # Rebuild: strip + re-append => exactly one copy, no accumulation.
    rebuilt = stripped + _knowledge_graph_block(_StubAgent(_populated()))
    assert rebuilt.count("KNOWLEDGE GRAPH") == 1


def test_marker_is_the_single_source_of_truth() -> None:
    """The builder prefixes the marker exported by run_utils, not a copy."""
    block = _knowledge_graph_block(_StubAgent(_populated()))
    assert KNOWLEDGE_GRAPH_MARKER in block
