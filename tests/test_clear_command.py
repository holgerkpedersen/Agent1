"""Regression tests for the REPL ``clear`` command (agent_core/commands/clear_cmd.py).

Issue ``iss-general-kanban_f01a7cb8fb514e7d8e48f8d224b66d24`` ("clear command")
is open, so these tests pin the observable behaviour of the command:

* ``clear stats`` prints the memory summary and never mutates state.
* ``clear --force`` clears without prompting.
* ``clear`` without ``--force`` prompts and only clears on ``y``/``yes``.
* An empty agent reports "Nothing to clear." and does not call clear_history.
* The knowledge graph line counts ENTRIES (not sections) and an empty graph
  prints a hint, so the counter is never a silent zero (issue ask: "why is
  the knowledge graph line always showing 0 entries").
"""

import pytest
from unittest.mock import patch

from agent_core.commands.clear_cmd import ClearCommand


class _StubAgent:
    """Minimal stand-in exposing exactly what ClearCommand uses."""

    def __init__(self, stats: dict) -> None:
        self._stats = stats
        self.clear_catches = 0

    def memory_stats(self) -> dict:
        return dict(self._stats)

    def clear_history(self) -> None:
        self.clear_catches += 1


def _empty_stats() -> dict:
    return {
        "chat_history": 0,
        "files_read": 0,
        "stale_files": 0,
        "working_memory": 0,
        "semantic_index": 0,
        "knowledge_graph": 0,
        "knowledge_graph_entries": 0,
    }


def _non_empty_stats() -> dict:
    return {
        "chat_history": 3,
        "files_read": 2,
        "stale_files": 1,
        "working_memory": 3,
        "semantic_index": 6,
        "knowledge_graph": 0,
        "knowledge_graph_entries": 0,
    }


def _kg_stats() -> dict:
    return {
        "chat_history": 0,
        "files_read": 0,
        "stale_files": 0,
        "working_memory": 0,
        "semantic_index": 0,
        "knowledge_graph": 2,
        "knowledge_graph_entries": 5,
    }


@pytest.mark.asyncio
async def test_clear_stats_counts_knowledge_graph_entries(capsys) -> None:
    """The KG line reports entries, the unit the user asked about."""
    agent = _StubAgent(_kg_stats())
    cmd = ClearCommand()

    with patch("builtins.input", side_effect=AssertionError("prompted")):
        await cmd.execute(["stats"], agent)

    out = capsys.readouterr().out
    assert "knowledge graph:" in out
    assert "5 entries" in out
    assert "2 sections" in out
    assert "kg add" not in out, "hint must only show when the graph is empty"
    assert agent.clear_catches == 0


@pytest.mark.asyncio
async def test_clear_stats_hint_when_knowledge_graph_empty(capsys) -> None:
    """An empty graph explains itself instead of showing a bare zero."""
    agent = _StubAgent(_empty_stats())
    cmd = ClearCommand()

    with patch("builtins.input", side_effect=AssertionError("prompted")):
        await cmd.execute(["stats"], agent)

    out = capsys.readouterr().out
    assert "knowledge graph:" in out
    assert "0 entries" in out
    assert "kg add fact" in out
    assert "seed_kg.py" in out


@pytest.mark.asyncio
async def test_clear_with_only_knowledge_graph_entries_prompts(capsys) -> None:
    """KG entries alone count as state: the user must be prompted, not told
    'Nothing to clear.', and confirming clears them."""
    agent = _StubAgent(_kg_stats())
    cmd = ClearCommand()

    with patch("builtins.input", return_value="y"):
        await cmd.execute([], agent)

    out = capsys.readouterr().out
    assert "Nothing to clear." not in out
    assert agent.clear_catches == 1


@pytest.mark.asyncio
async def test_clear_stats_survives_missing_entries_key(capsys) -> None:
    """Old stubs/custom agents without the new key must not crash clear."""
    stats = _non_empty_stats()
    stats.pop("knowledge_graph_entries")
    agent = _StubAgent(stats)
    cmd = ClearCommand()

    with patch("builtins.input", side_effect=AssertionError("prompted")):
        await cmd.execute(["stats"], agent)

    assert "knowledge graph:" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_clear_stats_does_not_mutate_state(capsys) -> None:
    agent = _StubAgent(_non_empty_stats())
    cmd = ClearCommand()
    await cmd.execute(["stats"], agent)

    out = capsys.readouterr().out
    assert "Memory state:" in out
    assert "chat history:       3 messages" in out
    assert "stale files:        1 (changed externally)" in out
    assert "knowledge graph:" in out
    assert "0 sections" in out
    assert agent.clear_catches == 0, "stats must not clear memory"


@pytest.mark.asyncio
async def test_clear_force_skips_prompt(capsys) -> None:
    agent = _StubAgent(_non_empty_stats())
    cmd = ClearCommand()

    # Any input() call would raise here, proving --force never prompts.
    with patch("builtins.input", side_effect=AssertionError("prompted")):
        await cmd.execute(["--force"], agent)

    out = capsys.readouterr().out
    assert "Agent memory cleared." in out
    assert agent.clear_catches == 1


@pytest.mark.asyncio
async def test_clear_requires_confirmation(capsys) -> None:
    agent = _StubAgent(_non_empty_stats())
    cmd = ClearCommand()

    with patch("builtins.input", return_value="n"):
        await cmd.execute([], agent)

    out = capsys.readouterr().out
    assert "Cancelled." in out
    assert agent.clear_catches == 0, "declining confirmation must not clear"

    with patch("builtins.input", return_value="yes"):
        await cmd.execute([], agent)

    out = capsys.readouterr().out
    assert "Agent memory cleared." in out
    assert agent.clear_catches == 1


@pytest.mark.asyncio
async def test_clear_on_empty_agent_reports_nothing_to_clear(capsys) -> None:
    agent = _StubAgent(_empty_stats())
    cmd = ClearCommand()

    with patch("builtins.input", side_effect=AssertionError("prompted")):
        await cmd.execute([], agent)

    out = capsys.readouterr().out
    assert "Nothing to clear." in out
    assert agent.clear_catches == 0


@pytest.mark.asyncio
async def test_clear_command_metadata() -> None:
    cmd = ClearCommand()
    assert cmd.name == "clear"
    assert "clear" in cmd.help_text
