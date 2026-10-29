"""Regression tests for the ``kg`` REPL command (agent_core/commands/kg_cmd.py).

Pins the observable contract of the feature "add knowledge graph entries so
they are visible in the clear command":

* add writes into the canonical section and stamps a timestamp.
* every mutation calls agent._save_memory() exactly once; show never does.
* remove deletes an entry and prunes its section when it becomes empty.
* clear's stats counter (len of _knowledge_graph) reflects added sections.
* metadata: name/help text.
"""

import pytest

from agent_core.commands.kg_cmd import KGCommand, add_entry, remove_entry


class _StubAgent:
    """Minimal stand-in exposing exactly what KGCommand uses."""

    def __init__(self, graph=None) -> None:
        self._knowledge_graph = {} if graph is None else graph
        self.save_count = 0

    def _save_memory(self) -> None:
        self.save_count += 1

    def memory_stats(self) -> dict:
        return {"knowledge_graph": len(self._knowledge_graph)}


@pytest.mark.asyncio
async def test_add_fact_stores_value_and_timestamp(capsys) -> None:
    agent = _StubAgent()
    cmd = KGCommand()
    await cmd.execute(["add", "fact", "theme", "dark"], agent)

    entry = agent._knowledge_graph["facts"]["theme"]
    assert entry["value"] == "dark"
    assert "ts" in entry
    assert agent.save_count == 1, "add must persist exactly once"
    assert "[kg] added fact/theme" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_add_json_value_is_parsed(capsys) -> None:
    agent = _stub_with_graph()
    cmd = KGCommand()
    await cmd.execute(["add", "fact", "counts", "[1, 2]"], agent)
    assert agent._knowledge_graph["facts"]["counts"]["value"] == [1, 2]


def _stub_with_graph() -> _StubAgent:
    return _StubAgent()


@pytest.mark.asyncio
async def test_add_file_without_note_stores_empty_list(capsys) -> None:
    agent = _StubAgent()
    cmd = KGCommand()
    await cmd.execute(["add", "file", "agent.py"], agent)
    assert agent._knowledge_graph["files"]["agent.py"]["value"] == ""
    assert agent.save_count == 1


@pytest.mark.asyncio
async def test_add_unknown_section_errors_without_saving(capsys) -> None:
    agent = _StubAgent()
    cmd = KGCommand()
    await cmd.execute(["add", "bogus", "k", "v"], agent)
    out = capsys.readouterr().out
    assert "unknown section" in out
    assert agent.save_count == 0, "a rejected add must not persist"


@pytest.mark.asyncio
async def test_show_lists_sections_and_keys(capsys) -> None:
    agent = _StubAgent(
        {"facts": {"theme": {"value": "dark", "ts": "2025-01-01T00:00:00"}}}
    )
    cmd = KGCommand()
    await cmd.execute(["show"], agent)
    out = capsys.readouterr().out
    assert "Knowledge graph:" in out
    assert "facts (1):" in out
    assert "theme: dark [2025-01-01T00:00:00]" in out
    assert agent.save_count == 0, "show must not mutate/persist"


@pytest.mark.asyncio
async def test_show_single_key(capsys) -> None:
    agent = _StubAgent(
        {"symbols": {"run": {"value": "agent.py:10", "ts": "t"}}}
    )
    cmd = KGCommand()
    await cmd.execute(["show", "symbols", "run"], agent)
    out = capsys.readouterr().out
    assert "symbols/run:" in out
    assert "agent.py:10" in out


@pytest.mark.asyncio
async def test_remove_prunes_empty_section(capsys) -> None:
    agent = _StubAgent({"tags": {"d": {"value": "x", "ts": "t"}}})
    cmd = KGCommand()
    await cmd.execute(["remove", "tags", "d"], agent)
    assert "tags" not in agent._knowledge_graph, "empty section must be pruned"
    assert agent.save_count == 1

    # Removing a missing entry reports the error and does not persist.
    capsys.readouterr()
    await cmd.execute(["remove", "tags", "d"], agent)
    assert "no entry tags/d" in capsys.readouterr().out
    assert agent.save_count == 1


@pytest.mark.asyncio
async def test_add_then_clear_stats_sees_one_section(capsys) -> None:
    agent = _StubAgent()
    cmd = KGCommand()
    await cmd.execute(["add", "fact", "k", "v"], agent)
    assert agent.memory_stats()["knowledge_graph"] == 1


@pytest.mark.asyncio
async def test_no_args_prints_usage(capsys) -> None:
    agent = _StubAgent()
    cmd = KGCommand()
    await cmd.execute([], agent)
    assert "Usage: kg add" in capsys.readouterr().out


def test_add_entry_and_remove_entry_helpers() -> None:
    agent = _StubAgent()
    entry = add_entry(agent, "symbols", "run_interactive", {"file": "cli.py"})
    assert entry["value"] == {"file": "cli.py"}
    assert remove_entry(agent, "symbols", "run_interactive") is True
    assert remove_entry(agent, "symbols", "run_interactive") is False


def test_kg_command_metadata() -> None:
    cmd = KGCommand()
    assert cmd.name == "kg"
    assert "kg" in cmd.help_text
