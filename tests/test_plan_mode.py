"""Regression tests for build/plan session modes (agent_core/modes.py).

Covers:
- filter_tool_schemas: build keeps everything; plan exposes only the
  read-only subset
- check_tool_allowed: mutating tools rejected in plan mode (case-insensitive),
  read-only tools allowed, build never rejects
- Agent integration: default mode is build, set_mode validates unknown tags,
  _execute_tool_call rejects a mutating call in plan mode and still executes
  read-only calls
- ModeCommand: show/switch wiring through the registry-facing interface
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent import Agent
from agent_core.commands.mode_cmd import ModeCommand
from agent_core.modes import (
    MODE_BUILD,
    MODE_PLAN,
    PLAN_MODE_TOOLS,
    check_tool_allowed,
    filter_tool_schemas,
    is_plan_mode,
    plan_mode_system_suffix,
    plan_mode_turn_note,
)
from agent_core.tool_schemas import NLP_TOOL_NAMES


@pytest.fixture()
def agent() -> Agent:
    return Agent(workspace=".")


# ---------------------------------------------------------------------------
# Schema filtering (what the LLM is offered)
# ---------------------------------------------------------------------------

def test_build_mode_offers_every_schema(agent: Agent) -> None:
    filtered = filter_tool_schemas(
        [{"function": {"name": n}} for n in NLP_TOOL_NAMES], MODE_BUILD
    )
    assert {s["function"]["name"] for s in filtered} == set(NLP_TOOL_NAMES)


def test_plan_mode_filters_to_read_only_subset() -> None:
    filtered = filter_tool_schemas(
        [{"function": {"name": n}} for n in NLP_TOOL_NAMES], MODE_PLAN
    )
    assert {s["function"]["name"] for s in filtered} == set(PLAN_MODE_TOOLS)
    assert set(PLAN_MODE_TOOLS) <= set(NLP_TOOL_NAMES)


def test_unknown_mode_fails_safe_to_full_toolset() -> None:
    schemas = [{"function": {"name": "write"}}]
    assert filter_tool_schemas(schemas, "gibberish") == schemas


# ---------------------------------------------------------------------------
# Executor-level rejection (the choke point)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool", ["write", "edit", "run", "git", "tests", "fix", "analyze"])
def test_plan_mode_blocks_mutating_tools(tool: str) -> None:
    rejection = check_tool_allowed(tool, MODE_PLAN)
    assert rejection is not None
    assert "plan mode" in rejection.lower()
    assert tool in rejection


@pytest.mark.parametrize("tool", sorted(PLAN_MODE_TOOLS))
def test_plan_mode_allows_read_only_tools(tool: str) -> None:
    assert check_tool_allowed(tool, MODE_PLAN) is None


def test_rejection_is_case_insensitive() -> None:
    assert check_tool_allowed("WRITE", MODE_PLAN) is not None


def test_build_mode_never_rejects() -> None:
    for tool in NLP_TOOL_NAMES:
        assert check_tool_allowed(tool, MODE_BUILD) is None


# ---------------------------------------------------------------------------
# Agent session state + end-to-end executor behaviour
# ---------------------------------------------------------------------------

def test_agent_defaults_to_build_mode(agent: Agent) -> None:
    assert agent.mode == MODE_BUILD
    assert not agent.is_plan_mode()


def test_set_mode_round_trip_and_validation(agent: Agent) -> None:
    agent.set_mode(MODE_PLAN)
    assert agent.is_plan_mode()
    agent.set_mode(MODE_BUILD)
    assert not agent.is_plan_mode()
    with pytest.raises(ValueError):
        agent.set_mode("stealth")


def test_plan_mode_executor_rejects_write_end_to_end(
    agent: Agent, tmp_path: Path
) -> None:
    target = tmp_path / "must_not_exist.txt"
    agent.set_mode(MODE_PLAN)
    result = asyncio.run(agent._execute_tool_call(
        "write", {"path": str(target), "content": "nope"}
    ))
    assert result.startswith("[plan mode]")
    assert not target.exists()


def test_plan_mode_executor_still_executes_reads(
    agent: Agent, tmp_path: Path
) -> None:
    probe = tmp_path / "probe.txt"
    probe.write_text("readable", encoding="utf-8")
    agent.set_mode(MODE_PLAN)
    result = asyncio.run(agent._execute_tool_call(
        "read", {"path": str(probe), "offset": 1, "limit": 10}
    ))
    assert result == "readable"


def test_build_mode_executor_writes_normally(
    agent: Agent, tmp_path: Path
) -> None:
    target = tmp_path / "allowed.txt"
    result = asyncio.run(agent._execute_tool_call(
        "write", {"path": str(target), "content": "ok"}
    ))
    assert result.startswith("Written")
    assert target.read_text(encoding="utf-8") == "ok"


# ---------------------------------------------------------------------------
# Steering text + command surface
# ---------------------------------------------------------------------------

def test_plan_mode_texts_mention_read_only_and_escape_hatch() -> None:
    suffix = plan_mode_system_suffix()
    # The escape hatch lives in the session-start system suffix.
    assert "mode build" in suffix
    assert "REJECTED" in suffix
    # The per-turn note stays short but must state the read-only contract.
    turn = plan_mode_turn_note()
    assert "change" in turn.lower() and "nothing" in turn.lower()
    assert "blocked" in turn.lower()


def test_is_plan_mode_is_exact_match() -> None:
    assert is_plan_mode(MODE_PLAN)
    assert not is_plan_mode(MODE_BUILD)
    assert not is_plan_mode("")  # anything else fails safe to build behaviour


def test_mode_command_show_and_switch(agent: Agent, capsys) -> None:
    asyncio.run(ModeCommand().execute(["show"], agent))
    assert "build" in capsys.readouterr().out

    asyncio.run(ModeCommand().execute(["plan"], agent))
    assert agent.is_plan_mode()

    asyncio.run(ModeCommand().execute(["build"], agent))
    assert not agent.is_plan_mode()


def test_mode_command_rejects_unknown_tag(agent: Agent) -> None:
    # Must not raise and must not change the mode.
    asyncio.run(ModeCommand().execute(["stealth"], agent))
    assert agent.mode == MODE_BUILD
