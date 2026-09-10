"""Tests for create_subagent and run_subagent_task NLP tools."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

import agent
from agent import Agent


class _FakeLLMClient:
    """Stands in for agent.LLMClient so tests never reach the LM Studio API."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name: str = model_name or "test-model"
        self.seen_messages: list[dict[str, str]] = []

    async def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        self.seen_messages.extend(messages)
        if messages:
            last = messages[-1]["content"]
            return f"Response to: {last[:50]}"
        return "empty-response"


def _make_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Agent:
    """Create an Agent with a fake LLM client for testing."""
    monkeypatch.setattr(agent, "LLMClient", _FakeLLMClient)
    return Agent(workspace=str(tmp_path), model_name="test-model")


def test_create_subagent_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_nlp_create_subagent creates a persistent subagent."""
    a = _make_agent(tmp_path, monkeypatch)
    result = asyncio.run(
        a._nlp_create_subagent({"name": "test1", "role": "planner"})
    )
    assert "Created subagent" in result
    assert "test1" in result
    assert "planner" in result
    assert hasattr(a, "_subagents")
    assert "test1" in a._subagents


def test_create_subagent_with_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_nlp_create_subagent respects the workspace argument."""
    a = _make_agent(tmp_path, monkeypatch)
    ws = str(tmp_path / "sub")
    result = asyncio.run(
        a._nlp_create_subagent(
            {"name": "ws1", "role": "tester", "workspace": ws}
        )
    )
    assert "Created subagent" in result
    sub = a._subagents["ws1"]
    assert sub.workspace == ws


def test_create_subagent_duplicate_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_nlp_create_subagent rejects duplicate names."""
    a = _make_agent(tmp_path, monkeypatch)
    asyncio.run(a._nlp_create_subagent({"name": "dup", "role": "planner"}))
    result = asyncio.run(
        a._nlp_create_subagent({"name": "dup", "role": "planner"})
    )
    assert "already exists" in result


def test_create_subagent_unknown_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_nlp_create_subagent rejects unknown roles."""
    a = _make_agent(tmp_path, monkeypatch)
    result = asyncio.run(
        a._nlp_create_subagent({"name": "x", "role": "nonexistent"})
    )
    assert "unknown role" in result


def test_create_subagent_missing_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_nlp_create_subagent requires both name and role."""
    a = _make_agent(tmp_path, monkeypatch)
    result = asyncio.run(a._nlp_create_subagent({}))
    assert "Error" in result
    result = asyncio.run(a._nlp_create_subagent({"name": "x"}))
    assert "Error" in result
    result = asyncio.run(a._nlp_create_subagent({"role": "planner"}))
    assert "Error" in result


def test_run_subagent_task_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_nlp_run_subagent_task errors on nonexistent subagent."""
    a = _make_agent(tmp_path, monkeypatch)
    result = asyncio.run(
        a._nlp_run_subagent_task({"name": "ghost", "task": "do something"})
    )
    assert "not found" in result


def test_run_subagent_task_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_nlp_run_subagent_task sends a task and returns the result."""
    a = _make_agent(tmp_path, monkeypatch)
    asyncio.run(a._nlp_create_subagent({"name": "w1", "role": "planner"}))
    result = asyncio.run(
        a._nlp_run_subagent_task({"name": "w1", "task": "What files exist?"})
    )
    assert len(result) > 0
    assert "Response to:" in result


def test_run_subagent_task_missing_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_nlp_run_subagent_task requires both name and task."""
    a = _make_agent(tmp_path, monkeypatch)
    result = asyncio.run(a._nlp_run_subagent_task({}))
    assert "Error" in result
    result = asyncio.run(a._nlp_run_subagent_task({"name": "x"}))
    assert "Error" in result
    result = asyncio.run(a._nlp_run_subagent_task({"task": "do stuff"}))
    assert "Error" in result


def test_tools_registered_in_dispatch_table() -> None:
    """New tools appear in the dispatch table."""
    agent_instance = Agent.__new__(Agent)
    agent_instance.workspace = "."
    agent_instance._nlp_workspace = None
    handlers = agent_instance._nlp_tool_handlers()
    assert "create_subagent" in handlers
    assert "run_subagent_task" in handlers


def test_tools_in_schemas() -> None:
    """New tools have schemas in NLP_TOOL_SCHEMAS."""
    from agent_core.tool_schemas import NLP_TOOL_SCHEMAS, NLP_TOOL_NAMES
    assert "create_subagent" in NLP_TOOL_NAMES
    assert "run_subagent_task" in NLP_TOOL_NAMES
    schemas_by_name = {s["function"]["name"]: s for s in NLP_TOOL_SCHEMAS}
    cs = schemas_by_name["create_subagent"]
    assert "name" in cs["function"]["parameters"]["properties"]
    assert "role" in cs["function"]["parameters"]["properties"]
    rs = schemas_by_name["run_subagent_task"]
    assert "name" in rs["function"]["parameters"]["properties"]
    assert "task" in rs["function"]["parameters"]["properties"]
