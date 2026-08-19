"""Core tests for the agent entry point: LLMClient wiring, configuration
error handling, run-tool timeout validation, and workspace paths."""

import asyncio
import os
from pathlib import Path
from typing import Any, NoReturn, cast

import pytest

import agent
from agent import Agent


class _FakeLLMClient:
    """Stands in for agent.LLMClient so tests never reach the LM Studio API."""

    def __init__(self, model_name: str | None = None, api_key: str | None = None) -> None:
        self.model_name: str = model_name or "test-model"
        self.seen_messages: list[dict[str, str]] | None = None

    async def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        self.seen_messages = list(messages)
        return "fake-answer"


def _make_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Agent:
    monkeypatch.setattr(agent, "LLMClient", _FakeLLMClient)
    return Agent(workspace=str(tmp_path), model_name="test-model")


def test_agent_entrypoint_returns_llm_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a = _make_agent(tmp_path, monkeypatch)
    answer = asyncio.run(a.llm.chat([{"role": "user", "content": "hello"}]))
    assert answer == "fake-answer"
    assert a.llm.model_name == "test-model"


def test_agent_complete_accepts_messages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a = _make_agent(tmp_path, monkeypatch)
    messages = [{"role": "user", "content": "user"}]
    asyncio.run(a.llm.chat(messages))
    assert cast(_FakeLLMClient, a.llm).seen_messages == messages


def test_agent_configuration_error_is_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(settings: Any, model_name: str) -> NoReturn:
        raise RuntimeError("provider configuration failed")

    monkeypatch.setattr("agent_core.llm.provider.build_provider", _boom)
    with pytest.raises(RuntimeError) as exc_info:
        agent.LLMClient(model_name="test-model")
    assert isinstance(exc_info.value, Exception)


def test_agent_workspace_is_path_like(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a = _make_agent(tmp_path, monkeypatch)
    assert isinstance(a.workspace, str)
    assert os.path.isabs(a.workspace)
    assert a.workspace == os.path.abspath(str(tmp_path))


def test_tool_timeout_validation_accepts_positive_int(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a = _make_agent(tmp_path, monkeypatch)
    output = asyncio.run(
        a._execute_tool_call("run", {"command": "echo timeout-ok", "timeout": 60})
    )
    assert "timeout-ok" in output


def test_tool_timeout_validation_rejects_non_positive_int(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a = _make_agent(tmp_path, monkeypatch)
    output = asyncio.run(
        a._execute_tool_call("run", {"command": "echo timeout-default", "timeout": 0})
    )
    assert "timeout-default" in output
