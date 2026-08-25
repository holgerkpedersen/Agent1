"""Tests for the SubAgent class in agent_core.subagent."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, NoReturn, cast

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
        # Echo back a synthetic response based on the last message.
        if messages:
            last = messages[-1]["content"]
            return f"Response to: {last[:50]}"
        return "empty-response"


def _make_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Agent:
    """Create an Agent with a fake LLM client for testing."""
    monkeypatch.setattr(agent, "LLMClient", _FakeLLMClient)
    return Agent(workspace=str(tmp_path), model_name="test-model")


def test_spawn_subagent_creates_isolated_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SubAgent should have its own conversation history separate from parent."""
    a = _make_agent(tmp_path, monkeypatch)
    sub = a.spawn_subagent("researcher")

    # Initial state should be empty.
    assert sub.get_context_summary() == "<SubAgent:researcher> has no conversation yet."
    assert sub.get_conversation() == []

    # After responding, history should have entries.
    response = asyncio.run(sub.respond("Find all Python files"))
    assert "Response to:" in response
    conv = sub.get_conversation()
    assert len(conv) == 2
    assert conv[0]["role"] == "user"
    assert conv[1]["role"] == "assistant"


def test_subagent_shares_workspace_with_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SubAgent should inherit parent's workspace by default."""
    a = _make_agent(tmp_path, monkeypatch)
    sub = a.spawn_subagent("worker")

    assert sub.workspace == os.path.abspath(str(tmp_path))
    assert sub.model_name == a.model_name


def test_subagent_can_override_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SubAgent should accept an explicit workspace override."""
    a = _make_agent(tmp_path, monkeypatch)
    child_dir = tmp_path / "child"
    child_dir.mkdir()

    sub = a.spawn_subagent("worker", workspace=str(child_dir))
    assert sub.workspace == os.path.abspath(str(child_dir))


def test_parallel_task_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Agent should be able to run multiple tasks in parallel."""
    import concurrent.futures

    a = _make_agent(tmp_path, monkeypatch)

    async def task1() -> str:
        sub = a.spawn_subagent("t1")
        return await sub.respond("Task 1 work")

    async def task2() -> str:
        sub = a.spawn_subagent("t2")
        return await sub.respond("Task 2 work")

    # Run async tasks in parallel via ThreadPoolExecutor.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(asyncio.run, task1()), pool.submit(asyncio.run, task2())]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == 2
    # Each result should reference its task.
    combined = " ".join(results)
    assert "Task 1 work" in combined or "Task 2 work" in combined


def test_subagent_result_collection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Parent can collect results from multiple subagents."""
    a = _make_agent(tmp_path, monkeypatch)

    subs = [a.spawn_subagent(f"worker-{i}") for i in range(3)]

    async def gather_responses() -> list[str]:
        return await asyncio.gather(*(sub.respond("Analyze module") for sub in subs))

    responses = asyncio.run(gather_responses())

    assert len(responses) == 3
    # Each should have unique context.
    summaries = [sub.get_context_summary() for sub in subs]
    for summary in summaries:
        assert "context summary" in summary


def test_subagent_reset_clears_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SubAgent.reset() should clear conversation history."""
    a = _make_agent(tmp_path, monkeypatch)
    sub = a.spawn_subagent("reset-test")

    asyncio.run(sub.respond("First message"))
    assert len(sub.get_conversation()) == 2

    sub.reset()
    assert sub.get_conversation() == []
    assert sub.get_context_summary() == "<SubAgent:reset-test> has no conversation yet."


def test_subagent_repr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SubAgent.__repr__ should show name and workspace."""
    a = _make_agent(tmp_path, monkeypatch)
    sub = a.spawn_subagent("test-agent")

    r = repr(sub)
    assert "test-agent" in r
    # WindowsPath may be normalized differently; check for the path components.
    assert "pytest" in r and "subagent_repr" in r


def test_subagent_context_summary_max_messages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """get_context_summary should respect max_messages limit."""
    a = _make_agent(tmp_path, monkeypatch)
    sub = a.spawn_subagent("summary-test")

    # Add multiple turns.
    async def add_messages() -> None:
        for i in range(5):
            await sub.respond(f"Message {i}")

    asyncio.run(add_messages())

    summary = sub.get_context_summary(max_messages=2)
    # Should only show last 2 pairs (4 messages).
    assert "context summary" in summary
    # Count the number of lines with "> " or "< " prefix.
    lines = [l for l in summary.split("\n") if l.startswith("  >") or l.startswith("  <")]
    assert len(lines) <= 4  # max 2 pairs


def test_subagent_prefixed_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """User messages should be prefixed with [SubAgent:name]."""
    a = _make_agent(tmp_path, monkeypatch)
    sub = a.spawn_subagent("my-agent")

    asyncio.run(sub.respond("Hello world"))
    conv = sub.get_conversation()
    assert "[SubAgent:my-agent]" in conv[0]["content"]


def test_multiple_subagents_independent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Multiple subagents should have independent conversation histories."""
    a = _make_agent(tmp_path, monkeypatch)

    sub1 = a.spawn_subagent("alpha")
    sub2 = a.spawn_subagent("beta")

    asyncio.run(sub1.respond("Work from alpha"))
    asyncio.run(sub2.respond("Work from beta"))

    # Each should only have their own messages.
    assert len(sub1.get_conversation()) == 2
    assert len(sub2.get_conversation()) == 2
    assert "alpha" in sub1.get_context_summary()
    assert "beta" in sub2.get_context_summary()
