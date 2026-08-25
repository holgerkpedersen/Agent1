"""Tests for issue routing and the delegate/delegate_batch tools.

Covers (plan phases 2b-4):
- ``agent_core.issue_router``: deterministic keyword→role classification,
  planner fallback, valid registry names, human-readable table;
- ``Agent._nlp_delegate``: argument validation, concurrency cap,
  plan-mode capping of build-mode children, timeout surfacing as an
  error report, termination-reason propagation ("stopped early: cap"),
  and read-loop-streak isolation (child reads never inflate the parent);
- ``Agent._nlp_delegate_batch``: validation, dedupe + concurrency cap,
  merged per-role reports from parallel isolated children.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import agent
from agent import Agent
from agent_core.issue_router import (
    route_issue,
    route_with_score,
    routing_table,
)
from agent_core.modes import MODE_PLAN
from agent_core.subagent_roles import ROLES, get_role


class _EchoLLMClient:
    """Deterministic LLM stand-in: echoes the last user message.

    Optional per-instance ``script`` overrides the echo behaviour (consumed
    FIFO); a queued JSON tool_calls payload drives the child's tool loop.
    """

    instances: list["_EchoLLMClient"] = []

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or "test-model"
        self.script: list[str] = []
        self.calls: list[list[dict[str, Any]]] = []
        _EchoLLMClient.instances.append(self)

    async def chat(
        self, messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None, **kwargs: Any,
    ) -> str:
        self.calls.append(list(messages))
        if self.script:
            return self.script.pop(0)
        last = messages[-1]["content"] if messages else ""
        return f"ack: {last[:60]}"


def _tool_call_raw(name: str, args: dict[str, Any]) -> str:
    return json.dumps({
        "content": "",
        "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }],
    })


@pytest.fixture()
def bot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Agent:
    monkeypatch.setattr(agent, "LLMClient", _EchoLLMClient)
    _EchoLLMClient.instances.clear()
    return Agent(workspace=str(tmp_path), model_name="test-model")


# ---------------------------------------------------------------------------
# Issue router
# ---------------------------------------------------------------------------
class TestIssueRouter:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Secret key hardcoded in settings.py", "security"),
            ("TypeError: cannot read property of None in agent.py", "debugger"),
            ("pytest suite fails after refactor", "tester"),
            ("please write a conventional commit message and push", "integrator"),
            ("the README flags are stale", "documenter"),
            ("code review my diff before committing", "reviewer"),
            ("implement pagination for the dashboard API", "implementer"),
        ],
    )
    def test_keyword_routing(self, text: str, expected: str) -> None:
        assert route_issue(text) == expected

    def test_unknown_text_falls_back_to_planner(self) -> None:
        role, score = route_with_score("")
        assert role == "planner"
        assert score == 0

    def test_score_counts_distinct_keyword_hits(self) -> None:
        _role, score = route_with_score("traceback shows TypeError and KeyError crash")
        assert score >= 3

    def test_every_routed_role_exists_in_registry(self) -> None:
        for sample in (
            "bug", "tests fail", "commit", "docs", "review",
            "new feature", "vulnerability", "", "latest version of requests?",
        ):
            assert get_role(route_issue(sample)) is not None

    def test_routing_table_lists_all_roles(self) -> None:
        table = routing_table()
        for name in ROLES:
            assert name in table


# ---------------------------------------------------------------------------
# delegate (single)
# ---------------------------------------------------------------------------
class TestDelegateTool:
    def test_requires_role_and_task(self, bot: Agent) -> None:
        out = asyncio.run(bot._nlp_delegate({"role": "planner"}))
        assert "requires 'role' and 'task'" in out

    def test_unknown_role_rejected(self, bot: Agent) -> None:
        out = asyncio.run(bot._nlp_delegate({"role": "wizard", "task": "x"}))
        assert "unknown role 'wizard'" in out
        assert "planner" in out  # available roles listed

    def test_concurrency_cap(self, bot: Agent) -> None:
        bot._active_subagents = {"a", "b", "c"}
        out = asyncio.run(bot._nlp_delegate({"role": "planner", "task": "x"}))
        assert "already running" in out

    def test_happy_path_returns_answer_summary_and_meta(
        self, bot: Agent, tmp_path: Path,
    ) -> None:
        (tmp_path / "m.py").write_text("x = 1\n", encoding="utf-8")
        out = asyncio.run(bot._nlp_delegate({
            "role": "planner", "task": "inspect m.py",
        }))
        assert "ack: [SubAgent:planner-1] inspect m.py" in out
        assert "context summary" in out
        assert "[delegate:planner-1] finished (Planner/Triage, plan mode)." in out
        # Concurrency slot released afterwards.
        assert bot._active_subagents == set()

    def test_plan_mode_parent_caps_build_child(
        self, bot: Agent, tmp_path: Path,
    ) -> None:
        bot.mode = MODE_PLAN
        out = asyncio.run(bot._nlp_delegate({
            "role": "implementer", "task": "write something",
        }))
        assert "capped to read-only" in out
        # And the child really could not have written anything.
        assert not list(tmp_path.glob("*.py"))

    def test_timeout_surfaces_as_error_not_crash(
        self, bot: Agent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(agent, "_DELEGATE_TIMEOUT_S", 0.01)

        class _Slow(_EchoLLMClient):
            async def chat(self, messages, tools=None, **kw):  # type: ignore[override]
                await asyncio.sleep(5)
                return "never"

        monkeypatch.setattr(agent, "LLMClient", _Slow)
        out = asyncio.run(bot._nlp_delegate({"role": "planner", "task": "slow"}))
        assert "timed out after 0s" in out
        assert bot._active_subagents == set()

    def test_child_exception_reported(self, bot: Agent, monkeypatch) -> None:
        def _boom(name, workspace=None, role=None):
            raise RuntimeError("spawn failed")

        monkeypatch.setattr(bot, "spawn_subagent", _boom)
        out = asyncio.run(bot._nlp_delegate({"role": "planner", "task": "x"}))
        assert "failed to spawn" in out
        # The concurrency slot must be released even when spawning blew up.
        assert bot._active_subagents == set()

    def test_termination_reason_propagated_on_early_stop(
        self, bot: Agent,
    ) -> None:
        # Child script: first response is a tool call (list_files), loop
        # budget is 1 iteration -> cap fires and forced synthesis consumes
        # the second queued item.  The parent report must say so.
        real_spawn = bot.spawn_subagent

        def _seeded(name, workspace=None, role=None):
            sub = real_spawn(name, workspace=workspace, role=role)
            sub.role_spec = replace(sub.role_spec, max_turns=1)  # type: ignore[arg-type]
            llm = sub.llm
            assert isinstance(llm, _EchoLLMClient)
            llm.script = [
                _tool_call_raw("list_files", {}),
                "synthesized partial answer.",
            ]
            return sub

        bot.spawn_subagent = _seeded  # type: ignore[method-assign]
        out = asyncio.run(bot._nlp_delegate({"role": "planner", "task": "t"}))
        assert "stopped early: cap" in out
        assert "synthesized partial answer." in out

    def test_child_reads_do_not_inflate_parent_read_streak(
        self, bot: Agent, tmp_path: Path,
    ) -> None:
        f = tmp_path / "m.py"
        f.write_text("x = 1\n", encoding="utf-8")
        bot._read_streak = agent._MAX_CONSECUTIVE_READS - 1
        bot._delegating = True
        try:
            for _ in range(5):
                out = asyncio.run(bot._nlp_read({"path": str(f)}))
                assert "read-loop guard" not in out
            assert bot._read_streak == agent._MAX_CONSECUTIVE_READS - 1
        finally:
            bot._delegating = False
        # Parent's own reads still count (and still trigger the note).
        asyncio.run(bot._nlp_read({"path": str(f)}))
        assert bot._read_streak == agent._MAX_CONSECUTIVE_READS


# ---------------------------------------------------------------------------
# delegate_batch
# ---------------------------------------------------------------------------
class TestDelegateBatchTool:
    def test_validation(self, bot: Agent) -> None:
        assert "must be a list" in asyncio.run(
            bot._nlp_delegate_batch({"roles": "planner", "task": "x"}))
        assert "requires 'roles'" in asyncio.run(
            bot._nlp_delegate_batch({"roles": [], "task": "x"}))
        assert "unknown role(s): wizard" in asyncio.run(
            bot._nlp_delegate_batch({"roles": ["wizard"], "task": "x"}))

    def test_dedupe_and_concurrency_cap(self, bot: Agent) -> None:
        out = asyncio.run(bot._nlp_delegate_batch({
            "roles": ["planner", "reviewer", "security", "tester", "tester"],
            "task": "triage this",
        }))
        assert "3 role(s)" in out
        assert "(2 extra role(s) dropped" in out
        for section in ("### planner", "### reviewer", "### security"):
            assert section in out

    def test_parallel_children_answer_independently(self, bot: Agent) -> None:
        out = asyncio.run(bot._nlp_delegate_batch({
            "roles": ["planner", "reviewer"],
            "task": "look at module X",
        }))
        assert "[delegate_batch:1] 2 role(s)" in out
        # Each child echoed ITS OWN prefixed task back — proof the histories
        # are isolated (planner never saw reviewer's task and vice versa).
        assert "ack: [SubAgent:planner-b1-0] look at module X" in out
        assert "ack: [SubAgent:reviewer-b1-1] look at module X" in out

    def test_batch_children_are_isolated_instances(self, bot: Agent) -> None:
        before = len(_EchoLLMClient.instances)
        asyncio.run(bot._nlp_delegate_batch({
            "roles": ["researcher", "documenter"], "task": "t",
        }))
        new = _EchoLLMClient.instances[before:]
        assert len(new) == 2
        assert new[0] is not new[1]
