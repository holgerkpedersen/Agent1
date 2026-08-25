"""Tests for the subagent role registry and role-aware SubAgents.

Covers (plan phases 0-2):
- registry consistency: every role's tools exist in NLP_TOOL_NAMES,
  read-only roles stay inside PLAN_MODE_TOOLS and plan mode;
- SubAgent role wiring: unknown role rejection, mode capping by the
  parent, tool-whitelist enforcement BEFORE the parent choke point;
- the bounded tool loop: a scripted tool_calls response is executed
  through the whitelist and the loop terminates on a text answer;
- the turn cap feeding stuck-prevention;
- the rewritten ``subagent`` REPL command: registration in the real
  registry, async ``run`` (the old code returned an un-awaited coroutine).
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
from agent_core.commands.registry import CommandRegistry
from agent_core.modes import MODE_PLAN, PLAN_MODE_TOOLS
from agent_core.subagent_roles import (
    ROLES,
    get_role,
    role_names,
    validate_roles,
)
from agent_core.tool_schemas import NLP_TOOL_NAMES


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _ScriptedLLMClient:
    """LLM stand-in returning queued raw responses (JSON or plain text)."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or "test-model"
        self.script: list[str] = []
        self.calls: list[list[dict[str, Any]]] = []

    async def chat(
        self, messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None, **kwargs: Any,
    ) -> str:
        self.calls.append(list(messages))
        return self.script.pop(0) if self.script else "done."


def _make_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Agent:
    monkeypatch.setattr(agent, "LLMClient", _ScriptedLLMClient)
    return Agent(workspace=str(tmp_path), model_name="test-model")


def _tool_call_raw(name: str, args_json: str) -> str:
    return json.dumps({
        "content": "",
        "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": name, "arguments": args_json},
        }],
    })


# ---------------------------------------------------------------------------
# Registry consistency
# ---------------------------------------------------------------------------
class TestRoleRegistry:
    def test_validate_roles_finds_no_problems(self) -> None:
        assert validate_roles() == []

    def test_every_tool_exists_in_nlp_tool_names(self) -> None:
        for role in ROLES.values():
            assert role.tools_allowed <= NLP_TOOL_NAMES, role.name

    def test_read_only_roles_stay_within_plan_mode_tools(self) -> None:
        for role in ROLES.values():
            if role.read_only:
                assert role.tools_allowed <= PLAN_MODE_TOOLS, role.name
                assert role.mode == MODE_PLAN

    def test_expected_nine_roles_present(self) -> None:
        assert role_names() == [
            "debugger", "documenter", "implementer", "integrator",
            "planner", "researcher", "reviewer", "security", "tester",
        ]

    def test_get_role_is_case_insensitive_and_none_for_unknown(self) -> None:
        assert get_role(" PLANNER ") is ROLES["planner"]
        assert get_role("nope") is None


# ---------------------------------------------------------------------------
# SubAgent role wiring
# ---------------------------------------------------------------------------
class TestSubagentRoleWiring:
    def test_unknown_role_raises_valueerror(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        a = _make_agent(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="Unknown subagent role"):
            a.spawn_subagent("x", role="wizard")

    def test_role_sets_mode_and_whitelist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        a = _make_agent(tmp_path, monkeypatch)
        sub = a.spawn_subagent("impl", role="implementer")
        assert sub.role_name == "implementer"
        assert sub.mode == "build"
        assert "write" in sub._tools_allowed

    def test_plan_mode_parent_caps_build_child_to_readonly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        a = _make_agent(tmp_path, monkeypatch)
        a.mode = MODE_PLAN
        sub = a.spawn_subagent("impl", role="implementer")
        # Delegation must not become a plan-mode escape hatch.
        assert sub.mode == MODE_PLAN
        out = asyncio.run(sub._execute_tool_call("write", {"path": "x.py", "content": "1"}))
        assert "blocked" in out.lower()

    def test_disallowed_tool_rejected_before_parent_choke_point(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        a = _make_agent(tmp_path, monkeypatch)
        sub = a.spawn_subagent("rev", role="reviewer")
        called: list[str] = []
        monkeypatch.setattr(
            a, "_execute_tool_call",
            lambda n, args: called.append(n) or asyncio.sleep(0, result="ok"),
        )
        out = asyncio.run(sub._execute_tool_call("write", {"path": "x", "content": ""}))
        assert "not allowed" in out
        assert called == []  # never reached the parent executor


# ---------------------------------------------------------------------------
# Bounded tool loop inside a role subagent
# ---------------------------------------------------------------------------
class TestSubagentToolLoop:
    def test_scripted_tool_call_executes_and_loop_terminates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        a = _make_agent(tmp_path, monkeypatch)
        sub = a.spawn_subagent("plan", role="planner")
        target = tmp_path / "mod.py"
        target.write_text("VALUE = 41\n", encoding="utf-8")
        llm = sub.llm
        assert isinstance(llm, _ScriptedLLMClient)
        llm.script = [
            _tool_call_raw("search", json.dumps({"query": "VALUE"})),
            "Found VALUE in mod.py. Plan done.",
        ]
        out = asyncio.run(sub.respond("Where is VALUE?"))
        assert "Plan done." in out
        # The search really ran against the workspace through the whitelist.
        assert any("mod.py" in m[-1]["content"] for m in llm.calls[1:])

    def test_role_cannot_call_tools_outside_whitelist_in_loop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        a = _make_agent(tmp_path, monkeypatch)
        sub = a.spawn_subagent("plan", role="planner")
        llm = sub.llm
        assert isinstance(llm, _ScriptedLLMClient)
        llm.script = [
            _tool_call_raw("write", json.dumps({"path": "evil.py", "content": "x=1"})),
            "Trying to write.",
        ]
        asyncio.run(sub.respond("sabotage"))
        assert not (tmp_path / "evil.py").exists()

    def test_turn_cap_refuses_further_work(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        a = _make_agent(tmp_path, monkeypatch)
        sub = a.spawn_subagent("plan", role="planner")
        sub.role_spec = replace(sub.role_spec, max_turns=1)  # type: ignore[arg-type]
        llm = sub.llm
        assert isinstance(llm, _ScriptedLLMClient)
        llm.script = ["turn one answer"]
        assert "turn one answer" in asyncio.run(sub.respond("t1"))
        refusal = asyncio.run(sub.respond("t2"))
        assert "turn cap reached" in refusal
        assert len(llm.calls) == 1  # second request never hit the LLM


# ---------------------------------------------------------------------------
# REPL command
# ---------------------------------------------------------------------------
def _registry_with_subagent() -> CommandRegistry:
    reg = CommandRegistry()
    from agent_core.commands.subagent_cmd import SubAgentCommand
    reg.register(SubAgentCommand())
    return reg


class TestSubagentCommand:
    def test_registered_in_real_registry(self) -> None:
        reg = agent._build_registry()
        cmd = reg.get("subagent")
        assert cmd is not None
        assert "roles" in cmd.help_text

    def test_roles_listing_has_no_registry_problems(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        a = _make_agent(tmp_path, monkeypatch)
        asyncio.run(_registry_with_subagent().execute("subagent", ["roles"], a))
        out = capsys.readouterr().out
        assert "Available subagent roles:" in out
        assert "planner" in out and "! " not in out

    def test_create_with_unknown_role_prints_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        a = _make_agent(tmp_path, monkeypatch)
        asyncio.run(_registry_with_subagent().execute(
            "subagent", ["create", "w", "--role", "wizard"], a))
        assert "Unknown role 'wizard'" in capsys.readouterr().out

    def test_run_awaits_respond_and_prints_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        a = _make_agent(tmp_path, monkeypatch)
        sub = a.spawn_subagent("r1")
        llm = sub.llm
        assert isinstance(llm, _ScriptedLLMClient)
        llm.script = ["echo answer"]
        a._subagents = {"r1": sub}  # type: ignore[attr-defined]
        asyncio.run(_registry_with_subagent().execute(
            "subagent", ["run", "r1", "do", "stuff"], a))
        out = capsys.readouterr().out
        assert "[r1] echo answer" in out

    def test_list_and_reset_roundtrip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        a = _make_agent(tmp_path, monkeypatch)
        sub = a.spawn_subagent("r2")
        llm = sub.llm
        assert isinstance(llm, _ScriptedLLMClient)
        llm.script = ["hi"]
        asyncio.run(sub.respond("hello"))
        a._subagents = {"r2": sub}  # type: ignore[attr-defined]
        asyncio.run(_registry_with_subagent().execute(
            "subagent", ["list"], a))
        assert "role=None" not in capsys.readouterr().out
        asyncio.run(_registry_with_subagent().execute(
            "subagent", ["reset", "r2"], a))
        assert sub.get_conversation() == []


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------
class TestBackwardCompat:
    def test_roleless_subagent_behaviour_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        a = _make_agent(tmp_path, monkeypatch)
        sub = a.spawn_subagent("plain")
        assert sub.role_name is None
        assert sub.role_spec is None
        assert sub._tools_allowed == frozenset()
        out = asyncio.run(sub.respond("hello"))
        assert out.startswith("done.") or out == "done."
        conv = sub.get_conversation()
        assert [m["role"] for m in conv] == ["user", "assistant"]

    def test_spawn_signature_still_supports_workspace_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        a = _make_agent(tmp_path, monkeypatch)
        child = tmp_path / "child"
        child.mkdir()
        sub = a.spawn_subagent("w", workspace=str(child))
        assert sub.workspace == str(child.resolve())[:3] + sub.workspace[3:] \
            if False else True  # abs path equality below
        import os
        assert os.path.abspath(str(child)) == sub.workspace
