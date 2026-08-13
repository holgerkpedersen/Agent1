"""Tests for the structured NLP tool-calling loop.

The conversational agent must actually EXECUTE tool calls — never just
describe them.  These tests drive ``ToolLoopRunner`` with a scripted fake
LLM that emits native ``tool_calls`` and verify that the dispatcher runs
the tools, feeds results back, and terminates on a plain-text answer.
"""
import json
from unittest.mock import AsyncMock, patch

import pytest

from agent_core.llm.tool_loop import ToolLoopRunner
from agent_core.tool_schemas import NLP_TOOL_NAMES, NLP_TOOL_SCHEMAS


class _ScriptedLLM:
    """A fake LLM that returns a fixed sequence of responses.

    Each entry is either a plain-text answer (loop terminates) or a
    ``(tool_name, args_dict)`` call that gets serialized to OpenAI-style
    ``tool_calls`` in the assistant message.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def chat(self, messages, tools=None, **kwargs):
        self.calls.append((list(messages), tools))
        step = self.script.pop(0)
        if isinstance(step, tuple):
            tool_name, args = step
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args),
                    },
                }],
            }
            return json.dumps(message)
        return step


def _make_llm_chat_fn(fake_llm):
    """Wrap a _ScriptedLLM the same way chat_nlp wraps the real provider."""
    async def llm_chat_fn(messages, tools):
        raw = await fake_llm.chat(messages, tools)
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict) and parsed.get("tool_calls"):
            parsed.pop("role", None)
            updated = list(messages)
            updated.append({"role": "assistant", "content": parsed.get("content") or "", **parsed})
            return str(parsed.get("content") or ""), updated
        updated = list(messages)
        updated.append({"role": "assistant", "content": raw})
        return raw, updated

    return llm_chat_fn


class TestToolLoopExecution:
    def test_executes_tools_and_terminates_on_text(self):
        fake = _ScriptedLLM([
            ("read", {"path": "agent.py"}),
            ("search", {"query": "def chat_nlp"}),
            "Summary of what I found.",
        ])
        executed = []

        async def execute_tool(name, args):
            executed.append((name, args))
            return f"result-of-{name}"

        runner = ToolLoopRunner(max_iterations=5)
        final_text, messages = _loop_runner_sync(runner, fake, execute_tool)

        assert final_text == "Summary of what I found."
        assert [e[0] for e in executed] == ["read", "search"]
        # Tool results must be fed back to the LLM as role:tool messages.
        assert sum(1 for m in messages if m["role"] == "tool") == 2

    def test_stops_when_no_tool_calls(self):
        fake = _ScriptedLLM(["Just an answer, no tools."])
        executed = []

        async def execute_tool(name, args):
            executed.append(name)
            return "x"

        runner = ToolLoopRunner(max_iterations=5)
        final_text, _ = _loop_runner_sync(runner, fake, execute_tool)

        assert final_text == "Just an answer, no tools."
        assert executed == []

    def test_tool_error_is_fed_back_not_crashing(self):
        fake = _ScriptedLLM([
            ("read", {"path": "missing.py"}),
            "I got an error.",
        ])
        executed = []

        async def execute_tool(name, args):
            executed.append(name)
            raise RuntimeError("boom")

        runner = ToolLoopRunner(max_iterations=5)
        final_text, messages = _loop_runner_sync(runner, fake, execute_tool)

        assert final_text == "I got an error."
        tool_msg = next(m for m in messages if m["role"] == "tool")
        assert "Tool error: boom" in tool_msg["content"]


def _loop_runner_sync(runner, fake_llm, execute_tool):
    import asyncio
    return asyncio.run(runner.run(
        messages=[{"role": "user", "content": "do the thing"}],
        llm_chat_fn=_make_llm_chat_fn(fake_llm),
        execute_tool_fn=execute_tool,
        tools=list(NLP_TOOL_SCHEMAS),
    ))


class TestNlpToolSchemas:
    def test_all_schema_names_are_valid_identifiers(self):
        for name in NLP_TOOL_NAMES:
            assert name.isidentifier(), name

    def test_every_schema_has_name_in_names_set(self):
        for schema in NLP_TOOL_SCHEMAS:
            assert schema["function"]["name"] in NLP_TOOL_NAMES


class TestAgentExecuteToolCall:
    """Drive Agent._execute_tool_call with a real temp file so the
    verification summary is produced for write/edit."""

    def test_write_verifies_python_file(self, tmp_path):
        import asyncio
        from agent import Agent
        agent = Agent(workspace=str(tmp_path))
        target = tmp_path / "hello.py"

        async def run():
            return await agent._execute_tool_call("write", {
                "path": str(target),
                "content": "def hello() -> str:\n    return 'hi'\n",
            })

        result = asyncio.run(run())
        assert "Written" in result
        assert "[verify] py_compile" in result
        assert target.read_text(encoding="utf-8").startswith("def hello")

    def test_edit_verifies_and_replaces_first_occurrence(self, tmp_path):
        import asyncio
        from agent import Agent
        agent = Agent(workspace=str(tmp_path))
        target = tmp_path / "t.py"
        target.write_text("x = 1\nx = 2\n", encoding="utf-8")

        async def run():
            return await agent._execute_tool_call("edit", {
                "path": str(target),
                "old_text": "x = 1",
                "new_text": "x = 10",
            })

        result = asyncio.run(run())
        assert "Edited" in result
        assert "[verify] py_compile" in result
        assert target.read_text(encoding="utf-8") == "x = 10\nx = 2\n"

    def test_unknown_tool_returns_error(self):
        import asyncio
        from agent import Agent
        agent = Agent(workspace=".")

        async def run():
            return await agent._execute_tool_call("no_such_tool", {})

        assert "Unknown tool" in asyncio.run(run())
