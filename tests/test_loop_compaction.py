"""Regression tests for in-run context compaction (decision #B-5-compact).

The tool loop previously only APPENDED to the live message list, so a long
autonomous/NLP task shipped the entire (ever-growing) transcript to the model
on every iteration.  That made per-iteration prompt processing grow linearly
with history and eventually trip the llama-server client socket timeout (the
"Client disconnected" symptom).  These tests lock in the fix: the loop must
drop the oldest CLOSED tool-exchange prefix once the list passes a bound,
never split an assistant/tool pair, and never drop the live (pending) tail.
"""
import json
from typing import Any

import pytest

from agent_core.llm.tool_loop import (
    GUARD_COMPACT,
    ToolLoopRunner,
    _compact_messages,
    _split_for_compaction,
)
from agent_core.tool_schemas import NLP_TOOL_SCHEMAS


def _make_llm_chat_fn(fake_llm):
    """Wrap a scripted fake LLM the same way chat_nlp wraps the real provider."""
    async def llm_chat_fn(messages, tools):
        raw = await fake_llm.chat(messages, tools)
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict) and parsed.get("tool_calls"):
            parsed.pop("role", None)
            updated = list(messages)
            updated.append(
                {"role": "assistant", "content": parsed.get("content") or "", **parsed})
            return str(parsed.get("content") or ""), updated
        updated = list(messages)
        updated.append({"role": "assistant", "content": raw})
        return raw, updated

    return llm_chat_fn


class _ScriptedLLM:
    """Scripted fake: each entry is a (tool_name, args) call or a text answer."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def chat(self, messages, tools=None, **kwargs):
        self.calls.append(list(messages))
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


def _loop_runner_sync(runner, fake_llm, execute_tool, **kwargs):
    import asyncio
    return asyncio.run(runner.run(
        messages=[{"role": "user", "content": "do the thing"}],
        llm_chat_fn=_make_llm_chat_fn(fake_llm),
        execute_tool_fn=execute_tool,
        tools=list(NLP_TOOL_SCHEMAS),
        **kwargs,
    ))


def _build_history(n_exchanges: int, tail_text: str = "answer") -> list[dict[str, Any]]:
    """Build a message list with ``n_exchanges`` closed tool exchanges."""
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
    ]
    for i in range(n_exchanges):
        msgs.append({
            "role": "assistant",
            "content": f"call {i}",
            "tool_calls": [{
                "id": f"t{i}",
                "type": "function",
                "function": {"name": "read", "arguments": json.dumps({"path": f"f{i}"})},
            }],
        })
        msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": f"result {i}"})
    msgs.append({"role": "assistant", "content": tail_text})
    return msgs


def _strip_system(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the leading system prompt so tests can build tool-only histories."""
    return [m for m in messages if m.get("role") != "system"]


class TestSplitForCompaction:
    def test_no_drop_when_under_bound(self):
        msgs = _build_history(5)
        assert _split_for_compaction(msgs, 80, 1_000_000) == (None, None)

    def test_drops_oldest_closed_prefix(self):
        # No system prompt: the leading task prompt (first user) is preserved.
        msgs = _strip_system(_build_history(10))
        dropped, kept = _split_for_compaction(msgs, 8, 1_000_000)
        assert dropped is not None
        # Leading task prompt preserved in the kept list.
        assert kept[0] == {"role": "user", "content": "task"}
        # The kept list ends with the live assistant answer.
        assert kept[-1]["role"] == "assistant"

    def test_never_splits_assistant_tool_pair(self):
        msgs = _strip_system(_build_history(10))
        dropped, kept = _split_for_compaction(msgs, 8, 1_000_000)
        # A tool message in `kept` must always have its owning assistant call.
        assistant_ids = {
            tc["id"]
            for m in kept
            if m.get("role") == "assistant" and m.get("tool_calls")
            for tc in m["tool_calls"]
        }
        for m in kept:
            if m.get("role") == "tool":
                assert m["tool_call_id"] in assistant_ids

    def test_char_budget_respected(self):
        msgs = _strip_system(_build_history(10))
        # Tight count bound (8) so compaction triggers, with a tiny char budget
        # (400) the kept tail must honour.  The leading task prompt is fixed
        # and excluded from the budget.
        dropped, kept = _split_for_compaction(msgs, 8, 400)
        assert dropped is not None
        total = sum(len(json.dumps(m, default=str)) for m in kept)
        assert total <= 400 + 400  # tolerance for the fixed task prompt


class TestCompactMessages:
    def test_inserts_stripable_note(self):
        msgs = _build_history(10)
        out, note = _compact_messages(msgs, 8, 1_000_000, "summary text")
        assert note is not None
        notes = [m for m in out if m.get("content", "").startswith("CONTEXT NOTE:")]
        assert notes, "compacted history must carry a CONTEXT NOTE"
        assert "summary text" in notes[0]["content"]

    def test_noop_under_bound(self):
        msgs = _build_history(3)
        out, note = _compact_messages(msgs, 80, 1_000_000, "x")
        assert note is None
        assert out is msgs


class TestInLoopCompaction:
    def test_compacts_during_run(self):
        """A long scripted run must plateau the message count via compaction."""
        script = []
        for i in range(20):
            script.append(("read", {"path": f"f{i}.py"}))
        script.append("Final answer after many calls.")
        # Low threshold triggers compaction well before the 20 calls finish.
        fake = _ScriptedLLM(script)

        async def execute_tool(name, args):
            return f"content of {args.get('path')}"

        runner = ToolLoopRunner(
            max_iterations=30, in_loop_compact=True,
            compact_strategy="note", compact_max_messages=6,
        )
        final_text, messages = _loop_runner_sync(runner, fake, execute_tool)

        assert final_text == "Final answer after many calls."
        # The live message list must never balloon past the bound + slack.
        assert len(messages) <= runner.compact_max_messages + 4
        # The compact note is stripped from the returned history at run end.
        assert not any(
            m.get("content", "").startswith("CONTEXT NOTE:") for m in messages
        )

    def test_compact_guard_emitted_in_trace(self):
        script = []
        for i in range(12):
            script.append(("read", {"path": f"f{i}.py"}))
        script.append("Done.")
        fake = _ScriptedLLM(script)
        sink = _TraceSink()

        async def execute_tool(name, args):
            return "ok"

        runner = ToolLoopRunner(
            max_iterations=20, in_loop_compact=True,
            compact_strategy="note", compact_max_messages=6, trace=sink,
        )
        _loop_runner_sync(runner, fake, execute_tool)
        guards = [e for e in sink.events if e.get("kind") == "guard_triggered"
                  and e.get("guard") == GUARD_COMPACT]
        assert guards, "compaction must emit a GUARD_COMPACT trace event"

    def test_disabled_does_not_compact(self):
        script = []
        for i in range(12):
            script.append(("read", {"path": f"f{i}.py"}))
        script.append("Done.")
        fake = _ScriptedLLM(script)
        sink = _TraceSink()

        async def execute_tool(name, args):
            return "ok"

        runner = ToolLoopRunner(
            max_iterations=20, in_loop_compact=False, compact_max_messages=6,
            trace=sink,
        )
        _loop_runner_sync(runner, fake, execute_tool)
        guards = [e for e in sink.events if e.get("kind") == "guard_triggered"
                  and e.get("guard") == GUARD_COMPACT]
        assert not guards


class _TraceSink:
    def __init__(self):
        self.events = []

    def emit(self, record):
        self.events.append(record)
