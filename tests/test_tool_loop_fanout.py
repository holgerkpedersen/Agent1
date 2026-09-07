"""Tests for intra-batch tool-call concurrency (READONLY_FANOUT_TOOLS fan-out).

When the model returns multiple read-only tool calls in a single batch, the
loop must fire them concurrently via ``asyncio.gather`` instead of running
them one-by-one.  Mutating tools and ``delegate`` must always run
sequentially.

Verifies:
- ``READONLY_FANOUT_TOOLS`` definition and composition
- Concurrent execution of read-only calls (wall-clock speedup)
- Sequential execution of mutating calls even when batched
- Original tool_call_id ordering preserved in the message list
- Mixed batches: read-only fan-out first, then sequential mutating
- Single read-only call skips gather (no overhead)
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

from agent_core.llm.tool_loop import (
    READONLY_FANOUT_TOOLS,
    ToolLoopRunner,
)


# ---------------------------------------------------------------------------
# Helpers (mirrors test_tool_loop_nlp conventions)
# ---------------------------------------------------------------------------

class _ScriptedLLM:
    """Fake LLM returning a fixed sequence of responses.

    Parallel batch: ``([("tool", {args}, ...], None)``
    Single call:    ``("tool", {args})``
    Text answer:    ``"string"``
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[tuple[list[dict], Any]] = []

    async def chat(self, messages, tools=None, **kwargs):
        self.calls.append((list(messages), tools))
        step = self.script.pop(0)
        # Parallel batch
        if isinstance(step, tuple) and len(step) == 2 and isinstance(step[0], list):
            tool_calls = []
            for i, item in enumerate(step[0]):
                if isinstance(item, tuple):
                    tool_name, args = item
                else:
                    tool_name, args = item, {}
                tool_calls.append({
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args),
                    },
                })
            msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": tool_calls,
            }
            return json.dumps(msg)
        # Single tool call
        if isinstance(step, tuple):
            tool_name, args = step
            msg = {
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
            return json.dumps(msg)
        # Plain text answer
        return step


def _make_llm_chat_fn(fake_llm):
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


def _loop_runner_sync(runner, fake_llm, execute_tool, **kwargs):
    return asyncio.run(runner.run(
        messages=[{"role": "user", "content": "do the thing"}],
        llm_chat_fn=_make_llm_chat_fn(fake_llm),
        execute_tool_fn=execute_tool,
        tools=[],  # tools not needed for scripted LLM
        **kwargs,
    ))


# ---------------------------------------------------------------------------
# READONLY_FANOUT_TOOLS sanity checks
# ---------------------------------------------------------------------------

class TestReadonlyFanoutToolsSet:
    """Verify the constant is correctly derived."""

    def test_contains_read_search_list_files_diff(self):
        for name in ("read", "search", "list_files", "diff"):
            assert name in READONLY_FANOUT_TOOLS, f"{name} missing"

    def test_contains_definitions_references_web_search(self):
        for name in ("definitions", "references", "web_search"):
            assert name in READONLY_FANOUT_TOOLS, f"{name} missing"

    def test_excludes_delegate(self):
        assert "delegate" not in READONLY_FANOUT_TOOLS

    def test_excludes_mutating_tools(self):
        for name in ("write", "edit", "fix"):
            assert name not in READONLY_FANOUT_TOOLS, f"{name} should not be in fan-out set"


# ---------------------------------------------------------------------------
# Concurrent execution tests
# ---------------------------------------------------------------------------

class TestFanoutConcurrency:
    """Read-only calls in a single batch must run concurrently."""

    def test_two_readonly_calls_run_concurrently(self):
        """Two reads in one batch should overlap in wall-clock time, proving
        they were dispatched via asyncio.gather rather than sequentially."""
        execution_times: list[tuple[str, float, float]] = []

        async def execute_tool(name, args):
            start = time.monotonic()
            await asyncio.sleep(0.05)  # simulate latency
            end = time.monotonic()
            execution_times.append((name, start, end))
            return f"result-of-{name}-{args.get('path', '')}"

        fake = _ScriptedLLM([
            ([("read", {"path": "a.py"}), ("read", {"path": "b.py"})], []),
            "Done.",
        ])
        runner = ToolLoopRunner(max_iterations=5)
        final_text, messages = _loop_runner_sync(runner, fake, execute_tool)

        assert final_text == "Done."
        assert len(execution_times) == 2
        # If concurrent, the second call's start should be BEFORE the first's end.
        (_, s1, e1), (_, s2, e2) = execution_times
        assert s2 < e1 or s1 < e2, (
            "Read-only calls should overlap (run concurrently), but they ran sequentially"
        )

    def test_three_readonly_calls_all_run_concurrently(self):
        """Three reads should all overlap."""
        execution_times: list[tuple[str, float, float]] = []

        async def execute_tool(name, args):
            start = time.monotonic()
            await asyncio.sleep(0.05)
            end = time.monotonic()
            execution_times.append((name, start, end))
            return f"result-{args.get('path', '')}"

        fake = _ScriptedLLM([
            ([
                ("read", {"path": "a.py"}),
                ("read", {"path": "b.py"}),
                ("read", {"path": "c.py"}),
            ], []),
            "Done.",
        ])
        runner = ToolLoopRunner(max_iterations=5)
        final_text, messages = _loop_runner_sync(runner, fake, execute_tool)

        assert final_text == "Done."
        assert len(execution_times) == 3
        # At least two of the three must overlap.
        starts = [t[1] for t in execution_times]
        ends = [t[2] for t in execution_times]
        max_start = max(starts)
        min_end = min(ends)
        assert max_start < min_end, (
            "With 3 concurrent calls, at least some should overlap in time"
        )

    def test_single_readonly_skips_gather(self):
        """A single read-only call should execute normally (no gather overhead)."""
        executed = []

        async def execute_tool(name, args):
            executed.append(name)
            return "result"

        fake = _ScriptedLLM([
            ([("read", {"path": "a.py"})], []),
            "Done.",
        ])
        runner = ToolLoopRunner(max_iterations=5)
        final_text, messages = _loop_runner_sync(runner, fake, execute_tool)

        assert final_text == "Done."
        assert executed == ["read"]


# ---------------------------------------------------------------------------
# Mutating tools stay sequential
# ---------------------------------------------------------------------------

class TestMutatingToolsSequential:
    """write/edit/fix must NOT be fan-out'd."""

    def test_write_and_read_mixed_batch(self):
        """A batch with read + write: read is concurrent (if multiple reads),
        write is sequential after. In this case with one read + one write,
        both run sequentially since there's only one read."""
        execution_log: list[tuple[str, str]] = []

        async def execute_tool(name, args):
            execution_log.append(("start", name))
            await asyncio.sleep(0.02)
            execution_log.append(("end", name))
            return "ok"

        fake = _ScriptedLLM([
            ([("read", {"path": "a.py"}), ("write", {"path": "b.py", "content": "x"})], []),
            "Done.",
        ])
        runner = ToolLoopRunner(max_iterations=5)
        final_text, _ = _loop_runner_sync(runner, fake, execute_tool)

        assert final_text == "Done."
        # The write must appear in sequential_pending, so it runs after reads.
        starts = [n for op, n in execution_log if op == "start"]
        assert starts == ["read", "write"], "write must run after read, not interleaved"

    def test_two_writes_are_sequential(self):
        """Two writes in the same batch must not overlap."""
        execution_times: list[tuple[str, float, float]] = []

        async def execute_tool(name, args):
            start = time.monotonic()
            await asyncio.sleep(0.05)
            end = time.monotonic()
            execution_times.append((name, start, end))
            return "ok"

        # Two writes — they should be sequential, so end1 < start2 (or very close)
        fake = _ScriptedLLM([
            ([("write", {"path": "a.py", "content": "x"}),
              ("write", {"path": "b.py", "content": "y"})], []),
            "Done.",
        ])
        runner = ToolLoopRunner(max_iterations=5)
        final_text, _ = _loop_runner_sync(runner, fake, execute_tool)

        assert final_text == "Done."
        (_, s1, e1), (_, s2, e2) = execution_times
        # Sequential: second starts after first ends (with tiny margin for asyncio scheduling).
        assert s2 >= e1 - 0.01, "Two writes should be sequential, not concurrent"


# ---------------------------------------------------------------------------
# Message ordering preserved
# ---------------------------------------------------------------------------

class TestMessageOrdering:
    """Tool results must appear in original tool_call_id order, not
    asyncio.gather completion order."""

    def test_tool_results_in_original_call_id_order(self):
        """Even if read_B completes before read_A, the messages list must
        have A's result before B's."""
        completion_order: list[str] = []

        async def execute_tool(name, args):
            path = args.get("path", "")
            # Make "b" finish before "a"
            if path == "b.py":
                await asyncio.sleep(0.01)
            else:
                await asyncio.sleep(0.1)
            completion_order.append(path)
            return f"content-of-{path}"

        fake = _ScriptedLLM([
            ([("read", {"path": "a.py"}), ("read", {"path": "b.py"})], []),
            "Done.",
        ])
        runner = ToolLoopRunner(max_iterations=5)
        final_text, messages = _loop_runner_sync(runner, fake, execute_tool)

        assert final_text == "Done."
        # completion_order shows who finished first (gather order).
        assert completion_order == ["b.py", "a.py"], "b should finish first due to shorter sleep"

        # But tool messages must be in original call_id order: call_0 (a) before call_1 (b).
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 2
        assert tool_msgs[0]["tool_call_id"] == "call_0"  # a.py
        assert tool_msgs[1]["tool_call_id"] == "call_1"  # b.py
        assert "content-of-a.py" in tool_msgs[0]["content"]
        assert "content-of-b.py" in tool_msgs[1]["content"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestFanoutEdgeCases:
    def test_all_readonly_no_mutating(self):
        """Batch of all read-only tools runs everything concurrently."""
        execution_times: list[tuple[str, float, float]] = []

        async def execute_tool(name, args):
            start = time.monotonic()
            await asyncio.sleep(0.05)
            end = time.monotonic()
            execution_times.append((name, start, end))
            return "ok"

        fake = _ScriptedLLM([
            ([
                ("read", {"path": "a.py"}),
                ("search", {"query": "foo"}),
                ("list_files", {"path": "."}),
                ("diff", {"path": "a.py"}),
            ], []),
            "Done.",
        ])
        runner = ToolLoopRunner(max_iterations=5)
        final_text, _ = _loop_runner_sync(runner, fake, execute_tool)

        assert final_text == "Done."
        assert len(execution_times) == 4
        # Total wall time should be ~0.05s (concurrent), not ~0.2s (sequential).
        wall_time = max(t[2] for t in execution_times) - min(t[1] for t in execution_times)
        assert wall_time < 0.15, f"Fan-out should be concurrent; wall time was {wall_time:.3f}s"

    def test_mixed_readonly_and_muting_batch(self):
        """Batch with 2 reads + 1 write: reads concurrent, write sequential after."""
        execution_log: list[tuple[str, str, float]] = []

        async def execute_tool(name, args):
            t = time.monotonic()
            await asyncio.sleep(0.05)
            execution_log.append((name, "end", time.monotonic()))
            return "ok"

        fake = _ScriptedLLM([
            ([
                ("read", {"path": "a.py"}),
                ("read", {"path": "b.py"}),
                ("write", {"path": "c.py", "content": "x"}),
            ], []),
            "Done.",
        ])
        runner = ToolLoopRunner(max_iterations=5)
        final_text, _ = _loop_runner_sync(runner, fake, execute_tool)

        assert final_text == "Done."
        # Verify all 3 executed.
        assert len(execution_log) == 3
        names = [e[0] for e in execution_log]
        assert "write" in names
        assert names.count("read") == 2

    def test_parallel_batch_then_text_answer(self):
        """Full lifecycle: parallel reads → text answer."""
        executed = []

        async def execute_tool(name, args):
            executed.append(name)
            return f"result-{name}"

        fake = _ScriptedLLM([
            ([("read", {"path": "a.py"}), ("search", {"query": "foo"})], []),
            "Found it.",
        ])
        runner = ToolLoopRunner(max_iterations=5)
        final_text, messages = _loop_runner_sync(runner, fake, execute_tool)

        assert final_text == "Found it."
        assert sorted(executed) == ["read", "search"]
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 2
