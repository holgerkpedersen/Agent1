"""Regression tests for read-only tool fan-out in ToolLoopRunner.

Read-only, non-duplicate tool calls in a single model batch must execute
concurrently (one wall-clock wait, not N), while mutating calls and
``delegate`` still run sequentially and all existing guards stay intact.
"""
import asyncio
import time

from agent_core.llm.tool_loop import ToolLoopRunner, READONLY_FANOUT_TOOLS
from test_tool_loop_nlp import _ScriptedLLM, _loop_runner_sync, _make_llm_chat_fn


def _runner(fake, execute_tool, **kwargs):
    runner = ToolLoopRunner(max_iterations=10)
    return _loop_runner_sync(runner, fake, execute_tool, **kwargs)


def test_readonly_batch_runs_concurrently():
    """Two read-only calls in one batch overlap (total ~= one sleep, not two)."""
    order = []

    async def execute_tool(name, args):
        start = time.monotonic()
        await asyncio.sleep(0.1)
        end = time.monotonic()
        order.append((name, start, end))
        return f"result-of-{name}"

    fake = _ScriptedLLM([
        ([("read", {"path": "a.py"}), ("search", {"query": "x"})], None),
        "Done.",
    ])
    _runner(fake, execute_tool)
    assert len(order) == 2
    # The second call started before the first finished -> genuine overlap.
    assert order[1][1] < order[0][2]


def test_tool_messages_keep_tool_call_id_order():
    """Tool messages are appended in assistant tool_call_id order, not
    gather-completion order."""
    async def execute_tool(name, args):
        # Make the first call artificially slow so gather finishes it last.
        if name == "read":
            await asyncio.sleep(0.05)
        return f"result-of-{name}"

    fake = _ScriptedLLM([
        ([("read", {"path": "a.py"}), ("search", {"query": "x"})], None),
        "Done.",
    ])
    final_text, messages = _runner(fake, execute_tool)
    assert final_text == "Done."
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_0", "call_1"]


def test_mutating_calls_run_after_reads():
    """In a [read, write, read] batch the write runs strictly after the two
    reads finish; the reads overlap each other."""
    order = []

    async def execute_tool(name, args):
        start = time.monotonic()
        await asyncio.sleep(0.05)
        end = time.monotonic()
        order.append((name, start, end))
        return f"result-of-{name}"

    fake = _ScriptedLLM([
        ([("read", {"path": "a.py"}), ("write", {"path": "b.py", "content": "x"}),
          ("read", {"path": "c.py"})], None),
        "Done.",
    ])
    _runner(fake, execute_tool)
    names = [o[0] for o in order]
    assert names.count("read") == 2 and names.count("write") == 1
    read_ends = [o[2] for o in order if o[0] == "read"]
    write = next(o for o in order if o[0] == "write")
    # Write starts only after both reads completed.
    assert write[1] >= max(read_ends)
    # The two reads overlapped.
    assert order[1][1] < order[0][2]


def test_delegate_is_not_fanned_out():
    """delegate is excluded from READONLY_FANOUT_TOOLS and therefore runs
    sequentially (no concurrency assumption)."""
    assert "delegate" not in READONLY_FANOUT_TOOLS


def test_single_call_batch_is_sequential_noop():
    """A batch with a single call falls back to sequential (byte-identical)."""
    seen = []

    async def execute_tool(name, args):
        seen.append(name)
        return f"result-of-{name}"

    fake = _ScriptedLLM([("read", {"path": "a.py"}), "Done."])
    final_text, messages = _runner(fake, execute_tool)
    assert final_text == "Done."
    assert seen == ["read"]
    assert sum(1 for m in messages if m["role"] == "tool") == 1


def test_duplicate_third_call_still_stops_loop():
    """Repeated identical calls still trigger stuck synthesis under fan-out."""
    fake = _ScriptedLLM([
        ([("list_files", {"path": "."}), ("list_files", {"path": "."}),
          ("list_files", {"path": "."})], None),
        "Final answer.",
    ])
    executed = []

    async def execute_tool(name, args):
        executed.append(name)
        return "entries"

    final_text, _ = _runner(fake, execute_tool)
    assert final_text == "Final answer."
    # Only the first of the identical calls actually executes.
    assert executed == ["list_files"]


def test_plan_mode_batch_fans_out_all_readonly():
    """A plan-mode batch of read/search/references all execute concurrently."""
    order = []

    async def execute_tool(name, args):
        start = time.monotonic()
        await asyncio.sleep(0.05)
        order.append((name, start))
        return f"result-of-{name}"

    fake = _ScriptedLLM([
        ([("read", {"path": "a.py"}), ("search", {"query": "x"}),
          ("references", {"symbol": "foo"})], None),
        "Done.",
    ])
    _runner(fake, execute_tool)
    assert len(order) == 3
    # All three started within a tiny window of each other (concurrent).
    starts = [o[1] for o in order]
    assert max(starts) - min(starts) < 0.05
