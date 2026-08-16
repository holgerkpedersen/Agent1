"""Phase 0 tests: trace capture from a fake ToolLoopRunner run.

Covers the spec's acceptance criteria:
- a fake run emits a valid, complete trace (all event kinds, loop_end present)
- every guard firing is recorded with its injected note
- tracing is non-invasive: byte-identical behaviour with and without a sink
- AGENT_NO_TRACE env toggle disables trace writing
"""
from __future__ import annotations

import asyncio
import json

import pytest

from agent_core.llm.tool_loop import ToolLoopRunner
from harnessfix.reader import TraceValidationError, read_trace, task_id_of
from harnessfix.tracing import (
    GUARD_DEADLINE,
    KIND_GUARD_TRIGGERED,
    KIND_LLM_RESPONSE,
    KIND_LOOP_END,
    KIND_STEP_START,
    KIND_TOOL_CALL,
    KIND_TOOL_ERROR,
    KIND_TOOL_RESULT,
    TraceWriter,
    trace_enabled,
)


def _fake_llm(script):
    """script: sequence of ("text", text) | ("tools", [{name, arguments}]) steps."""
    calls: list[list[dict]] = []

    async def llm_chat_fn(messages, tools):
        calls.append([dict(m) for m in messages])
        step = script[min(len(calls) - 1, len(script) - 1)]
        if step[0] == "text":
            return step[1], messages + [{"role": "assistant", "content": step[1]}]
        tool_calls = [
            {
                "id": f"call-{i}",
                "type": "function",
                "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
            }
            for i, tc in enumerate(step[1])
        ]
        updated = messages + [{"role": "assistant", "content": "", "tool_calls": tool_calls}]
        return "", updated

    return llm_chat_fn, calls


def _fake_executor():
    async def execute_tool_fn(name, args):
        if name == "run":
            raise ValueError("boom")
        return f"result-of-{name}:{args.get('path', '')}"

    return execute_tool_fn


def _run(script, *, trace, max_iterations=3):
    llm, calls = _fake_llm(script)
    loop = ToolLoopRunner(
        max_iterations=max_iterations,
        deadline_window=1,
        display_mode="quiet",
        trace=trace,
    )
    final_text, final_messages = asyncio.run(
        loop.run(
            messages=[{"role": "user", "content": "do it"}],
            llm_chat_fn=llm,
            execute_tool_fn=_fake_executor(),
            tools=[{"type": "function", "function": {"name": "read"}}],
        )
    )
    return final_text, final_messages, calls


SCRIPT = [
    ("tools", [{"name": "read", "arguments": {"path": "a.txt"}}]),
    ("tools", [{"name": "run", "arguments": {"command": "x"}}]),
    ("text", "done"),
]


def test_fake_run_emits_valid_complete_trace(tmp_path):
    writer = TraceWriter(task_id="t1", directory=tmp_path)
    final_text, _messages, _calls = _run(SCRIPT, trace=writer)

    events = read_trace(writer.path)
    assert final_text == "done"
    assert writer.path.is_file()
    assert task_id_of(writer.path) == "t1"

    kinds = {e["kind"] for e in events}
    assert kinds == {
        KIND_STEP_START,
        KIND_LLM_RESPONSE,
        KIND_TOOL_CALL,
        KIND_TOOL_RESULT,
        KIND_TOOL_ERROR,
        KIND_GUARD_TRIGGERED,
        KIND_LOOP_END,
    }
    # Every event carries a valid layer facet.
    for event in events:
        assert event["layer"]
        assert event["correlation_id"] is not None

    assert events[0]["kind"] == KIND_STEP_START
    assert events[-1]["kind"] == KIND_LOOP_END
    assert events[-1]["outcome"] == "completed"
    assert events[-1]["termination_reason"] == "answer"

    # Guard firing recorded: the deadline warning fired at the final iteration.
    guards = [e for e in events if e["kind"] == KIND_GUARD_TRIGGERED]
    assert any(e["guard"] == GUARD_DEADLINE for e in guards)
    assert all(e["note"] for e in guards)

    # Tool error recorded with its exception type.
    errors = [e for e in events if e["kind"] == KIND_TOOL_ERROR]
    assert errors and errors[0]["exception"] == "ValueError"
    assert "boom" in errors[0]["message"]


def test_tracing_is_non_invasive(tmp_path):
    """Same script, with and without a sink -> identical results and messages."""
    writer = TraceWriter(task_id="t2", directory=tmp_path)
    traced_text, traced_msgs, traced_calls = _run(SCRIPT, trace=writer)
    plain_text, plain_msgs, plain_calls = _run(SCRIPT, trace=None)

    assert traced_text == plain_text == "done"
    assert traced_msgs == plain_msgs
    assert traced_calls == plain_calls  # identical LLM request payloads


def test_no_trace_no_file(tmp_path):
    _run(SCRIPT, trace=None)
    assert list(tmp_path.glob("*.jsonl")) == []


def test_trace_enabled_env_toggle(monkeypatch):
    monkeypatch.setenv("AGENT_NO_TRACE", "1")
    assert trace_enabled() is False
    monkeypatch.setenv("AGENT_NO_TRACE", "0")
    assert trace_enabled() is True
    monkeypatch.delenv("AGENT_NO_TRACE")
    assert trace_enabled() is True


def test_reader_rejects_malformed_trace(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"kind": "step_start", "layer": "lifecycle"}\nnot json\n', encoding="utf-8")
    with pytest.raises(TraceValidationError):
        read_trace(bad)

    no_layer = tmp_path / "no_layer.jsonl"
    no_layer.write_text('{"kind": "step_start"}\n', encoding="utf-8")
    with pytest.raises(TraceValidationError):
        read_trace(no_layer)
