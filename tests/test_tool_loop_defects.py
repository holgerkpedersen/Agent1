"""Regression tests for tool-loop defects found in the July 2026 tool-calling audit.

Each test targets one verified defect (all fixed in this revision):

A. Multi-line tool arguments wiped to ``{}`` — the whole-text newline replace
   in agent.llm_chat_fn mangled pretty-printed payloads; repair now escapes
   control characters inside string literals only (agent._repair_tool_arguments).
B. Out-of-order tool messages for mixed duplicate/pending batches — dup results
   handled during pre-check were appended inline, landing BEFORE earlier pending
   calls' results; they are now merged in original announcement order.
C. In-loop compaction dropped legitimate "CONTEXT NOTE:" content — the note
   filter matched a content prefix; notes are now tagged with LOOP_NOTE_TAG_KEY
   (stripped by sanitize_message_roles at the provider payload boundary).
D. Path-miss recovery leaked the hidden list_files' effect into the next tool
   event's affected_files — effects are discarded after each recovery call.
E. Runner state (stats, termination reason) leaked across run() calls on a
   reused runner — per-run state resets at the start of every run.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from agent_core.constants import LOOP_NOTE_TAG_KEY
from agent_core.llm.lmstudio import sanitize_message_roles
from agent_core.llm.tool_loop import (
    _COMPACT_NOTE_TAG,
    _compact_messages,
    ToolLoopRunner,
)
from harnessfix.tracing import KIND_TOOL_RESULT

import agent as agent_module


# ---------------------------------------------------------------------------
# Helpers (same scripting convention as tests/test_tool_loop_fanout.py)
# ---------------------------------------------------------------------------

class _ScriptedLLM:
    """Fake LLM returning a fixed sequence of responses.

    Parallel batch: ``([(tool, args), ...])``  Single call: ``(tool, args)``
    Text answer:    plain string.
    """

    def __init__(self, script):
        self.script = list(script)

    async def chat(self, messages, tools=None, **kwargs):
        step = self.script.pop(0)
        # Parallel batch.  Accept BOTH spellings: the bare list documented
        # above and used by these scripts (``[(tool, args), ...]``), and the
        # ``([(tool, args), ...], None)`` tuple used by test_tool_loop_fanout.py.
        # Without the bare-list branch the step fell through to ``return step``,
        # handing a *list* back as ``response_text`` and crashing the loop's
        # final-text scan with ``'list' object has no attribute 'strip'``.
        batch = None
        if isinstance(step, list):
            batch = step
        elif isinstance(step, tuple) and len(step) == 2 and isinstance(step[0], list):
            batch = step[0]
        if batch is not None:
            tool_calls = []
            for i, item in enumerate(batch):
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
            msg = {"role": "assistant", "content": "", "tool_calls": tool_calls}
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
            updated.append(
                {"role": "assistant",
                 "content": parsed.get("content") or "", **parsed})
            return str(parsed.get("content") or ""), updated
        updated = list(messages)
        updated.append({"role": "assistant", "content": raw})
        return raw, updated
    return llm_chat_fn


def _run_sync(runner, fake_llm, execute_tool, **kwargs):
    return asyncio.run(runner.run(
        messages=[{"role": "user", "content": "go"}],
        llm_chat_fn=_make_llm_chat_fn(fake_llm),
        execute_tool_fn=execute_tool,
        tools=[],  # tools not needed for a scripted LLM
        **kwargs,
    ))


# ---------------------------------------------------------------------------
# A. Multi-line tool arguments must survive the repair path (agent.py)
# ---------------------------------------------------------------------------

def test_repair_tool_arguments_multiline_string_value():
    """Pretty-printed payload with an unescaped newline INSIDE a string value —
    exactly the case the old whole-text replace wiped to ``{}``."""
    raw = '{\n  "path": "/tmp/x.py",\n  "note": "first line\nsecond line"\n}'
    assert agent_module._repair_tool_arguments(raw) == {
        "path": "/tmp/x.py",
        "note": "first line\nsecond line",
    }


def test_repair_tool_arguments_valid_json_unchanged():
    assert agent_module._repair_tool_arguments(
        '{"a": 1, "b": [2]}') == {"a": 1, "b": [2]}


def test_repair_tool_arguments_single_line_raw_newline_still_works():
    """The case the old whole-text replace could fix must keep working."""
    assert agent_module._repair_tool_arguments(
        '{"cmd": "echo hi\nexit"}') == {"cmd": "echo hi\nexit"}


def test_repair_tool_arguments_tab_and_cr_inside_string():
    raw = '{"q": "a\tb\rc"}'
    assert agent_module._repair_tool_arguments(raw) == {"q": "a\tb\rc"}


def test_repair_tool_arguments_unrepairable_degrades_to_empty_dict():
    assert agent_module._repair_tool_arguments("not json at all") == {}
    assert agent_module._repair_tool_arguments("") == {}
    assert agent_module._repair_tool_arguments("   ") == {}
    # Valid JSON that is not an object is not a tool-args payload — degrade.
    assert agent_module._repair_tool_arguments("[1, 2]") == {}


def test_repair_tool_arguments_escaped_quotes_not_misread():
    """The state machine must treat ``\\"`` (an escaped quote) as INSIDE the
    string and still escape a later raw newline in the same value."""
    raw = '{"s": "say \\"hi\\"", "t": "a\nb"}'
    assert agent_module._repair_tool_arguments(raw) == {
        "s": 'say "hi"',
        "t": "a\nb",
    }


# ---------------------------------------------------------------------------
# B. Mixed duplicate/pending batches keep announcement order (tool_loop.py)
# ---------------------------------------------------------------------------

def test_duplicate_and_pending_batch_keep_announcement_order():
    """When an assistant batch mixes a pending call with a duplicate of the
    previous batch, every tool message must still follow its assistant message
    in announced order — the inline pre-check append used to put the dup result
    FIRST (rejected by strict chat templates)."""
    executed: list[tuple[str, dict]] = []

    async def execute(name, args):
        executed.append((name, dict(args)))
        if name == "read":
            return "file content for /tmp/x.py"
        return f"search results: {args.get('query')}"

    llm = _ScriptedLLM([
        [("search", {"query": "alpha"})],  # batch 1
        # batch 2: read is pending (executes via fan-out), search duplicates
        # the previous batch (handled during pre-check, never re-executed).
        [("read", {"path": "/tmp/x.py"}), ("search", {"query": "alpha"})],
        "done",
    ])
    runner = ToolLoopRunner(max_iterations=8, display_mode="quiet")
    final_text, messages = _run_sync(runner, llm, execute)

    assert final_text == "done"
    batch2 = [
        m for m in messages
        if m.get("role") == "assistant" and len(m.get("tool_calls", [])) == 2
    ]
    assert len(batch2) == 1
    announced = [tc["id"] for tc in batch2[0]["tool_calls"]]
    after = messages[messages.index(batch2[0]) + 1:]
    tool_ids = [m.get("tool_call_id") for m in after if m.get("role") == "tool"]
    assert tool_ids[:2] == announced, (
        f"tool messages out of announcement order: {tool_ids} vs {announced}"
    )
    dup_msg = next(m for m in after if m.get("tool_call_id") == announced[1])
    assert "This exact call has now been executed" in str(dup_msg["content"])
    # The duplicate search was NOT re-executed; read ran exactly once.
    assert sum(1 for n, _a in executed if n == "read") == 1
    assert sum(1 for n, _a in executed if n == "search") == 1


# ---------------------------------------------------------------------------
# C. Compaction must not drop legitimate "CONTEXT NOTE:" content (tool_loop.py)
# ---------------------------------------------------------------------------

def test_compact_keeps_legit_context_note_content():
    """A real user prompt that merely STARTS with 'CONTEXT NOTE:' must survive
    in-loop compaction — only tagged agent notes are replaced."""
    legit = {
        "role": "user",
        "content": (
            "CONTEXT NOTE: investigate the failing build first, then fix it."
        ),
    }
    msgs = [{"role": "system", "content": "sys prompt"}]
    msgs.append(legit)
    for i in range(10):
        msgs.append({"role": "assistant", "content": f"step {i}"})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "ok"})

    out, note = _compact_messages(msgs, 8, 4_000, "old steps dropped")
    assert note is not None and note.startswith("CONTEXT NOTE:")
    assert legit in out
    tagged = [m for m in out if m.get(LOOP_NOTE_TAG_KEY) == _COMPACT_NOTE_TAG]
    assert len(tagged) == 1

    # A second compaction REPLACES the old tag (which sits inside the kept
    # window here, so positional dropping cannot explain its absence).
    more = list(out) + [
        {"role": "assistant", "content": f"later {i}"} for i in range(4)
    ]
    out2, note2 = _compact_messages(more, 8, 4_000, "still dropping")
    assert note2 is not None
    assert legit in out2
    tagged2 = [m for m in out2 if m.get(LOOP_NOTE_TAG_KEY) == _COMPACT_NOTE_TAG]
    assert len(tagged2) == 1


def test_compact_note_tag_never_reaches_provider_payload():
    """The tag is loop-internal: sanitize_message_roles must strip it before
    the payload reaches any provider (unknown fields are rejected)."""
    msgs = [{"role": "system", "content": "s"}]
    for i in range(12):
        msgs.append({"role": "assistant", "content": f"a{i}"})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "ok"})
    out, _note = _compact_messages(msgs, 8, 4_000, "dropped")
    assert any(LOOP_NOTE_TAG_KEY in m for m in out)  # tag present internally...
    payload = sanitize_message_roles(out)
    assert all(LOOP_NOTE_TAG_KEY not in m for m in payload)  # ...never externally


# ---------------------------------------------------------------------------
# D. Path-miss recovery must not leak listing effects (tool_loop.py)
# ---------------------------------------------------------------------------

class _MemorySink:
    """In-memory TraceSink collecting every emitted event."""

    def __init__(self):
        self.events: list[dict[str, Any]] = []

    def emit(self, event: dict[str, Any]) -> None:
        self.events.append(event)


def test_path_miss_recovery_effects_discarded():
    """The hidden list_files that path-miss recovery runs records its own
    effect in the shared buffer; it must be discarded so the parent dir is
    attributed to NO tool event (it was only listed, never touched)."""
    buf: list[str] = []
    listing_dirs: list[str] = []

    async def execute(name, args):
        if name == "read":
            buf.append("/tmp/missing.txt")  # simulate _note_effect on the target
            return "File not found: /tmp/missing.txt"
        if name == "list_files":
            listing_dirs.append(str(args.get("path")))
            buf.append(str(args.get("path")))  # the hidden call records itself
            return "x.py\ny.py"
        if name == "write":
            buf.append("/tmp/out.txt")
            return "wrote /tmp/out.txt"
        raise AssertionError(f"unexpected tool {name}")

    def effects_fn(tool_name, args):  # mirrors agent._take_trace_effects
        drained = list(buf)
        buf.clear()
        return drained

    llm = _ScriptedLLM([
        [("read", {"path": "/tmp/missing.txt"})],  # path miss -> hidden listing
        [("write", {"path": "/tmp/out.txt"})],     # the next real call
        "done",
    ])
    sink = _MemorySink()
    runner = ToolLoopRunner(max_iterations=8, display_mode="quiet")
    final_text, _messages = _run_sync(
        runner, llm, execute, trace=sink, effects_fn=effects_fn)

    assert final_text == "done"
    assert listing_dirs, "expected the recovery to list a parent directory"
    results = [e for e in sink.events if e.get("kind") == KIND_TOOL_RESULT]
    read_ev = next(e for e in results if e.get("tool") == "read")
    write_ev = next(e for e in results if e.get("tool") == "write")
    assert list(read_ev["affected_files"]) == ["/tmp/missing.txt"]
    # Without the fix, the recovery's listing dir would leak into this event.
    assert list(write_ev["affected_files"]) == ["/tmp/out.txt"]
    all_affected = {p for e in results for p in e.get("affected_files", [])}
    assert not (set(listing_dirs) & all_affected), (
        f"recovery listing dir leaked into a tool event: {all_affected}"
    )


# ---------------------------------------------------------------------------
# E. Reused runners must not leak run state across run() calls (tool_loop.py)
# ---------------------------------------------------------------------------

def test_runner_state_resets_between_runs():
    """Stats, termination reason and the mutated-file record describe THIS run —
    a second run() on the same runner instance must start from a clean slate."""

    async def execute(name, args):
        return "ok"

    runner = ToolLoopRunner(max_iterations=8, display_mode="quiet")
    t1, _m1 = _run_sync(
        runner, _ScriptedLLM([("read", {"path": "/a.py"}), "done"]), execute)
    assert t1 == "done"
    assert runner.tool_calls_made == 1
    assert runner.termination_reason == "answer"

    # Second run on the SAME runner instance: nothing may accumulate.
    llm2 = _ScriptedLLM(
        [("read", {"path": "/b.py"}), ("search", {"query": "q"}), "done"])
    t2, _m2 = _run_sync(runner, llm2, execute)
    assert t2 == "done"
    # Without the reset this would be 3 (1 from run 1 + 2 from run 2).
    assert runner.tool_calls_made == 2
    assert runner.tools_used == {"read": 1, "search": 1}
    assert runner.termination_reason == "answer"
    assert not runner._mutated_files
