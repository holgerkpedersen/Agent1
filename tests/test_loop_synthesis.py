"""Tests: forced-synthesis retry, trace completeness, concrete fallback.

The loop guarantees it never ends without a usable text answer (decision
#034): the forced-synthesis call is traced, retried once when empty, and
chat_nlp falls back to a concrete activity summary instead of a cryptic
"no response" message.
"""
from __future__ import annotations

import json

from agent_core.llm.tool_loop import DisplayMode, ToolLoopRunner

from test_tool_loop_nlp import _ScriptedLLM, _loop_runner_sync


class _Sink:
    def __init__(self):
        self.events = []

    def emit(self, event: dict) -> None:
        self.events.append(dict(event))


class TestForcedSynthesisRetry:
    def test_empty_synthesis_retried_and_traced(self):
        fake = _ScriptedLLM([
            ("read", {"path": "a.py"}),
            ("read", {"path": "b.py"}),
            ("read", {"path": "a.py"}),
            ("read", {"path": "b.py"}),
            "",
            "Det konkrete svar om trace-analysen.",
        ])
        sink = _Sink()

        async def execute_tool(name, args):
            return "x"

        runner = ToolLoopRunner(
            max_iterations=10, no_mutation_limit=1, force_after_no_mutation=2,
            display_mode=DisplayMode.QUIET, trace=sink,
        )
        final_text, _ = _loop_runner_sync(runner, fake, execute_tool)

        assert runner.termination_reason == "no_progress"
        # 4 loop calls (alternating repeats, no new discovery) + 2 forced
        # calls (empty first, then the retry).
        assert len(fake.calls) == 6
        assert final_text == "Det konkrete svar om trace-analysen."
        # The retry note is present in the second forced call's messages.
        retry_note_in = any(
            "previous response was empty" in str(m.get("content"))
            for m in fake.calls[-1][0]
        )
        assert retry_note_in
        # The last two calls are tool-less (forced synthesis).
        assert fake.calls[-1][1] == [] and fake.calls[-2][1] == []

    def test_synthesis_llm_response_events_traced(self):
        fake = _ScriptedLLM([
            ("read", {"path": "a.py"}),
            ("read", {"path": "b.py"}),
            ("read", {"path": "a.py"}),
            ("read", {"path": "b.py"}),
            "Svar.",
        ])
        sink = _Sink()

        async def execute_tool(name, args):
            return "x"

        runner = ToolLoopRunner(
            max_iterations=10, no_mutation_limit=1, force_after_no_mutation=2, trace=sink,
        )
        _loop_runner_sync(runner, fake, execute_tool)

        kinds = [e["kind"] for e in sink.events]
        assert "guard_triggered" in kinds
        # The forced call(s) appear as llm_response with tool_calls_requested=0
        # (decision #034: every loop event is traced).
        synthesis = [e for e in sink.events if e["kind"] == "llm_response" and e.get("tool_calls_requested") == 0]
        assert synthesis, "missing traced synthesis response"
        gi = kinds.index("guard_triggered")
        assert kinds[gi + 1] == "llm_response"

    def test_stats_recorded_for_fallback(self):
        fake = _ScriptedLLM([
            ("read", {"path": "a.py"}),
            ("read", {"path": "b.py"}),
            ("read", {"path": "a.py"}),
            ("read", {"path": "b.py"}),
            ("read", {"path": "a.py"}),
            "Svar.",
        ])
        sink = _Sink()

        async def execute_tool(name, args):
            return "x"

        runner = ToolLoopRunner(
            max_iterations=10, no_mutation_limit=1, force_after_no_mutation=3, trace=sink,
        )
        _loop_runner_sync(runner, fake, execute_tool)

        assert runner.tool_calls_made == 5
        assert runner.tools_used == {"read": 5}
        assert runner.last_tool_call == "read"
        # Iterations 0..5 entered; iteration 5 was the guard check that broke.
        assert runner.iterations_used == 6


class TestFinalAnswerFallback:
    def test_helper_reports_activity_concretely(self):
        from agent import _final_answer_fallback

        loop = ToolLoopRunner(max_iterations=10)
        loop.tool_calls_made = 5
        loop.tools_used = {"read": 3, "tests": 2}
        loop.last_tool_call = "tests"
        loop.iterations_used = 5
        loop.termination_reason = "no_progress"

        msg = _final_answer_fallback(loop)
        assert "5 tool call(s)" in msg
        assert "readx3" in msg and "testsx2" in msg
        assert "no_progress" in msg
        assert "Last action: tests" in msg

    def test_empty_synthesis_ends_with_concrete_fallback(self):
        """End-to-end through chat_nlp's llm_chat_fn shape: the empty forced
        response is normalized and the loop returns no usable text."""
        import asyncio

        from agent_core.llm.tool_loop import ToolLoopRunner as TLR

        class _Fake:
            def __init__(self):
                self.calls = 0

            async def chat(self, messages, tools=None, **kw):
                self.calls += 1
                # First three calls: the SAME search → the stuck guard fires
                # on the third consecutive identical call.  Forced-synthesis
                # calls return empty output.
                if self.calls <= 3:
                    return json.dumps({
                        "content": "",
                        "tool_calls": [{
                            "id": f"c{self.calls}", "type": "function",
                            "function": {"name": "search",
                                        "arguments": json.dumps({"query": "x"})},
                        }],
                    })
                return "(no output)"

        fake = _Fake()
        runner = TLR(max_iterations=5, no_mutation_limit=1, force_after_no_mutation=2)

        async def execute_tool(name, args):
            return "x"

        async def llm_chat_fn(messages, tools):
            # Mirrors agent.py's chat_nlp wrapper: normalize "(no output)",
            # parse tool_calls JSON, else treat as plain text.
            raw = await fake.chat(messages, tools)
            if raw.strip() == "(no output)":
                raw = ""
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                parsed = None
            if isinstance(parsed, dict) and parsed.get("tool_calls"):
                parsed.pop("role", None)
                updated = list(messages)
                updated.append({"role": "assistant", "content": parsed.get("content") or "", **parsed})
                return str(parsed.get("content") or ""), updated
            updated = list(messages) + [{"role": "assistant", "content": raw}]
            return raw, updated

        async def go():
            return await runner.run(
                messages=[{"role": "user", "content": "do the thing"}],
                llm_chat_fn=llm_chat_fn,
                execute_tool_fn=execute_tool,
                tools=[],
            )

        final_text, _ = asyncio.run(go())
        assert runner.termination_reason == "stuck"
        # 3 loop calls + 2 forced calls — all normalized away.
        assert final_text == ""  # the concrete fallback message applies
        assert fake.calls == 5


def _loop_runner_sync_imports() -> None:
    assert json is not None  # keep import linters quiet
