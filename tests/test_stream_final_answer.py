"""Tests: the final answer streams instead of blocking silently (plan #9).

``chat_stream`` exists on every provider (with model-eviction recovery), but
``chat_nlp`` only ever called the blocking ``chat`` — so the tool-less
forced-synthesis call, which is exactly the long-context "produce the final
answer now" moment after a cap/stuck verdict, showed the user nothing until it
finished.

Contract under test:
1. when a ``stream_fn`` is supplied, the tool-less forced-synthesis call is
   served by it (the blocking call is NOT used for that answer);
2. the runner reports ``final_answer_streamed`` so the caller can skip
   re-printing the answer that already appeared live;
3. streaming FAILS OPEN: an exception, an empty string, or ``(no output)``
   falls back to the blocking final call — a broken stream must never swallow
   the answer (the loop's "always end with a usable answer" guarantee);
4. with no ``stream_fn`` (the default) behaviour is byte-identical: the
   blocking call serves the final answer and nothing is marked streamed.
"""
from __future__ import annotations

from agent_core.llm.tool_loop import DisplayMode, ToolLoopRunner

from test_tool_loop_nlp import _ScriptedLLM, _loop_runner_sync

#: Four alternating repeated reads: no new discovery, so the no-progress guard
#: fires and the run ends with a forced-synthesis (tool-less) final answer.
_NO_PROGRESS_SCRIPT = [
    ("read", {"path": "a.py"}),
    ("read", {"path": "b.py"}),
    ("read", {"path": "a.py"}),
    ("read", {"path": "b.py"}),
]


async def _execute_tool(name, args):
    return "x"


def _runner(**kwargs) -> ToolLoopRunner:
    return ToolLoopRunner(
        max_iterations=10, no_mutation_limit=1, force_after_no_mutation=2,
        display_mode=DisplayMode.QUIET, **kwargs,
    )


class TestForcedSynthesisStreams:
    def test_stream_fn_serves_the_final_answer_instead_of_blocking_call(self):
        fake = _ScriptedLLM(_NO_PROGRESS_SCRIPT + ["", "Blocking answer."])
        streamed: list[list] = []

        async def stream_fn(messages):
            streamed.append(list(messages))
            return "Streamet slutsvar."

        runner = _runner()
        final_text, _ = _loop_runner_sync(
            runner, fake, _execute_tool, stream_fn=stream_fn)

        assert final_text == "Streamet slutsvar."
        assert runner.final_answer_streamed is True
        # The stream served the final call, so the blocking path was never
        # asked for it: only the four tool iterations went through the LLM.
        assert len(fake.calls) == len(_NO_PROGRESS_SCRIPT)
        # The streamed call is the forced-synthesis one: its last message is the
        # no-progress steering note (not a tool result).
        assert len(streamed) == 1
        assert "final answer" in streamed[0][-1]["content"]

    def test_streamed_answer_is_not_reprinted_by_the_loop(self):
        """The stream already showed the text; the runner must not append a
        second copy of it as an intermediate text part."""
        fake = _ScriptedLLM(_NO_PROGRESS_SCRIPT)

        async def stream_fn(messages):
            return "Streamet svar."

        runner = _runner()
        final_text, messages = _loop_runner_sync(
            runner, fake, _execute_tool, stream_fn=stream_fn)

        assert final_text == "Streamet svar."
        # Exactly one assistant message carries the answer.
        answers = [
            m for m in messages
            if m.get("role") == "assistant"
            and "Streamet svar." in str(m.get("content"))
        ]
        assert len(answers) == 1


class TestStreamingFailsOpen:
    def test_stream_exception_falls_back_to_blocking_final_call(self):
        fake = _ScriptedLLM(_NO_PROGRESS_SCRIPT + ["Blocking answer."])

        async def stream_fn(messages):
            raise RuntimeError("stream exploded")

        runner = _runner()
        final_text, _ = _loop_runner_sync(
            runner, fake, _execute_tool, stream_fn=stream_fn)

        assert final_text == "Blocking answer."
        assert runner.final_answer_streamed is False

    def test_empty_stream_falls_back_and_keeps_the_retry_path(self):
        """An empty stream is not an answer: the blocking call must run, and
        the loop's existing empty-response retry must still be available."""
        fake = _ScriptedLLM(_NO_PROGRESS_SCRIPT + ["", "Blocking retry."])

        async def stream_fn(messages):
            return ""

        runner = _runner()
        final_text, _ = _loop_runner_sync(
            runner, fake, _execute_tool, stream_fn=stream_fn)

        assert final_text == "Blocking retry."
        assert runner.final_answer_streamed is False

    def test_no_output_stream_falls_back(self):
        fake = _ScriptedLLM(_NO_PROGRESS_SCRIPT + ["Blocking answer."])

        async def stream_fn(messages):
            return "(no output)"

        runner = _runner()
        final_text, _ = _loop_runner_sync(
            runner, fake, _execute_tool, stream_fn=stream_fn)

        assert final_text == "Blocking answer."
        assert runner.final_answer_streamed is False


class TestNoStreamFnIsUnchanged:
    def test_default_uses_blocking_call_and_marks_nothing_streamed(self):
        fake = _ScriptedLLM(_NO_PROGRESS_SCRIPT + ["Blocking answer."])

        runner = _runner()
        final_text, _ = _loop_runner_sync(runner, fake, _execute_tool)

        assert final_text == "Blocking answer."
        assert runner.final_answer_streamed is False


class TestProviderErrorSentinelIsNotAnAnswer:
    """Providers RETURN failure sentinels instead of raising, and
    ``FailoverProvider.chat_stream`` delegates to the FIRST provider only — so
    a streamed ``[Error: ...]`` must never be accepted as the answer, or the
    user sees an outage as content AND the blocking call that would have failed
    over is skipped."""

    def test_streamed_error_sentinel_falls_back_to_blocking_call(self):
        fake = _ScriptedLLM(_NO_PROGRESS_SCRIPT + ["Real answer."])

        async def stream_fn(messages):
            return "[Error: llama-server connection refused]"

        runner = _runner()
        final_text, _ = _loop_runner_sync(
            runner, fake, _execute_tool, stream_fn=stream_fn)

        assert final_text == "Real answer."
        assert runner.final_answer_streamed is False

    def test_streamed_lmstudio_stream_error_sentinel_falls_back(self):
        fake = _ScriptedLLM(_NO_PROGRESS_SCRIPT + ["Real answer."])

        async def stream_fn(messages):
            return "[LM Studio stream error: HTTP Error 400: Bad Request]"

        runner = _runner()
        final_text, _ = _loop_runner_sync(
            runner, fake, _execute_tool, stream_fn=stream_fn)

        assert final_text == "Real answer."
        assert runner.final_answer_streamed is False

    def test_streamed_llama_server_stream_error_sentinel_falls_back(self):
        fake = _ScriptedLLM(_NO_PROGRESS_SCRIPT + ["Real answer."])

        async def stream_fn(messages):
            return "[llama-server stream error: timed out]"

        runner = _runner()
        final_text, _ = _loop_runner_sync(
            runner, fake, _execute_tool, stream_fn=stream_fn)

        assert final_text == "Real answer."
        assert runner.final_answer_streamed is False

    def test_answer_merely_mentioning_error_bracket_is_still_an_answer(self):
        """The sentinel check is anchored at the start: a real answer that
        talks about ``[Error:`` (e.g. writing error handling) is content."""
        prose = "Wrap it so it returns [Error: ...] on failure."
        fake = _ScriptedLLM(_NO_PROGRESS_SCRIPT)

        async def stream_fn(messages):
            return prose

        runner = _runner()
        final_text, _ = _loop_runner_sync(
            runner, fake, _execute_tool, stream_fn=stream_fn)

        assert final_text == prose
        assert runner.final_answer_streamed is True
