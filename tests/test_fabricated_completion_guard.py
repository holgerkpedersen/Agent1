"""Claim-vs-evidence guard: a turn whose every tool call failed must not end
with a fabricated "done".

History (2026-11-16 session): the user asked to continue the work.  The chat
model called ``run(pytest)``; the full-suite gate REFUSED it (no real source
change since the session baseline).  The very next answer was::

    Task complete.
    Queue is now empty and cleaned up; Git commit verified with pre-commit
    checks.
    - Commit hash: 8bcbcad on branch master.

One attempted call, zero evidence: the cited hash (8bcbcad) is a REAL commit
from an EARLIER session (14:44 the same day) reported as if this turn had
produced it — the turn itself only saw its pytest call refused.  The
heuristics (``_looks_incomplete``) saw nothing to flag — "Task complete" reads
as an answer — and the Jev delivery check voted DELIVERED (p(no)=0.33), so the
turn ended on the stale report.

The guard is deterministic (no model call): every tool call failed => the task
cannot be complete.  It fires when the answer claims success OR when the user
asked for an action and the answer hides that nothing worked, and it shares the
ONE post-answer nudge with the Jev delivery check, so an answer is retried at
most once.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from agent_core.config import AgentDisplayMode
from agent_core.turn_loop import (
    _EVIDENCE_NOTE,
    _acknowledges_failure,
    _action_request_without_evidence,
    _claims_completion,
    _run_chained_tool_loop,
)

FABRICATED = (
    "Task complete.\n"
    "Queue is now empty and cleaned up; Git commit verified with pre-commit "
    "checks.\n"
    "- Commit hash: 8bcbcad on branch master."
)
HONEST = (
    "The full-suite pytest gate refused the run: no source file changed since "
    "the session baseline. Nothing was verified this turn."
)


class _ScriptedLLM:
    """LLM that returns a fixed sequence of responses (text or tool calls)."""

    def __init__(self, script):
        self.script = list(script)
        self.requests: list[tuple[list, list | None]] = []

    async def chat(self, messages, tools=None, **kwargs):
        self.requests.append((list(messages), tools))
        step = self.script.pop(0)
        if isinstance(step, tuple):
            tool_name, args = step
            return json.dumps(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(args),
                            },
                        }
                    ],
                }
            )
        return step


class _FakeAgent:
    model_name = "test-model"
    workspace = "."
    session_id = ""
    mode = "build"

    def __init__(self, llm, tool_result):
        self.llm = llm
        self.tool_result = tool_result
        self.tool_calls: list[tuple[str, dict]] = []

    async def _execute_tool_call(self, name, args):
        self.tool_calls.append((name, args))
        return self.tool_result

    def _take_trace_effects(self):
        return []

    def _record_llm_experience(self, **kwargs):
        pass


REFUSAL = (
    "Error: full-suite pytest REFUSED - no real source file has changed since "
    "the session baseline"
)
SUCCESS = "On branch main\nnothing to commit, working tree clean"


def _run(agent, user_input="continue with the current work"):
    return asyncio.run(
        _run_chained_tool_loop(
            agent,  # type: ignore[arg-type]
            user_input=user_input,
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": user_input},
            ],
            display_mode=AgentDisplayMode.QUIET,
            seen_calls={},
        )
    )


class TestClaimDetector:
    def test_fabricated_answer_claims_completion(self) -> None:
        assert _claims_completion(FABRICATED) is True

    def test_plain_progress_narration_is_not_a_claim(self) -> None:
        assert _claims_completion("Checking impact before closing out.") is False

    def test_empty_text_is_not_a_claim(self) -> None:
        assert _claims_completion("") is False


class TestFailureAcknowledgement:
    def test_honest_report_acknowledges(self) -> None:
        assert _acknowledges_failure(HONEST) is True

    def test_fabricated_report_hides_the_refusal(self) -> None:
        assert _acknowledges_failure(FABRICATED) is False


class TestActionRequestWithoutEvidence:
    def _loop(self, made: int, failed: int):
        return SimpleNamespace(tool_calls_made=made, tool_calls_failed=failed)

    def test_imperative_request_with_only_failed_calls(self) -> None:
        assert (
            _action_request_without_evidence(
                "continue with the current work", self._loop(1, 1)
            )
            is True
        )

    def test_question_is_never_a_work_order(self) -> None:
        assert (
            _action_request_without_evidence(
                "how do I run the tests?", self._loop(1, 1)
            )
            is False
        )

    def test_no_tool_calls_is_not_this_guard(self) -> None:
        """A text-only answer is the delivery/completeness heuristics' job."""
        assert (
            _action_request_without_evidence(
                "continue with the current work", self._loop(0, 0)
            )
            is False
        )

    def test_once_something_worked_the_guard_is_silent(self) -> None:
        assert (
            _action_request_without_evidence(
                "continue with the current work", self._loop(2, 1)
            )
            is False
        )


class TestTurnLoopGuard:
    def test_fabricated_completion_triggers_one_honest_retry(self) -> None:
        llm = _ScriptedLLM(
            [
                ("run", {"command": "python -m pytest"}),
                FABRICATED,
                HONEST,
            ]
        )
        agent = _FakeAgent(llm, REFUSAL)
        final_text, messages, llm_error, loop = _run(agent)

        assert llm_error is None
        assert final_text == HONEST
        # One attempted call, and the loop recorded it as a failure.
        assert agent.tool_calls == [("run", {"command": "python -m pytest"})]
        assert loop.tool_calls_made == 0  # last run made no calls
        # Two runs: the fabricated answer, then the honest one.
        assert len(llm.requests) == 3
        # The honesty note reached the model exactly once, then was stripped.
        assert not any(_EVIDENCE_NOTE == str(m.get("content")) for m in messages)

    def test_honest_report_is_not_retried(self) -> None:
        llm = _ScriptedLLM(
            [
                ("run", {"command": "python -m pytest"}),
                HONEST,
            ]
        )
        agent = _FakeAgent(llm, REFUSAL)
        final_text, _, llm_error, _ = _run(agent)

        assert llm_error is None
        assert final_text == HONEST
        assert len(llm.requests) == 2

    def test_successful_tool_work_is_not_retried(self) -> None:
        """Evidence exists, so a completion claim is allowed."""
        llm = _ScriptedLLM(
            [
                ("git", {"subcommand": "status"}),
                "Task complete: working tree clean.",
            ]
        )
        agent = _FakeAgent(llm, SUCCESS)
        final_text, _, llm_error, loop = _run(agent)

        assert llm_error is None
        assert final_text == "Task complete: working tree clean."
        assert loop.tool_calls_made == 1
        assert len(llm.requests) == 2

    def test_question_with_failed_lookup_is_not_retried(self) -> None:
        llm = _ScriptedLLM(
            [
                ("search", {"query": "widget_factory"}),
                "The symbol is not present in any indexed file.",
            ]
        )
        agent = _FakeAgent(llm, "Search error: no such directory")
        final_text, _, llm_error, _ = _run(
            agent, user_input="where is widget_factory?"
        )

        assert llm_error is None
        assert "not present" in final_text
        assert len(llm.requests) == 2

    def test_text_only_completion_claim_triggers_retry(self) -> None:
        """No tool call at all + "task complete" on an imperative turn."""
        llm = _ScriptedLLM([FABRICATED, HONEST])
        agent = _FakeAgent(llm, SUCCESS)
        final_text, messages, llm_error, _ = _run(agent)

        assert llm_error is None
        assert agent.tool_calls == []
        assert final_text == HONEST
        assert len(llm.requests) == 2
        assert not any(_EVIDENCE_NOTE == str(m.get("content")) for m in messages)

    def test_text_only_completion_claim_answers_a_question(self) -> None:
        """A question may be answered in text, even with a "done" claim."""
        llm = _ScriptedLLM(["Yes — the build is done."])
        agent = _FakeAgent(llm, SUCCESS)
        final_text, _, llm_error, _ = _run(
            agent, user_input="is the build done?"
        )

        assert llm_error is None
        assert final_text == "Yes — the build is done."
        assert len(llm.requests) == 1
