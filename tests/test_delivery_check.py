"""Jev-backed silent-ending delivery check (turn_loop).

Regression for the Kanban 2026-11-07 session: the model ended with
termination "answer" after real tool work but never delivered a clear final
summary (dangling narration / promised follow-up).  _looks_incomplete missed
it, so the turn stopped silently.  The Jev delivery check must trigger ONE
synthesis retry on a confident "not delivered" verdict, and fail open on
ambiguity, errors, or when heuristics already cover the case.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from agent_core.config import AgentDisplayMode
from agent_core.turn_loop import (
    _JEV_DELIVERY_NOTE,
    _jev_answer_delivered,
    _run_chained_tool_loop,
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

    def __init__(self, llm):
        self.llm = llm
        self.tool_calls: list[tuple[str, dict]] = []

    async def _execute_tool_call(self, name, args):
        self.tool_calls.append((name, args))
        return "On branch main\nnothing to commit, working tree clean"

    def _take_trace_effects(self):
        return []

    def _record_llm_experience(self, **kwargs):
        pass


def _run(agent, user_input="do the work", *, display_mode=AgentDisplayMode.QUIET):
    return asyncio.run(
        _run_chained_tool_loop(
            agent,  # type: ignore[arg-type]
            user_input=user_input,
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": user_input},
            ],
            display_mode=display_mode,
            seen_calls={},
        )
    )


class TestJevDeliveryHook:
    def test_confident_not_delivered_triggers_one_synthesis_retry(self, monkeypatch):
        # Script: tool call -> dangling narration (no tools) -> final summary
        llm = _ScriptedLLM(
            [
                ("git", {"subcommand": "status"}),
                "Checking impact before closing out.",
                "Done: committed f07c4cb and pushed main. No open issues.",
            ]
        )
        agent = _FakeAgent(llm)

        async def fake_delivered(user_input, final_text, *, workspace=None):
            return False  # confidently not delivered

        monkeypatch.setattr(
            "agent_core.turn_loop._jev_answer_delivered", fake_delivered
        )
        final_text, messages, llm_error, loop = _run(
            agent, user_input="finish the Kanban work and report"
        )

        assert llm_error is None
        assert "committed f07c4cb" in final_text
        assert loop.termination_reason == "answer"
        # The Jev note was injected (tagged) then stripped from history.
        assert not any(
            _JEV_DELIVERY_NOTE == str(m.get("content")) for m in messages
        )
        # Three LLM calls total: tool-call turn, dangling answer, synthesis retry.
        assert len(llm.requests) == 3

    def test_delivered_verdict_does_not_retry(self, monkeypatch):
        llm = _ScriptedLLM(
            [
                ("git", {"subcommand": "status"}),
                "All work is complete: committed and pushed.",
            ]
        )
        agent = _FakeAgent(llm)

        async def fake_delivered(user_input, final_text, *, workspace=None):
            return True

        monkeypatch.setattr(
            "agent_core.turn_loop._jev_answer_delivered", fake_delivered
        )
        final_text, _, llm_error, _ = _run(agent)
        assert llm_error is None
        assert final_text == "All work is complete: committed and pushed."
        assert len(llm.requests) == 2

    def test_jev_error_fail_open(self, monkeypatch):
        llm = _ScriptedLLM(
            [
                ("git", {"subcommand": "status"}),
                "Checking impact before closing out.",
            ]
        )
        agent = _FakeAgent(llm)

        async def fake_delivered(user_input, final_text, *, workspace=None):
            return None  # could not judge -> fail-open

        monkeypatch.setattr(
            "agent_core.turn_loop._jev_answer_delivered", fake_delivered
        )
        final_text, _, llm_error, _ = _run(agent)
        assert llm_error is None
        # Original (dangling) answer stands; no retry.
        assert final_text == "Checking impact before closing out."
        assert len(llm.requests) == 2

    def test_no_check_when_no_tool_work(self, monkeypatch):
        llm = _ScriptedLLM(["Hello, what can I do?"])
        agent = _FakeAgent(llm)
        called = {"n": 0}

        async def fake_delivered(user_input, final_text, *, workspace=None):
            called["n"] += 1
            return False

        monkeypatch.setattr(
            "agent_core.turn_loop._jev_answer_delivered", fake_delivered
        )
        final_text, _, llm_error, _ = _run(agent, user_input="hi")
        assert llm_error is None
        assert final_text == "Hello, what can I do?"
        assert called["n"] == 0

    def test_no_check_when_heuristics_already_flag_incomplete(self, monkeypatch):
        # Text carrying an incomplete marker is caught by _looks_incomplete; the
        # Jev delivery check must not run (the existing auto-continue handles it).
        llm = _ScriptedLLM(
            [
                ("git", {"subcommand": "status"}),
                "I'll keep working — follow-up work remains.",
                "Finished everything.",
            ]
        )
        agent = _FakeAgent(llm)
        called = {"n": 0}

        async def fake_delivered(user_input, final_text, *, workspace=None):
            called["n"] += 1
            return False

        monkeypatch.setattr(
            "agent_core.turn_loop._jev_answer_delivered", fake_delivered
        )
        final_text, _, llm_error, _ = _run(agent)
        assert llm_error is None
        assert final_text == "Finished everything."
        assert called["n"] == 0  # heuristic path handled it, not Jev


class TestJevAnswerDeliveredHelper:
    def _patch_engine(self, monkeypatch, *, probs, abstentions=0, n=3, error=""):
        decision = (
            "FALSE" if probs.get("no", 0) >= probs.get("yes", 0) else "TRUE"
        )

        class _FakeEngine:
            async def decide(self, question):
                return SimpleNamespace(
                    probabilities=probs,
                    abstentions=abstentions,
                    n=n,
                    error=error,
                    decision=decision,
                )

        monkeypatch.setattr(
            "agent_core.jev_engine.build_jev_engine",
            lambda **kwargs: _FakeEngine(),
        )

    def test_confident_no_is_not_delivered(self, monkeypatch):
        monkeypatch.delenv("AGENT_NO_JEV_DELIVERY_CHECK", raising=False)
        self._patch_engine(monkeypatch, probs={"yes": 0.2, "no": 0.8})
        assert asyncio.run(_jev_answer_delivered("req", "dangling")) is False

    def test_confident_yes_is_delivered(self, monkeypatch):
        monkeypatch.delenv("AGENT_NO_JEV_DELIVERY_CHECK", raising=False)
        self._patch_engine(monkeypatch, probs={"yes": 0.9, "no": 0.1})
        assert asyncio.run(_jev_answer_delivered("req", "done, committed")) is True

    def test_ambiguous_is_fail_open(self, monkeypatch):
        # Mirrors the weak/iffy sanity check: ~0.51 confidence -> don't intervene.
        monkeypatch.delenv("AGENT_NO_JEV_DELIVERY_CHECK", raising=False)
        self._patch_engine(monkeypatch, probs={"yes": 0.49, "no": 0.51})
        assert asyncio.run(_jev_answer_delivered("req", "maybe done")) is True

    def test_abstained_is_fail_open(self, monkeypatch):
        monkeypatch.delenv("AGENT_NO_JEV_DELIVERY_CHECK", raising=False)
        self._patch_engine(monkeypatch, probs={}, abstentions=3, n=3)
        assert asyncio.run(_jev_answer_delivered("req", "text")) is None

    def test_empty_final_text_is_not_delivered(self, monkeypatch):
        # Live path (no kill-switch): empty = not delivered.
        monkeypatch.delenv("AGENT_NO_JEV_DELIVERY_CHECK", raising=False)
        assert asyncio.run(_jev_answer_delivered("req", "   ")) is False

    def test_empty_final_text_fail_open_under_kill_switch(self, monkeypatch):
        monkeypatch.setenv("AGENT_NO_JEV_DELIVERY_CHECK", "1")
        assert asyncio.run(_jev_answer_delivered("req", "   ")) is None

    def test_engine_error_is_fail_open(self, monkeypatch):
        monkeypatch.delenv("AGENT_NO_JEV_DELIVERY_CHECK", raising=False)

        def boom(**kwargs):
            raise RuntimeError("LM Studio down")

        monkeypatch.setattr("agent_core.jev_engine.build_jev_engine", boom)
        assert asyncio.run(_jev_answer_delivered("req", "text")) is None
