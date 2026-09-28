"""Regression tests: the harness must not accept an unverified action claim.

Failure (2026-09-28 session): the user ran `git status` in the REPL, then typed
"commit changes".  ``git`` IS a registry command, so ``GitCommand.execute``
printed the status to stdout only — it never reached ``_chat_history``.  The
following turn therefore went to the model with NO repository state, the model
made ZERO tool calls, and it answered "Everything is already committed and
pushed."  Nothing was verified; nothing was committed.

Two layers of defence are covered here:
  1. the prompt rule (so the model is told not to assert unverified state), and
  2. the deterministic backstop (a zero-tool-call answer to an action request
     re-enters the loop instead of being accepted as the final answer).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import agent as agent_mod
from agent import Agent


# ---------------------------------------------------------------------------
# Layer 1 — the predicate that decides "this answer claims an unverified action"
# ---------------------------------------------------------------------------

class TestUnverifiedActionClaimPredicate:
    def test_action_request_with_zero_tool_calls_is_flagged(self) -> None:
        assert agent_mod._unverified_action_claim("commit changes", 0) is True

    def test_push_request_with_zero_tool_calls_is_flagged(self) -> None:
        assert agent_mod._unverified_action_claim("push it", 0) is True

    def test_action_request_made_no_claim_when_a_tool_was_used(self) -> None:
        # A turn that actually called the git tool has evidence; the model is
        # entitled to report what the tool returned.
        assert agent_mod._unverified_action_claim("commit changes", 1) is False

    def test_questions_about_git_are_not_action_requests(self) -> None:
        # "what does commit mean" is small talk / explanation, not a work order.
        assert agent_mod._unverified_action_claim("what does commit mean", 0) is False

    def test_question_asking_whether_something_is_committed_is_not_a_request(
        self,
    ) -> None:
        assert (
            agent_mod._unverified_action_claim("is my work already committed?", 0)
            is False
        )

    def test_non_action_chat_is_not_flagged(self) -> None:
        assert agent_mod._unverified_action_claim("hello there", 0) is False

    def test_empty_input_is_not_flagged(self) -> None:
        assert agent_mod._unverified_action_claim("", 0) is False


# ---------------------------------------------------------------------------
# Layer 2 — the backstop: a fabricated answer must not end the turn
# ---------------------------------------------------------------------------

def _bot(tmp_path: Path, monkeypatch) -> Agent:
    monkeypatch.setattr(agent_mod, "CHAT_HISTORY_JSON_PATH",
                        str(tmp_path / "chat_history.json"))
    monkeypatch.setattr(agent_mod, "AGENT_MEMORY_JSON_PATH",
                        str(tmp_path / "agent_memory.json"))
    return Agent(workspace=str(tmp_path))


class TestFabricatedCommitAnswerIsRejected:
    def test_zero_tool_call_answer_to_commit_request_chains_another_run(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        bot = _bot(tmp_path, monkeypatch)
        monkeypatch.setattr(agent_mod, "_MAX_CHAINED_RUNS", 3)

        seen: list[list[dict]] = []
        answers = iter([
            "Everything is already committed and pushed.",
            "git status showed staged.txt; staged, committed and pushed it.",
        ])

        async def fake_loop_run(self_loop, **kwargs):
            seen.append(list(kwargs["messages"]))
            if len(seen) == 1:
                # The buggy model state: a plain text answer, no tool ever run.
                self_loop.tool_calls_made = 0
            else:
                # It obeyed the verify note and actually looked at the repo.
                self_loop.tool_calls_made = 2
                self_loop.tools_used = {"git": 2}
            self_loop.last_tool_call = "git commit"
            self_loop.iterations_used = 1
            self_loop.termination_reason = "answer"
            return next(answers), kwargs["messages"]

        monkeypatch.setattr(agent_mod.ToolLoopRunner, "run", fake_loop_run)
        asyncio.run(bot.chat_nlp("commit changes"))

        # RED would fail here: today the fabricated answer ends the turn after
        # exactly one run, and the model never gets told to go verify.
        assert len(seen) == 2, (
            "a zero-tool-call answer to 'commit changes' must be re-run, not "
            "accepted as final"
        )
        injected = seen[1][-1]
        assert injected.get(agent_mod._CONTINUE_NOTE_TAG_KEY) == \
            agent_mod._CONTINUE_NOTE_TAG
        assert "git" in injected["content"].lower(), (
            "the steering note must point the model at the git tool"
        )

    def test_verified_answer_is_accepted_without_an_extra_run(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        bot = _bot(tmp_path, monkeypatch)
        monkeypatch.setattr(agent_mod, "_MAX_CHAINED_RUNS", 3)

        runs = {"n": 0}

        async def fake_loop_run(self_loop, **kwargs):
            runs["n"] += 1
            # The healthy path: the model DID call git this turn.
            self_loop.tool_calls_made = 3
            self_loop.tools_used = {"git": 3}
            self_loop.last_tool_call = "git commit"
            self_loop.iterations_used = 4
            self_loop.termination_reason = "answer"
            return "Committed and pushed.", kwargs["messages"]

        monkeypatch.setattr(agent_mod.ToolLoopRunner, "run", fake_loop_run)
        asyncio.run(bot.chat_nlp("commit changes"))

        assert runs["n"] == 1, "a tool-backed answer must end the turn"

    def test_injected_note_is_stripped_from_persisted_history(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        hist_file = tmp_path / "chat_history.json"
        bot = _bot(tmp_path, monkeypatch)
        monkeypatch.setattr(agent_mod, "_MAX_CHAINED_RUNS", 2)

        answers = iter(["Nothing to do, all committed.", "Verified with git."])
        runs = {"n": 0}

        async def fake_loop_run(self_loop, **kwargs):
            runs["n"] += 1
            # Run 1 fabricates (no tool); run 2 obeys the verify note and uses git.
            self_loop.tool_calls_made = 0 if runs["n"] == 1 else 1
            self_loop.tools_used = {} if runs["n"] == 1 else {"git": 1}
            self_loop.last_tool_call = "git status"
            self_loop.iterations_used = 1
            self_loop.termination_reason = "answer"
            return next(answers), kwargs["messages"]

        monkeypatch.setattr(agent_mod.ToolLoopRunner, "run", fake_loop_run)
        asyncio.run(bot.chat_nlp("commit changes"))

        saved = json.loads(hist_file.read_text(encoding="utf-8"))
        assert all(
            m.get(agent_mod._CONTINUE_NOTE_TAG_KEY) is None for m in saved
        ), "the loop-injected note must not persist into the next session"


def test_system_prompt_forbids_asserting_unverified_repo_state() -> None:
    """The prompt must state the rule the model keeps breaking."""
    low = agent_mod._SYSTEM_PROMPT.lower()
    assert "unverified" in low or "never claim" in low
    assert "git" in low, "the rule must be concrete about the git tool"
