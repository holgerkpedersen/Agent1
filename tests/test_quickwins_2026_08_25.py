"""Regression tests for the 2026-08-25 quick-win batch (improvement plan).

Covers:
- #1  decisions injected into the chat_nlp system message
- #14 plan-mode answers persisted to ``.docs/<ts>/plan_proposed.md``
- #5  char-budget chat-history trimming (count cap + budget cap, no
      assistant/tool pair splits)
- #15 ``multillm --synthesize`` merge call and its failure paths
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import agent
from agent import Agent, _HISTORY_CHAR_BUDGET, _MAX_CHAT_MESSAGES, _trim_chat_history
from agent_core.commands.multillm_cmd import MultiLlmCommand
from agent_core.llm.parallel import ParallelResult, ParallelRun


@pytest.fixture()
def agent_obj() -> Agent:
    return Agent(workspace=".")


def _msg(role: str, content: str, **kw) -> dict:
    return {"role": role, "content": content, **kw}


# ---------------------------------------------------------------------------
# #1 — decisions in the chat system prompt
# ---------------------------------------------------------------------------

class TestDecisionConstraintsInChat:
    def test_empty_when_no_files_read(self, agent_obj: Agent) -> None:
        agent_obj._files_read.clear()
        assert agent_obj._decision_constraints_block() == ""

    def test_block_mentions_decision_for_read_file(
        self, agent_obj: Agent,
    ) -> None:
        # tool_loop.py is referenced by several recorded decisions.
        agent_obj._files_read.add("agent_core/llm/tool_loop.py")
        block = agent_obj._decision_constraints_block()
        assert "CRITICAL DESIGN CONSTRAINTS" in block
        assert "Decision #" in block

    def test_never_raises_on_broken_ledger(self, agent_obj: Agent, monkeypatch) -> None:
        agent_obj._files_read.add("agent_core/llm/tool_loop.py")
        import agent as agent_mod

        def boom(*a, **kw):
            raise RuntimeError("ledger corrupt")

        monkeypatch.setattr(agent_mod, "decisions_as_system_prompt", boom)
        assert agent_obj._decision_constraints_block() == ""

    def test_chat_nlp_rebuilds_system_message_with_block(
        self, agent_obj: Agent, tmp_path: Path, monkeypatch,
    ) -> None:
        """chat_nlp must refresh message[0] with the constraints block.

        The agent's persisted state files are redirected to *tmp_path* so
        this test neither reads nor writes the real repo's chat history /
        memory (and cannot leak ``_files_read`` into sibling tests via the
        shared on-disk state).
        """
        import json as _json

        hist_file = tmp_path / "chat_history.json"
        mem_file = tmp_path / "agent_memory.json"
        hist_file.write_text("[]", encoding="utf-8")
        mem_file.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(agent, "CHAT_HISTORY_JSON_PATH", str(hist_file))
        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem_file))
        fresh = Agent(workspace=".")
        fresh._files_read.add("agent_core/llm/tool_loop.py")

        captured = {}

        async def fake_loop_run(self_loop, **kwargs):
            captured["messages"] = kwargs["messages"]
            return "done.", kwargs["messages"]

        monkeypatch.setattr(
            agent.ToolLoopRunner, "run", fake_loop_run,
        )
        asyncio.run(fresh.chat_nlp("hello"))
        system = captured["messages"][0]
        assert system["role"] == "system"
        assert "CRITICAL DESIGN CONSTRAINTS" in system["content"]
        assert agent._SYSTEM_PROMPT in system["content"]
        # The dynamic blocks must not accumulate across turns.
        base_len = len(agent._SYSTEM_PROMPT)
        rebuilt = agent._strip_dynamic_system_blocks(system["content"])
        assert len(rebuilt) == base_len


# ---------------------------------------------------------------------------
# #14 — plan-mode answer persistence
# ---------------------------------------------------------------------------

class TestPlanAnswerPersistence:
    def test_persist_writes_plan_proposed_md(self, agent_obj: Agent, tmp_path) -> None:
        # Point the agent at an isolated workspace so the test never touches
        # the real repo's .docs/ tree.
        agent_obj.workspace = str(tmp_path)
        agent_obj._persist_plan_answer("1. Do the thing\n2. Test it")
        docs = tmp_path / ".docs"
        run_dirs = [d for d in docs.iterdir() if d.is_dir()]
        assert run_dirs, "expected a fresh .docs run folder"
        out = run_dirs[0] / "plan_proposed.md"
        text = out.read_text(encoding="utf-8")
        assert "# Proposed plan" in text
        assert "1. Do the thing" in text
        assert "build mode" in text

    def test_persist_failure_is_contained(self, agent_obj: Agent, monkeypatch) -> None:
        import agent as agent_mod

        def boom(*a, **kw):
            raise OSError("disk full")

        monkeypatch.setattr(
            "agent_core.commands.doc_paths.new_run_dir", boom,
        )
        # Must not raise.
        agent_obj._persist_plan_answer("plan text")


# ---------------------------------------------------------------------------
# #5 — token-aware history trimming
# ---------------------------------------------------------------------------

class TestHistoryCharBudgetTrim:
    def test_small_history_untouched(self) -> None:
        msgs = [
            _msg("system", "sys"),
            _msg("user", "hi"),
            _msg("assistant", "hello"),
        ]
        assert _trim_chat_history(msgs) == msgs

    def test_count_cap_still_enforced(self) -> None:
        msgs = [_msg("system", "sys")] + [
            _msg("user", f"u{i}") for i in range(_MAX_CHAT_MESSAGES + 40)
        ]
        out = _trim_chat_history(msgs)
        assert len(out) <= _MAX_CHAT_MESSAGES
        assert out[0]["role"] == "system"

    def test_char_budget_trims_oldest_and_notes(self) -> None:
        msgs = [_msg("system", "sys")]
        for i in range(30):
            msgs.append(_msg("user", "x" * 4000))
            msgs.append(_msg("assistant", "y" * 4000))
        out = _trim_chat_history(msgs)
        body_chars = sum(
            agent._message_size(m)
            for m in out[1:]
            if "context compaction" not in str(m.get("content"))
        )
        assert body_chars <= _HISTORY_CHAR_BUDGET
        notes = [
            m for m in out if "context compaction" in str(m.get("content"))
        ]
        assert len(notes) == 1
        # Newest messages survive; oldest are gone.
        assert out[-1] == msgs[-1]

    def test_tool_pair_never_split(self) -> None:
        pair_msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "c1",
                "type": "function",
                "function": {"name": "read", "arguments": "{\"path\": \"a.py\"}"},
            }],
        }
        msgs = [
            _msg("system", "sys"),
            _msg("user", "q"),
            pair_msg,
            {"role": "tool", "tool_call_id": "c1", "content": "z" * (_HISTORY_CHAR_BUDGET + 10)},
            _msg("assistant", "done"),
        ]
        out = _trim_chat_history(msgs)
        roles = [(m.get("role"), "tool_calls" in m) for m in out]
        for i, (role, has_calls) in enumerate(roles):
            if role == "tool":
                prev_role, prev_has_calls = roles[i - 1]
                assert prev_role == "assistant" and prev_has_calls


# ---------------------------------------------------------------------------
# #15 — multillm --synthesize
# ---------------------------------------------------------------------------

def _run_with_results(*texts: str) -> ParallelRun:
    run = ParallelRun(template_id="t")
    run.results = [
        ParallelResult(model=f"m{i}", provider="lmstudio", text=t, ok=bool(t.strip()))
        for i, t in enumerate(texts)
    ]
    return run


class TestMultillmSynthesize:
    def test_flag_parsed_and_question_not_polluted(self, agent_obj: Agent, capsys) -> None:
        """--synthesize must not leak into the question text."""
        calls = []

        async def fake_parallel(*a, **kw):
            calls.append(kw)
            return _run_with_results("", "")  # no usable answers -> skip path

        with patch_module(fake_parallel):
            asyncio.run(MultiLlmCommand().execute(
                ["what", "is", "x", "--synthesize"], agent_obj,
            ))
        out = capsys.readouterr().out
        assert "--synthesize" not in out.split("question:")[-1].split("\n")[0]

    def test_synthesize_merges_through_llm(self, agent_obj: Agent, capsys) -> None:
        run = _run_with_results("Answer A says yes.", "Answer B says no.")
        llm_calls = []

        async def fake_chat(messages, **kw):
            llm_calls.append(messages)
            return "MERGED: it depends."

        agent_obj.llm.chat = fake_chat
        MultiLlmCommand()._synthesize(run, agent_obj, "the question?", True)
        out = capsys.readouterr().out
        assert "SYNTHESIZED ANSWER" in out
        assert "MERGED: it depends." in out
        assert len(llm_calls) == 1
        user_text = llm_calls[0][-1]["content"]
        assert "Answer A says yes." in user_text
        assert "Answer B says no." in user_text

    def test_synthesize_skips_with_fewer_than_two_answers(
        self, agent_obj: Agent, capsys,
    ) -> None:
        async def fail_chat(messages, **kw):
            raise AssertionError("LLM must not be called")

        agent_obj.llm.chat = fail_chat
        MultiLlmCommand()._synthesize(
            _run_with_results("only one"), agent_obj, "q", True,
        )
        assert "needs at least two usable answers" in capsys.readouterr().out

    def test_synthesize_swallows_llm_error(self, agent_obj: Agent, capsys) -> None:
        async def error_chat(messages, **kw):
            return "[Error: model down]"

        agent_obj.llm.chat = error_chat
        MultiLlmCommand()._synthesize(
            _run_with_results("A text here!", "B text here!"),
            agent_obj, "q", True,
        )
        out = capsys.readouterr().out
        assert "SYNTHESIZED ANSWER" not in out
        assert "model error" in out


def patch_module(fake_parallel):
    from unittest.mock import patch

    # execute() imports run_parallel from agent_core.llm.parallel at call
    # time, so the source module is what must be patched.
    return patch(
        "agent_core.llm.parallel.run_parallel",
        side_effect=fake_parallel,
    )
