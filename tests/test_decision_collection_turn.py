"""Phase 3.2 of the autonomous-decision plan: collect decisions per TURN.

Card ``9bc4327f`` ("Improve Autonomous Decisions in Agent1"), plan
``.decisions_plan.md`` — Phase 3.1 built
:class:`agent_core.decision_collector.DecisionCollector`; Phase 3.2 hooks it
into the session's turn boundary so every finished turn leaves a reviewable
summary of what the agent decided on its own.

The contract pinned here:

1. ``_finish_turn`` runs the collection at the END of a real ``chat_nlp``
   turn, from the tool calls and discrepancies buffered at the executor
   choke point — so the summary describes the turn that just happened.
2. The buffer is per turn: a new turn never reports the previous turn's
   calls, and a refused call is buffered WITH its rejection result (a call
   the guard refused decided nothing).
3. The turn's own messages are read BEFORE ``_trim_chat_history`` can drop
   them — the collector needs the user's text to tell "the agent chose this
   path" from "the user asked for this path".
4. Collection is bookkeeping: never promotes into the ledger (Phase 1 owns
   promotion), and a failing collector can never fail a finished turn.

These tests drive the REAL ``chat_nlp`` loop and the real collector — no
mocks of the collector, no network.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

from agent import Agent
from agent_core.decisions import load_decisions

#: A write target the user never names, so it is the agent's own choice.
_TARGET = "dates.py"


class _WriteLLM:
    """Scripted provider: one ``write`` tool call, then a final answer."""

    def __init__(self, path: str = _TARGET) -> None:
        self.path = path
        self.steps = 0

    async def chat(self, messages, tools=None, **kwargs):
        self.steps += 1
        if self.steps == 1:
            return json.dumps(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "write",
                                "arguments": json.dumps(
                                    {
                                        "path": self.path,
                                        "content": "def hi():\n    pass\n",
                                    }
                                ),
                            },
                        }
                    ],
                }
            )
        return "All done."


def _run_turn(tmp_path: Path, user_input: str, llm: _WriteLLM) -> Agent:
    """One real ``chat_nlp`` turn with all runtime state pointed at tmp_path."""
    with (
        patch("agent.CHAT_HISTORY_JSON_PATH", str(tmp_path / "chat_history.json")),
        patch("agent.AGENT_MEMORY_JSON_PATH", str(tmp_path / "agent_memory.json")),
    ):
        bot = Agent(workspace=str(tmp_path))
        bot.llm = llm

        async def run():
            await bot.chat_nlp(user_input)

        asyncio.run(run())
    return bot


# ── the turn boundary hook ────────────────────────────────────────────────


def test_finish_turn_collects_the_decisions_of_the_turn(tmp_path: Path) -> None:
    """A write to a path the user never named is collected at turn end."""
    bot = _run_turn(tmp_path, "Add a helper for dates.", _WriteLLM())

    titles = [d["title"] for d in bot._last_turn_decisions]
    assert titles == [f"Change {_TARGET}"]
    assert (tmp_path / _TARGET).exists(), "precondition: the write succeeded"


def test_collection_does_not_promote_into_the_ledger(tmp_path: Path) -> None:
    """Phase 1 already promoted per call; collection must never duplicate it."""
    _run_turn(tmp_path, "Add a helper for dates.", _WriteLLM())

    titles = [str(d.get("title")) for d in load_decisions(tmp_path)]
    assert titles.count(f"Change {_TARGET}") == 1, (
        f"expected exactly one ledger entry, got {titles!r}"
    )


def test_a_path_the_user_named_is_not_the_agents_decision(tmp_path: Path) -> None:
    """The collector's reference is the USER's text, not the tool result."""
    bot = _run_turn(tmp_path, f"Write {_TARGET} for me.", _WriteLLM())

    assert (tmp_path / _TARGET).exists(), "precondition: the write succeeded"
    assert [d["title"] for d in bot._last_turn_decisions] == []


def test_turn_messages_are_read_before_history_trimming(tmp_path: Path) -> None:
    """Trimming the history must not blind the collector.

    ``_finish_turn`` bounds ``_chat_history`` before the end-of-turn hooks.  If
    the turn's own user message is read AFTER the trim, the collector loses its
    "was this asked for?" reference and misreports a user-named path as an
    autonomous decision.
    """
    import agent_core.turn_loop as turn_loop

    with patch.object(turn_loop, "_trim_chat_history", lambda *a, **k: []):
        bot = _run_turn(tmp_path, f"Write {_TARGET} for me.", _WriteLLM())

    assert [d["title"] for d in bot._last_turn_decisions] == [], (
        "the user's own path was reported as a decision - the turn messages "
        "were read after _trim_chat_history emptied the history"
    )


# ── per-turn buffer semantics ─────────────────────────────────────────────


def test_the_buffer_is_reset_at_every_turn_start(tmp_path: Path) -> None:
    """A new turn must not inherit the previous turn's calls."""
    bot = Agent(workspace=str(tmp_path))
    bot._note_turn_tool_call("write", {"path": _TARGET}, "Verified.")
    assert bot._turn_tool_calls, "precondition: the call was buffered"

    bot._append_user_turn("next request", None)

    assert bot._turn_tool_calls == []
    assert bot._turn_discrepancy_records == []
    assert bot._last_turn_decisions == []


def test_a_refused_call_is_buffered_with_its_rejection(tmp_path: Path) -> None:
    """A write rejected by plan mode is buffered and decides nothing."""
    bot = Agent(workspace=str(tmp_path))
    bot.mode = "plan"

    asyncio.run(bot._execute_tool_call("write", {"path": _TARGET, "content": "x\n"}))

    assert len(bot._turn_tool_calls) == 1
    name, args, result = bot._turn_tool_calls[0]
    assert name == "write"
    assert args["path"] == _TARGET
    assert result, "the rejection result must be buffered with the call"

    bot._collect_autonomous_decisions_from_turn(
        [{"role": "user", "content": "Add a helper for dates."}]
    )
    assert bot._last_turn_decisions == []


def test_only_this_turns_calls_are_collected(tmp_path: Path) -> None:
    """Two turns in one session: each summary covers only its own calls."""
    bot = Agent(workspace=str(tmp_path))

    bot._note_turn_tool_call("write", {"path": "first.py"}, "Verified.")
    bot._collect_autonomous_decisions_from_turn(
        [{"role": "user", "content": "Add a helper."}]
    )
    assert [d["title"] for d in bot._last_turn_decisions] == ["Change first.py"]

    bot._append_user_turn("more work", None)
    bot._note_turn_tool_call("write", {"path": "second.py"}, "Verified.")
    bot._collect_autonomous_decisions_from_turn(
        [{"role": "user", "content": "Add another helper."}]
    )

    assert [d["title"] for d in bot._last_turn_decisions] == ["Change second.py"]


# ── never break a finished turn ───────────────────────────────────────────


def test_a_failing_collector_never_fails_the_turn(tmp_path: Path) -> None:
    """Collection is observability: its failure must be a no-op."""
    bot = _run_turn(tmp_path, "Add a helper for dates.", _WriteLLM())
    # The turn completed before the failure: prove the hook is guarded.
    assert bot._last_turn_decisions

    def _boom():
        raise RuntimeError("collector exploded")

    bot._decision_collector_for_workspace = _boom  # type: ignore[method-assign]
    bot._note_turn_tool_call("write", {"path": "other.py"}, "Verified.")

    collected = bot._collect_autonomous_decisions_from_turn(
        [{"role": "user", "content": "x"}]
    )
    assert collected == []
    assert bot._last_turn_decisions == []


def test_collection_survives_a_missing_hook(tmp_path: Path) -> None:
    """``_finish_turn`` must tolerate an object without the Phase 3 hook."""
    import agent_core.turn_loop as turn_loop
    from agent_core.llm.tool_loop import ToolLoopRunner

    class _Bare:
        """Only the attributes ``_finish_turn`` touches before the hook."""

        _chat_history: list = []
        _original_goal: str | None = None
        _turn_started_at = None
        _last_user_input = ""
        session_id = ""
        _turn_run_id = None
        mode = "build"

        def _mutating_files_this_turn(self):
            return []

        def is_plan_mode(self):
            return False

        def _save_chat_history(self):
            pass

        def _save_memory(self):
            pass

    with (
        patch("agent.CHAT_HISTORY_JSON_PATH", str(tmp_path / "chat_history.json")),
        patch("agent.AGENT_MEMORY_JSON_PATH", str(tmp_path / "agent_memory.json")),
    ):
        turn_loop._finish_turn(
            _Bare(),  # type: ignore[arg-type]
            final_text="All done.",
            llm_error=None,
            loop=ToolLoopRunner.__new__(ToolLoopRunner),
            display_mode=turn_loop.AgentDisplayMode.QUIET,
        )
