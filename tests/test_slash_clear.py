"""Regression tests for ``/clear`` in the dashboard chat.

THE BUG
-------
``clear`` was missing from ``slash.OFFLINE_COMMANDS``, so in the browser
``/clear`` was not a command at all: ``run_slash`` returned ``None``, the text
became an ordinary prompt and the LLM *answered* it ("I have cleared the
conversation", with nothing cleared).  The REPL cleared memory; the dashboard
invented a story about it.

THE TRAP BEHIND THE FIX
-----------------------
Adding ``clear`` to the allowlist is not enough.  ``ClearCommand.execute``
confirms with the BUILTIN ``input("Clear all memory? (y/n): ")``, and
``catalog.needs_terminal_input`` only matches ``commands.base.read_*`` — so the
stdin guard cannot see it and the request would park on a terminal nobody is
watching.  ``slash`` therefore drives the REPL command's own non-interactive
form instead, and these tests prove no prompt is ever issued.

The last test runs the REAL dashboard path (``AgentTurnRunner.send``), because
the symptom was never in ``slash`` alone: it needs the turn pipeline to *not*
be reached.
"""

from __future__ import annotations

import asyncio
import builtins

import pytest

from agent_core.commands.catalog import needs_terminal_input
from agent_core.commands.clear_cmd import ClearCommand
from agent_core.commands.slash import OFFLINE_COMMANDS, run_slash
from agent_core.monitoring.chat_api import AgentTurnRunner


class _FakeAgent:
    """Exactly what ``ClearCommand`` and ``slash`` touch, and nothing more."""

    def __init__(self, stats: dict | None = None) -> None:
        self.cleared = 0
        self._stats = stats or {
            "chat_history": 0,
            "files_read": 0,
            "working_memory": 0,
            "semantic_index": 0,
            "knowledge_graph": 0,
            "knowledge_graph_entries": 0,
            "stale_files": 0,
        }

    def memory_stats(self) -> dict:
        return dict(self._stats)

    def clear_history(self) -> None:
        self.cleared += 1
        for key in list(self._stats):
            self._stats[key] = 0


@pytest.fixture()
def no_prompt(monkeypatch):
    """Fail loudly if anything tries to read the terminal."""

    def boom(*a, **k):
        raise AssertionError("a browser command must never prompt on stdin")

    monkeypatch.setattr(builtins, "input", boom)


def _populated_agent() -> _FakeAgent:
    return _FakeAgent(
        {
            "chat_history": 4,
            "files_read": 2,
            "working_memory": 1,
            "semantic_index": 7,
            "knowledge_graph": 2,
            "knowledge_graph_entries": 9,
            "stale_files": 0,
        }
    )


# -- the command is no longer sent to the model -------------------------


def test_clear_is_an_offline_command() -> None:
    # The root cause: without this the message reached the LLM.
    assert "clear" in OFFLINE_COMMANDS


def test_clear_is_answered_without_the_model(no_prompt) -> None:
    agent = _populated_agent()

    result = run_slash(agent, "/clear")

    assert result is not None, "/clear must not fall through to the turn pipeline"
    assert result["ok"] is True
    assert result["command"] == "clear"
    assert agent.cleared == 1, "the REPL's own clear_history must run"


def test_clear_reports_what_it_destroyed(no_prompt) -> None:
    agent = _populated_agent()

    result = run_slash(agent, "/clear")

    assert result is not None
    reply = result["reply"]
    assert "Memory state" in reply, "the user must see what was there"
    assert "knowledge graph" in reply
    assert "Agent memory cleared." in reply
    assert result["kind"] == "clear"


def test_clear_on_empty_memory_says_so_and_clears_nothing(no_prompt) -> None:
    agent = _FakeAgent()

    result = run_slash(agent, "/clear")

    assert result is not None and result["ok"] is True
    assert "Nothing to clear" in result["reply"]
    assert agent.cleared == 0, "nothing to clear means clear_history is not called"


# -- ``clear stats`` only reports ---------------------------------------


def test_clear_stats_reports_without_clearing(no_prompt) -> None:
    agent = _populated_agent()

    result = run_slash(agent, "/clear stats")

    assert result is not None and result["ok"] is True
    assert "Memory state" in result["reply"]
    assert "chat history:       4 messages" in result["reply"]
    assert agent.cleared == 0, "/clear stats must not destroy memory"
    assert result["kind"] == "text", "a stats read is not a clear event"


# -- the stdin guard cannot save us, so the code must not prompt --------


def test_stdin_probe_does_not_see_clear_s_prompt() -> None:
    """Documents WHY ``slash`` bypasses the registry for ``clear``.

    ``ClearCommand`` confirms with the builtin ``input()``, which
    ``needs_terminal_input`` (a ``read_choice``/``read_input``/``read_stdin``
    source probe) does not match.  If this ever starts returning True the
    slash path would refuse ``/clear`` instead of running it.
    """
    assert needs_terminal_input(ClearCommand()) is False


def test_clear_never_blocks_on_a_terminal(no_prompt, monkeypatch) -> None:
    # ``input`` is patched to raise: reaching it fails this test.
    agent = _populated_agent()

    result = run_slash(agent, "/clear")

    assert result is not None and result["ok"] is True


# -- end to end: the dashboard route the page actually posts to ---------


class _RecordingAgent(_FakeAgent):
    """Fake agent for ``AgentTurnRunner``: records any LLM turn attempt."""

    def __init__(self) -> None:
        # Populated memory: "Nothing to clear" is a different branch, covered
        # by its own test above.
        super().__init__(
            {
                "chat_history": 2,
                "files_read": 1,
                "working_memory": 0,
                "semantic_index": 0,
                "knowledge_graph": 1,
                "knowledge_graph_entries": 3,
                "stale_files": 0,
            }
        )
        self.turns: list[str] = []
        self.model_name = "fake-model"
        self._chat_history: list[dict] = []
        self._last_turn_answer = ""
        self._last_turn_error = None

    async def chat_nlp(self, message: str):
        self.turns.append(message)
        self._last_turn_answer = "AN LLM ANSWER"
        return None


def test_dashboard_clear_runs_no_llm_turn() -> None:
    """The reported bug, at the real entry point."""
    agent = _RecordingAgent()
    runner = AgentTurnRunner(agent=agent)

    result = runner.send("/clear")

    assert agent.turns == [], "/clear must not reach the model in the dashboard"
    assert result["ok"] is True
    assert result["command"] == "clear"
    assert agent.cleared == 1
    assert "Agent memory cleared." in result["reply"]


def test_dashboard_plain_text_still_runs_a_turn() -> None:
    # The offline path must not swallow ordinary messages.
    agent = _RecordingAgent()
    runner = AgentTurnRunner(agent=agent)

    result = runner.send("hello there")

    assert agent.turns == ["hello there"]
    assert result["reply"] == "AN LLM ANSWER"


def test_run_slash_is_sync_safe_for_the_http_handler(no_prompt) -> None:
    """``run_slash`` is called from a sync HTTP handler; it must not need a loop."""
    agent = _populated_agent()

    result = run_slash(agent, "/clear")

    assert result is not None
    assert asyncio.iscoroutine(result) is False
