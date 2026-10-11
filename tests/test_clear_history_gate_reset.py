"""Regression (plan audit 2026-10): clear_history must reset the full-run gate.

Before the fix, ``clear`` wiped chat history and the memory JSONs but left
``_full_run_gate`` untouched: a session that had already spent its one
per-session full-suite budget left every later (cleared) session inheriting
that spent budget — permanently refused full runs. The 2026-10 plan audit of
the ``clear`` command found this ghost state; these tests pin the wiring
(gate reset IS invoked, best-effort, and its failure cannot break clear).

The gate itself is exercised in tests/test_pytest_full_run_gate.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent import Agent


def _make_agent() -> Agent:
    """Construct a fully-mocked Agent (mirrors test_clear_history.py)."""
    with patch("agent.LLMClient"), \
         patch("agent.resolve_model", return_value="mock-model"), \
         patch("agent.to_windows_path", side_effect=lambda x: x), \
         patch("agent.FullRunGate") as gate_cls, \
         patch("agent.FileSystem"), \
         patch("agent.FileSearcher"), \
         patch("agent.ToolDispatcher"), \
         patch("agent.load_agent_settings", return_value={}):
        agent = Agent(workspace=".")
    # The patched class is a MagicMock: the instance assigned in __init__
    # records calls, so we can assert reset() was invoked by clear_history.
    agent._full_run_gate = gate_cls.return_value
    return agent


@pytest.mark.asyncio
async def test_clear_history_resets_full_run_gate() -> None:
    agent = _make_agent()

    agent.clear_history()

    gate = agent._full_run_gate
    assert gate.reset.called, "clear_history must reset the full-run gate"


@pytest.mark.asyncio
async def test_gate_reset_failure_does_not_break_clear() -> None:
    """The gate reset is best-effort: its failure must not break clear."""
    agent = _make_agent()

    def _boom() -> None:
        raise RuntimeError("gate exploded")

    agent._full_run_gate.reset.side_effect = _boom

    # Must not raise: a broken gate must never kill the clear command.
    agent.clear_history()

    assert gate_ok(agent)


def gate_ok(agent: Agent) -> bool:
    """Clear completed fully when every core state container is empty."""
    return (
        len(agent._chat_history) == 0
        and len(agent._history) == 0
        and agent._turn_start_index == 0
    )


if __name__ == "__main__":
    pytest.main([__file__])
