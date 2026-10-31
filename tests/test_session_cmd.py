"""REPL session command."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent_core.commands.session_cmd import SessionCommand
from agent_core.session import activate_session, create_session, reset


class FakeAgent:
    def __init__(self) -> None:
        self.session_id = "default"
        self.workspace = "C:/ws"
        self.llm = type("LLM", (), {"model_name": "m"})()
        self._plan = False
        self.switched_to = None

    def is_plan_mode(self) -> bool:
        return self._plan

    def switch_session(self, session_id: str) -> None:
        self.switched_to = session_id
        self.session_id = session_id


@pytest.fixture(autouse=True)
def _session_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AGENT_SESSION_ROOT", str(tmp_path / "sessions"))
    reset()
    yield
    reset()


def _run(args, agent: FakeAgent) -> None:
    asyncio.run(SessionCommand().execute(args, agent))


def test_list_marks_active_and_free(capsys) -> None:
    create_session("work")
    activate_session("work", workspace="C:/ws")
    agent = FakeAgent()
    agent.session_id = "work"
    _run(["list"], agent)
    out = capsys.readouterr().out
    assert "work" in out
    assert "active" in out


def test_new_creates_and_switches(capsys) -> None:
    agent = FakeAgent()
    _run(["new", "Work Item"], agent)
    assert agent.switched_to == "work-item"
    assert "Created and switched" in capsys.readouterr().out


def test_use_switches_existing() -> None:
    create_session("work")
    agent = FakeAgent()
    _run(["use", "work"], agent)
    assert agent.switched_to == "work"


def test_plan_mode_blocks_mutating_subcommands(capsys) -> None:
    agent = FakeAgent()
    agent._plan = True
    _run(["new", "work"], agent)
    assert agent.switched_to is None
    assert "blocked in plan mode" in capsys.readouterr().out


def test_delete_refuses_active(capsys) -> None:
    create_session("work")
    agent = FakeAgent()
    agent.session_id = "work"
    _run(["delete", "work", "--force"], agent)
    assert "Cannot delete the active session" in capsys.readouterr().out
