"""Startup session selection and listing."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_core.session import create_session, list_sessions, reset
from agent_core.session.cli import (
    build_arg_parser,
    format_session_table,
    parse_and_activate,
    resolve_startup,
)
from agent_core.session.context import current_session_id


@pytest.fixture(autouse=True)
def _session_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AGENT_SESSION_ROOT", str(tmp_path / "sessions"))
    reset()
    yield
    reset()


def test_parser_supports_session_flags() -> None:
    parser = build_arg_parser()
    ns = parser.parse_known_args(["--session", "work", "--dashboard"])[0]
    assert ns.session == "work"
    assert ns.dashboard is True


def test_new_session_is_created_and_activated() -> None:
    selection = parse_and_activate(["--new-session", "Work"], workspace="C:/ws")
    assert selection.should_exit is False
    assert selection.session_id == "work"
    assert current_session_id() == "work"
    assert [m.id for m in list_sessions()] == ["work", "default"]


def test_missing_session_exits_with_guidance(capsys) -> None:
    selection = parse_and_activate(["--session", "nope"], workspace="C:/ws")
    assert selection.should_exit is True
    out = capsys.readouterr().out
    assert "does not exist" in out
    assert "--new-session" in out


def test_second_activation_of_same_session_is_refused(capsys) -> None:
    parse_and_activate(["--new-session", "work"], workspace="C:/ws")
    selection = parse_and_activate(["--session", "work"], workspace="C:/ws")
    assert selection.should_exit is True
    assert "already in use" in capsys.readouterr().out


def test_list_sessions_does_not_activate() -> None:
    create_session("work")
    reset()
    selection = resolve_startup(
        build_arg_parser().parse_known_args(["--list-sessions"])[0]
    )
    assert selection.should_exit is True
    assert current_session_id() == "default"


def test_format_session_table_marks_locks() -> None:
    create_session("work")
    text = format_session_table()
    assert "work" in text
    assert "default" in text
    assert "free" in text
