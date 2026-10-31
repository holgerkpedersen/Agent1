"""Session-layer isolation, storage, and locking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_core.memory.persistence import _load_chat_history, _save_chat_history
from agent_core.session import (
    SessionError,
    SessionLock,
    SessionLocked,
    SessionNotFound,
    activate_session,
    create_session,
    current_paths,
    current_session_id,
    delete_session,
    list_sessions,
    load_meta,
    paths_for,
    rename_session,
    reset,
    slugify,
)
from agent_core.session.context import is_active


class FakeAgent:
    def __init__(self) -> None:
        self._chat_history = []
        self._original_goal = ""


@pytest.fixture(autouse=True)
def _session_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AGENT_SESSION_ROOT", str(tmp_path / "sessions"))
    reset()
    yield
    reset()


def test_slugify_normalizes_and_rejects_traversal() -> None:
    assert slugify("Work Session") == "work-session"
    assert slugify("a/b") == "a-b"
    with pytest.raises(SessionError):
        slugify("...")
    with pytest.raises(SessionError):
        slugify("")


def test_default_session_aliases_legacy_root_paths() -> None:
    paths = paths_for("default")
    assert paths.session_id == "default"
    assert paths.chat_history_path.endswith("chat_history.json")
    assert paths.agent_memory_path.endswith("agent_memory.json")
    assert ".sessions" not in paths.chat_history_path


def test_named_session_paths_are_under_session_root(tmp_path: Path) -> None:
    paths = paths_for("work")
    assert paths.chat_history_path.startswith(
        str(tmp_path / "sessions" / "work")
    )
    assert paths.turn_log_path.endswith("turn_log.jsonl")
    assert paths.memory_db_path.endswith("agent_memory.db")


def test_create_list_load_rename_delete() -> None:
    meta = create_session("Work 1", workspace="C:/ws", model="m")
    assert meta.id == "work-1"
    assert [m.id for m in list_sessions()] == ["work-1", "default"]
    assert load_meta("work-1").model == "m"

    renamed = rename_session("work-1", "refactor")
    assert renamed.id == "refactor"
    assert not paths_for("work-1").chat_history.exists()
    assert paths_for("refactor").chat_history.parent.is_dir()

    delete_session("refactor")
    assert [m.id for m in list_sessions()] == ["default"]
    with pytest.raises(SessionNotFound):
        load_meta("refactor")


def test_activate_makes_paths_current_and_switch_is_isolated() -> None:
    create_session("one")
    create_session("two")
    activate_session("one")
    assert is_active()
    assert current_session_id() == "one"

    agent = FakeAgent()
    agent._chat_history = [
        {"role": "user", "content": "only in one"},
        {"role": "assistant", "content": "yes"},
    ]
    _save_chat_history(agent)
    one_path = Path(current_paths().chat_history_path)
    assert one_path.is_file()

    activate_session("two")
    assert current_session_id() == "two"
    assert _load_chat_history(FakeAgent()) == []

    agent._chat_history = [
        {"role": "user", "content": "only in two"},
        {"role": "assistant", "content": "yes"},
    ]
    _save_chat_history(agent)
    two_path = Path(current_paths().chat_history_path)
    assert two_path.is_file()

    activate_session("one")
    loaded = _load_chat_history(FakeAgent())
    assert loaded[-2]["content"] == "only in one"
    assert json.loads(two_path.read_text(encoding="utf-8"))[-2]["content"] == (
        "only in two"
    )


def test_session_lock_refuses_a_second_holder(tmp_path: Path) -> None:
    create_session("locked")
    paths = paths_for("locked")
    first = SessionLock(paths, "C:/ws")
    first.acquire()
    second = SessionLock(paths, "C:/ws")
    try:
        with pytest.raises(SessionLocked):
            second.acquire()
    finally:
        second.release()
        first.release()

    third = SessionLock(paths, "C:/ws")
    third.acquire()
    third.release()
