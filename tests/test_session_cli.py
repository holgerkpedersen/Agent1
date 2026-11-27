"""Startup session selection and listing."""

from __future__ import annotations

import logging
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


# ── --workspace round-trip through session meta ─────────────────────────────
#
# A resumed session must come back to the checkout it was created in.  Without
# this, ``--session foo`` silently snapped back to whatever directory the
# process sat in, so a run meant for one repository landed in another.


def _ws_dir(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.mkdir()
    (path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    return path


def test_new_session_records_workspace(tmp_path: Path) -> None:
    ws = _ws_dir(tmp_path, "checkout")
    selection = parse_and_activate(["--new-session", "work"], workspace=str(ws))
    assert not selection.should_exit
    from agent_core.session.store import load_meta

    assert Path(load_meta("work").workspace) == ws


def test_resumed_session_restores_workspace(tmp_path: Path) -> None:
    ws = _ws_dir(tmp_path, "checkout")
    parse_and_activate(["--new-session", "work"], workspace=str(ws))
    reset()
    # Resume with NO explicit workspace: the recorded one must win.
    selection = parse_and_activate(["--session", "work"])
    assert not selection.should_exit
    assert Path(selection.workspace) == ws


def test_explicit_workspace_beats_recorded(tmp_path: Path) -> None:
    recorded = _ws_dir(tmp_path, "recorded")
    other = _ws_dir(tmp_path, "other")
    parse_and_activate(["--new-session", "work"], workspace=str(recorded))
    reset()
    selection = parse_and_activate(["--session", "work"], workspace=str(other))
    assert Path(selection.workspace) == other


def test_vanished_recorded_workspace_falls_back(tmp_path: Path, capsys) -> None:
    ws = _ws_dir(tmp_path, "gone")
    parse_and_activate(["--new-session", "work"], workspace=str(ws))
    reset()
    import shutil

    shutil.rmtree(ws)
    selection = parse_and_activate(["--session", "work"])
    assert not selection.should_exit
    assert Path(selection.workspace) != ws
    assert Path(selection.workspace).is_dir()
    assert "no longer exists" in capsys.readouterr().out


# ── the real ``--workspace`` flag, not just the keyword ─────────────────────
#
# Every test above passes ``workspace=`` as a keyword, so none of them prove
# that ``python agent.py --workspace PATH`` is wired to the same selection.
# The flag is the documented entry point (``--help``), so it needs its own
# coverage: a regression that drops it from ``build_arg_parser`` would leave
# all of the keyword tests green.


def test_workspace_flag_is_parsed(tmp_path: Path) -> None:
    ws = _ws_dir(tmp_path, "checkout")
    ns = build_arg_parser().parse_known_args(
        ["--new-session", "work", "--workspace", str(ws)]
    )[0]
    assert Path(ns.workspace) == ws


def test_workspace_flag_selects_and_records(tmp_path: Path) -> None:
    ws = _ws_dir(tmp_path, "checkout")
    selection = parse_and_activate(["--new-session", "work", "--workspace", str(ws)])
    assert not selection.should_exit
    assert Path(selection.workspace) == ws

    from agent_core.session.store import load_meta

    assert Path(load_meta("work").workspace) == ws


def test_workspace_flag_beats_keyword_hint(tmp_path: Path) -> None:
    # The namespace value wins over the caller's default hint, so an explicit
    # flag can never be silently overridden by a stale default.
    flagged = _ws_dir(tmp_path, "flagged")
    hinted = _ws_dir(tmp_path, "hinted")
    selection = parse_and_activate(
        ["--new-session", "work", "--workspace", str(flagged)],
        workspace=str(hinted),
    )
    assert Path(selection.workspace) == flagged


def test_bad_workspace_flag_is_ignored_not_silently_used(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ghost = tmp_path / "does-not-exist"
    with caplog.at_level(logging.WARNING):
        selection = parse_and_activate(
            ["--new-session", "work", "--workspace", str(ghost)]
        )
    assert not selection.should_exit
    # The failure mode this guards: a typo'd path quietly becoming the default
    # with no signal at all.
    assert Path(selection.workspace).is_dir()
    assert str(ghost) not in str(selection.workspace)
    assert "is not a directory; ignoring it" in caplog.text

