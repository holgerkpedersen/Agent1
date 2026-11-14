"""Undo path + REPL surface for the Kanban purge (``harnessfix.kanban_purge``).

``tests/test_kanban_purge.py`` pins that the purge deletes *only* junk and
refuses unsafe writes.  This file pins the other half of "easy to delete":
that a wrong deletion is one command away from being reversed, and that the
REPL exposes the whole thing as ``kanban purge`` instead of a raw module call.

A purge is destructive, so the tests here are the safety net for the claim
"always backs the board up first".
"""

from __future__ import annotations

import json

import pytest

from agent_core.commands.kanban_cmd import KanbanCommand
from harnessfix import kanban_purge as kp


def _card(cid: str, title: str, frame: str = "f-issues", parent: str = "") -> dict:
    return {
        "id": cid,
        "title": title,
        "text": "",
        "frame_id": frame,
        "parent_id": parent,
        "tags": [],
        "system": "",
        "created_at": "",
        "updated_at": "",
        "status": "open",
    }


def _board(cards: list[dict]) -> dict:
    return {
        "frames": [
            {"id": "f-new", "title": "New", "is_reserved": True, "card_ids": []},
            {
                "id": "f-issues",
                "title": "Issues",
                "is_reserved": False,
                "card_ids": [c["id"] for c in cards],
            },
        ],
        "cards": cards,
        "sync": {"agent1": True},
    }


@pytest.fixture()
def board_file(tmp_path, monkeypatch):
    """A workspace-local data dir so the real board is never touched."""
    monkeypatch.setattr(kp.kb, "DATA_DIR", tmp_path)
    monkeypatch.setattr(kp, "_port_is_live", lambda *a, **k: False)
    monkeypatch.setattr(kp.kb, "CARD_ID_MAP_PATH", tmp_path / ".card_id_map.json")

    def write(board: dict) -> None:
        (tmp_path / "board-agent1.json").write_bytes(kp.serialize_board(board))

    write(_board([]))
    return tmp_path, write


def _cards_on_disk(tmp_path) -> list[str]:
    raw = (tmp_path / "board-agent1.json").read_bytes().decode("utf-8")
    return [c["id"] for c in json.loads(raw)["cards"]]


# --- restore / undo ---------------------------------------------------------


def test_restore_puts_every_card_back(board_file):
    tmp_path, write = board_file
    write(_board([_card("j1", "x"), _card("r1", "Work button")]))

    assert kp.apply_purge("agent1")["removed"] == 1
    assert _cards_on_disk(tmp_path) == ["r1"]

    result = kp.restore_board("agent1")

    assert result["cards_before"] == 1
    assert result["cards_after"] == 2
    assert _cards_on_disk(tmp_path) == ["j1", "r1"]


def test_undo_is_itself_undoable_and_keeps_the_current_file(board_file):
    tmp_path, write = board_file
    write(_board([_card("j1", "x"), _card("r1", "Work button")]))
    kp.apply_purge("agent1")

    result = kp.restore_board("agent1")
    aside = tmp_path / result["current_kept_as"].split("\\")[-1].split("/")[-1]

    assert aside.exists(), "the pre-undo file must be kept"
    assert [c["id"] for c in json.loads(aside.read_bytes())["cards"]] == ["r1"]


def test_restore_uses_the_newest_backup_by_default(board_file):
    tmp_path, write = board_file
    write(_board([_card("j1", "x")]))
    kp.apply_purge("agent1")
    # A second purge of an already-clean board still yields a backup.
    kp.apply_purge("agent1")

    backups = kp.list_backups("agent1")
    assert len(backups) == 2
    assert kp.restore_board("agent1")["restored_from"].endswith(backups[-1].name)


def test_backup_name_stays_unique_within_one_second(board_file):
    tmp_path, write = board_file
    write(_board([_card("j1", "x"), _card("r1", "Work button")]))

    kp.apply_purge("agent1")
    kp.apply_purge("agent1")  # same second: must not raise FileExistsError

    assert len(kp.list_backups("agent1")) == 2, "both backups must survive"
    assert _cards_on_disk(tmp_path) == ["r1"]


def test_restore_refuses_a_damaged_backup(board_file):
    tmp_path, write = board_file
    write(_board([_card("j1", "x")]))
    kp.apply_purge("agent1")

    bad = tmp_path / "board-agent1.json.bak-purge-19700101-000000"
    bad.write_bytes(json.dumps(_board([]), indent=4).encode())  # wrong format

    with pytest.raises(ValueError, match="does not round-trip"):
        kp.restore_board("agent1", backup=bad)

    assert _cards_on_disk(tmp_path) == [], "a refused restore must not write"


def test_restore_refuses_while_the_board_app_is_live(board_file, monkeypatch):
    tmp_path, write = board_file
    write(_board([_card("j1", "x")]))
    kp.apply_purge("agent1")

    monkeypatch.setattr(kp, "_port_is_live", lambda *a, **k: True)
    with pytest.raises(RuntimeError, match="listening"):
        kp.restore_board("agent1")


def test_restore_without_a_backup_says_so(board_file):
    with pytest.raises(FileNotFoundError, match="no purge backup"):
        kp.restore_board("agent1")


def test_restore_cli_lists_backups(board_file, capsys):
    _, write = board_file
    write(_board([_card("j1", "x")]))
    kp.apply_purge("agent1")
    capsys.readouterr()

    assert kp.main(["restore", "--list"]) == 0
    assert "board-agent1.json.bak-purge-" in capsys.readouterr().out


# --- board discovery --------------------------------------------------------


def test_available_boards_ignores_backup_files(board_file):
    tmp_path, write = board_file
    write(_board([_card("j1", "x")]))
    (tmp_path / "board-second.json").write_bytes(kp.serialize_board(_board([])))
    kp.apply_purge("agent1")

    assert kp.available_boards() == ["agent1", "second"]


# --- REPL command -----------------------------------------------------------


class _Agent:
    """Only what ``kanban purge`` touches."""

    def __init__(self, mode: str = "build") -> None:
        self.mode = mode


def _run(args: list[str], mode: str = "build") -> None:
    import asyncio

    asyncio.run(KanbanCommand().execute(args, _Agent(mode)))


def test_repl_purge_defaults_to_a_read_only_preview(board_file, capsys):
    tmp_path, write = board_file
    write(_board([_card("j1", "x"), _card("r1", "Work button")]))
    before = (tmp_path / "board-agent1.json").read_bytes()
    capsys.readouterr()

    _run(["purge"])

    out = capsys.readouterr().out
    assert "1 meaningless card" in out
    assert "kanban purge apply" in out
    assert (tmp_path / "board-agent1.json").read_bytes() == before
    assert not kp.list_backups("agent1")


def test_repl_purge_reports_an_already_clean_board(board_file, capsys):
    _, write = board_file
    write(_board([_card("r1", "Work button")]))
    capsys.readouterr()

    _run(["purge"])

    assert "already clean" in capsys.readouterr().out


def test_repl_purge_apply_deletes_and_undo_restores(board_file, capsys):
    tmp_path, write = board_file
    write(_board([_card("j1", "x"), _card("r1", "Work button")]))

    _run(["purge", "apply"])
    assert _cards_on_disk(tmp_path) == ["r1"]

    _run(["purge", "undo"])
    assert _cards_on_disk(tmp_path) == ["j1", "r1"]


def test_repl_purge_backups_lists_them(board_file, capsys):
    _, write = board_file
    write(_board([_card("j1", "x")]))
    kp.apply_purge("agent1")
    capsys.readouterr()

    _run(["purge", "backups"])

    assert "bak-purge-" in capsys.readouterr().out


def test_repl_purge_apply_is_blocked_in_plan_mode(board_file, capsys):
    tmp_path, write = board_file
    write(_board([_card("j1", "x")]))
    capsys.readouterr()

    _run(["purge", "apply"], mode="plan")

    out = capsys.readouterr().out
    assert "blocked in plan mode" in out
    assert _cards_on_disk(tmp_path) == ["j1"], "plan mode must not delete"


def test_repl_purge_preview_still_works_in_plan_mode(board_file, capsys):
    _, write = board_file
    write(_board([_card("j1", "x")]))
    capsys.readouterr()

    _run(["purge"], mode="plan")

    assert "1 meaningless card" in capsys.readouterr().out


def test_repl_purge_reports_a_refusal_instead_of_raising(board_file, capsys):
    _, write = board_file
    write(_board([_card("j1", "x")]))

    import agent_core.commands.kanban_cmd as kc

    def boom(*a, **k):
        raise RuntimeError("the Kanban app appears to be listening on :3000")

    orig = kc.kp.apply_purge
    kc.kp.apply_purge = boom
    try:
        _run(["purge", "apply"])
    finally:
        kc.kp.apply_purge = orig

    out = capsys.readouterr().out
    assert "Error:" in out and "listening" in out


def test_repl_purge_board_flag_targets_another_board(board_file, capsys):
    tmp_path, write = board_file
    write(_board([]))
    (tmp_path / "board-other.json").write_bytes(
        kp.serialize_board(_board([_card("j9", "bench")]))
    )
    capsys.readouterr()

    _run(["purge", "--board", "other"])

    assert "other" in capsys.readouterr().out
