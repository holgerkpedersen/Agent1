"""Tests for the Kanban board purge tool (``harnessfix.kanban_purge``).

The purge deletes data, so these tests pin the two properties that matter:
it removes *only* the exact junk shapes, and it refuses to write when any
safety check fails (format drift, orphaned children, live board app).
"""

from __future__ import annotations

import json

import pytest

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

    def write(board: dict, raw: bytes | None = None) -> None:
        path = tmp_path / "board-agent1.json"
        path.write_bytes(raw if raw is not None else kp.serialize_board(board))

    write(_board([]))
    return tmp_path, write


# --- classification ---------------------------------------------------------


@pytest.mark.parametrize(
    "title", ["x", "y", "z", "low", "high", "bench", "ælø", "æølæøl", " X "]
)
def test_placeholder_titles_are_junk(title):
    assert kp.classify_card(_card("c1", title)) == "placeholder-title"


def test_pytest_temp_path_title_is_junk():
    title = (
        "best-effort-except in C:/Users/Holger/AppData/Local/Temp/"
        "pytest-of-Holger/pytest-10313/test_resolve_foo0/sample.py:5"
    )
    assert kp.classify_card(_card("c1", title)) == "pytest-temp-title"


@pytest.mark.parametrize(
    "title",
    [
        "Deleting a Frame",
        "Work button",
        "Remove the opencode-zen freetier models",
        "Kanban card coloring",
        "xx",
        "best-effort-except in C:/Dev/Agent1/agent_core/suppress_log.py:1",
    ],
)
def test_real_titles_are_kept(title):
    assert kp.classify_card(_card("c1", title)) is None


# --- planning ---------------------------------------------------------------


def test_plan_separates_junk_from_real_and_counts_frames():
    cards = [
        _card("j1", "x"),
        _card("j2", "bench"),
        _card("r1", "Deleting a Frame"),
        _card("r2", "Work button", frame="f-new"),
    ]
    plan = kp.plan_purge(_board(cards))

    assert plan["total"] == 4
    assert plan["kept"] == 2
    assert plan["removed"] == 2
    assert plan["remove_ids"] == ["j1", "j2"]
    assert plan["reasons"] == {"placeholder-title": 2}
    assert plan["per_frame_kept"] == {"Issues": 1, "New": 1}


def test_plan_flags_a_kept_card_whose_parent_is_junk():
    cards = [_card("j1", "x"), _card("r1", "Deleting a Frame", parent="j1")]
    plan = kp.plan_purge(_board(cards))
    assert plan["kept_with_deleted_parent"] == ["r1"]


# --- serialization ----------------------------------------------------------


def test_serialize_board_matches_kanban_on_disk_format():
    board = _board([_card("c1", "Work button")])
    raw = kp.serialize_board(board)

    assert b"\r\n" in raw
    assert raw.count(b"\n") == raw.count(b"\r\n"), "bare LF would corrupt the file"
    assert not raw.endswith(b"\n"), "the app writes no trailing newline"
    assert json.loads(raw.decode("utf-8")) == board
    assert kp.serialize_board(json.loads(raw.decode("utf-8"))) == raw


def test_serialize_board_keeps_non_ascii_raw():
    board = _board([_card("c1", "ælø")])
    assert "ælø".encode("utf-8") in kp.serialize_board(board)


# --- purge + verify ---------------------------------------------------------


def test_purge_drops_cards_and_their_frame_membership():
    board = _board([_card("j1", "x"), _card("r1", "Work button")])
    new_board, plan = kp.purge_board(board)

    assert [c["id"] for c in new_board["cards"]] == ["r1"]
    issues = next(f for f in new_board["frames"] if f["id"] == "f-issues")
    assert issues["card_ids"] == ["r1"]
    assert kp.verify_purge(new_board, plan, board) == []


def test_purge_refuses_when_a_kept_card_would_lose_its_parent():
    board = _board([_card("j1", "x"), _card("r1", "Work button", parent="j1")])
    with pytest.raises(ValueError, match="lose their parent"):
        kp.purge_board(board)


def test_verify_purge_detects_a_frame_referring_to_a_deleted_card():
    board = _board([_card("j1", "x"), _card("r1", "Work button")])
    new_board, plan = kp.purge_board(board)
    new_board["frames"][1]["card_ids"].append("j1")

    problems = kp.verify_purge(new_board, plan, board)
    assert any("frame refers to deleted cards" in p for p in problems)


def test_verify_purge_detects_a_deleted_real_card():
    board = _board([_card("j1", "x"), _card("r1", "Work button")])
    new_board, plan = kp.purge_board(board)
    new_board["cards"] = []
    new_board["frames"][1]["card_ids"] = []

    problems = kp.verify_purge(new_board, plan, board)
    assert any("kept cards were deleted" in p for p in problems)


# --- apply ------------------------------------------------------------------


def test_apply_writes_board_makes_backup_and_verifies(board_file):
    tmp_path, write = board_file
    write(_board([_card("j1", "x"), _card("j2", "bench"), _card("r1", "Work button")]))

    result = kp.apply_purge("agent1")

    assert result["kept"] == 1
    assert result["removed"] == 2
    assert result["reasons"] == {"placeholder-title": 2}

    backup = tmp_path / result["backup"].split("\\")[-1]
    assert backup.exists()
    assert len(json.loads(backup.read_bytes())["cards"]) == 3, "backup must keep all"

    on_disk = json.loads((tmp_path / "board-agent1.json").read_bytes().decode())
    assert [c["id"] for c in on_disk["cards"]] == ["r1"]


def test_apply_refuses_format_drift(board_file):
    tmp_path, write = board_file
    board = _board([_card("j1", "x")])
    # indent=4 is not what the app writes; rewriting would make a huge diff.
    write(board, raw=json.dumps(board, indent=4).encode())

    with pytest.raises(ValueError, match="does not round-trip"):
        kp.apply_purge("agent1")


def test_apply_refuses_while_board_app_is_live(board_file, monkeypatch):
    _, write = board_file
    write(_board([_card("j1", "x")]))
    monkeypatch.setattr(kp, "_port_is_live", lambda *a, **k: True)

    with pytest.raises(RuntimeError, match="listening"):
        kp.apply_purge("agent1")

    assert kp.apply_purge("agent1", force=True)["removed"] == 1


def test_apply_reports_stranded_map_entries_without_editing_the_map(board_file):
    tmp_path, write = board_file
    write(_board([_card("j1", "x"), _card("r1", "Work button")]))
    map_path = tmp_path / ".card_id_map.json"
    original = {"j1": "iss-a", "iss-a": "j1", "r1": "iss-b", "iss-b": "r1"}
    map_path.write_text(json.dumps(original), encoding="utf-8")

    result = kp.apply_purge("agent1")

    assert result["links"]["stranded_card_keys"] == ["j1"]
    assert result["links"]["stranded_count"] == 1
    assert json.loads(map_path.read_text(encoding="utf-8")) == original


def test_scan_is_read_only(board_file):
    tmp_path, write = board_file
    board = _board([_card("j1", "x"), _card("r1", "Work button")])
    write(board)
    before = (tmp_path / "board-agent1.json").read_bytes()

    assert kp.main(["scan", "--board", "agent1"]) == 0
    assert (tmp_path / "board-agent1.json").read_bytes() == before
    assert not list(tmp_path.glob("*.bak-purge-*"))


def test_scan_reports_an_unknown_board(tmp_path, monkeypatch):
    monkeypatch.setattr(kp.kb, "DATA_DIR", tmp_path)
    assert kp.main(["scan", "--board", "nope"]) == 1
