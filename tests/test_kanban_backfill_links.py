"""Backfilling cards the bridge never linked — the "Work does nothing" repair.

A card only becomes an issue through a ``card_create`` message, which the board
emits when the card is created *through the API*.  A card typed into the board
JSON, restored from a backup, or predating the sync has no issue behind it, so
every later message for it (update/move/delete/**work**) is consumed as a quiet
no-op — the board reports success while nothing happens.

``kanban_inspect.backfill_card_links()`` replays the bridge's CREATE path for
those cards.  These tests pin the properties that make the repair safe to run
against a live board: it is idempotent, it never imports a finished card as an
open issue, it never bridges fixture junk, and ``--dry-run`` writes nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

AGENT1_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT1_ROOT))

from harnessfix import issues as issue_store  # noqa: E402
from harnessfix import kanban_bridge as kb  # noqa: E402
from harnessfix import kanban_inspect as ki  # noqa: E402


def _card(card_id: str, title: str, frame_id: str, text: str = "", **extra):
    return {
        "id": card_id,
        "title": title,
        "text": text,
        "frame_id": frame_id,
        "tags": extra.pop("tags", []),
        "status": extra.pop("status", "open"),
        **extra,
    }


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Isolate the ledger, queues, id map and board in one temp dir."""
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(kb, "DATA_DIR", data)
    monkeypatch.setattr(kb, "QUEUE_AGENT1_TO_KANBAN", data / "queue-agent1-to-kanban")
    monkeypatch.setattr(kb, "QUEUE_KANBAN_TO_AGENT1", data / "queue-kanban-to-agent1")
    monkeypatch.setattr(kb, "CARD_ID_MAP_PATH", data / ".card_id_map.json")
    monkeypatch.setattr(kb, "DEAD_LETTER_DIR", data / "dead-letter")
    monkeypatch.setattr(issue_store, "ISSUES_PATH", tmp_path / "issues.json")
    env_file = tmp_path / ".env"
    env_file.write_text("KANBAN_SYNC_ENABLED=1\n", encoding="utf-8")
    monkeypatch.setattr(kb, "ENV_FILE_PATH", env_file)
    monkeypatch.delenv("KANBAN_SYNC_ENABLED", raising=False)

    (data / "active.json").write_text(json.dumps({"active": "b1"}), encoding="utf-8")
    _write_board(
        data,
        [
            _card("copen", "Unlinked open card", "f1", text="do the thing"),
            _card("cdone", "Unlinked finished card", "f3"),
            _card("clinked", "Already linked card", "f1"),
            _card("x", "x", "f1"),
        ],
    )
    # A card named "already linked" must really be linked, or the fixture is
    # lying about what the backfill is supposed to leave alone.
    _linked_issue("clinked")
    return data


def _write_board(data: Path, cards: list[dict]) -> None:
    (data / "board-b1.json").write_text(
        json.dumps(
            {
                "frames": [
                    {"id": "f1", "title": "Issues"},
                    {"id": "f2", "title": "On-going"},
                    {"id": "f3", "title": "Finished"},
                ],
                "cards": cards,
                "sync": {"agent1": True},
            }
        ),
        encoding="utf-8",
    )


def _linked_issue(card_id: str) -> str:
    """Bridge one card properly, the way a real card_create would."""
    assert kb._apply_card_create(
        {"id": card_id, "title": "Already linked card", "text": "", "frame_id": "f1"}
    )
    issue_id = kb.resolve_id(card_id)
    assert issue_id
    return issue_id


def _by_card(summary: dict) -> dict[str, dict]:
    return {r["card_id"]: r for r in summary["results"]}


# ── the repair ───────────────────────────────────────────────────────────────


def test_unlinked_card_gains_an_issue_and_both_link_forms(sandbox) -> None:
    """The repair writes the issue and records the link where the bridge reads it."""
    summary = ki.backfill_card_links()

    assert summary["created"] == 1
    entry = _by_card(summary)["copen"]
    assert entry["action"] == "created"
    issue_id = entry["issue_id"]
    assert issue_id

    # Both link forms, because the bridge resolves through either: the map is
    # a cache, the ``kanban:<card>`` location is the durable copy.
    assert kb.resolve_id("copen") == issue_id
    issue = next(i for i in issue_store.load_issues() if i["id"] == issue_id)
    assert "kanban:copen" in issue["locations"]
    assert issue["status"] == "open"
    # The card body must travel with the title, as on the real create path.
    assert "do the thing" in (
        issue["suggested_approach"] + issue["evidence"]
    )


def test_backfill_unblocks_card_work(sandbox, monkeypatch) -> None:
    """The point of the repair: the Work button stops being a silent no-op."""
    ki.backfill_card_links()
    claimed: list[str] = []

    def fake_claim(issue_id: str) -> tuple[bool, str]:
        claimed.append(issue_id)
        return True, "ok"

    monkeypatch.setattr(issue_store, "claim_card_for_work", fake_claim)

    applied = kb._process_single_message(
        {"op": "card_work", "source_id": "copen", "payload": {"title": "t"}}
    )

    assert applied is True
    assert claimed == [kb.resolve_id("copen")]


def test_rerun_is_idempotent(sandbox) -> None:
    """A second run must not mint a second issue for the same card."""
    first = ki.backfill_card_links()
    assert first["created"] == 1
    before = len(issue_store.load_issues())

    second = ki.backfill_card_links()

    assert second["created"] == 0
    assert len(issue_store.load_issues()) == before


def test_issue_already_in_the_ledger_counts_as_linked(sandbox) -> None:
    """A lost id-map entry is not an unlinked card — the location still links it."""
    _linked_issue("clinked")
    kb.CARD_ID_MAP_PATH.unlink()  # simulate the map being lost

    summary = ki.backfill_card_links()

    assert "clinked" not in _by_card(summary), "the ledger location already links it"
    assert summary["created"] == 1, "only the genuinely unlinked card is created"


# ── the two ways a naive backfill does damage ────────────────────────────────


def test_finished_card_is_not_imported_as_open_work(sandbox) -> None:
    """A card in a closed column is skipped: the move that resolved it already ran.

    Creating a fresh issue for it would mint an *open* record for work the board
    has finished, and nothing would ever resolve it.
    """
    summary = ki.backfill_card_links()

    entry = _by_card(summary)["cdone"]
    assert entry["action"] == "skipped"
    assert "closed column" in entry["reason"]
    assert not kb.resolve_id("cdone")


def test_apply_closed_bridges_it_with_the_closed_status(sandbox) -> None:
    """With the flag, the issue inherits the column's status instead of `open`."""
    summary = ki.backfill_card_links(apply_closed=True)

    entry = _by_card(summary)["cdone"]
    assert entry["action"] == "created"
    issue = next(i for i in issue_store.load_issues() if i["id"] == entry["issue_id"])
    assert issue["status"] == "resolved"
    assert issue["resolved_at"], "a resolved record carries its resolution time"


def test_fixture_junk_is_never_bridged(sandbox) -> None:
    """A placeholder card must not become a real issue through a repair."""
    entry = _by_card(ki.backfill_card_links())["x"]
    assert entry["action"] == "skipped"
    assert entry["reason"].startswith("junk:")


def test_status_tag_cannot_overrule_the_column(sandbox) -> None:
    """A stale ``[status:...]`` tag must not reopen work on a NEW record."""
    _write_board(
        sandbox,
        [_card("ctag", "Tagged card", "f3", tags=["[status:open]"])],
    )

    summary = ki.backfill_card_links(apply_closed=True)

    issue = next(
        i
        for i in issue_store.load_issues()
        if i["id"] == summary["results"][0]["issue_id"]
    )
    assert issue["status"] == "resolved", "the column is the truth, not the tag"


# ── dry run ──────────────────────────────────────────────────────────────────


def test_dry_run_reports_what_it_would_do_and_writes_nothing(sandbox) -> None:
    before_issues = issue_store.load_issues()
    before_map = kb._load_id_map()
    assert before_issues, "the fixture links one card, so the ledger is not empty"

    summary = ki.backfill_card_links(dry_run=True)

    assert summary["dry_run"] is True
    assert summary["would_create"] == 1
    assert summary["created"] == 0
    assert _by_card(summary)["copen"]["action"] == "dry-run"
    assert issue_store.load_issues() == before_issues, "the dry run wrote to the ledger"
    assert kb._load_id_map() == before_map, "the dry run wrote to the id map"


# ── CLI wiring ───────────────────────────────────────────────────────────────


def test_cli_backfill_links_prints_the_summary(sandbox, capsys) -> None:
    assert ki.main(["backfill-links", "--dry-run"]) == 0
    out = json.loads(capsys.readouterr().out)

    assert out["dry_run"] is True
    assert out["cards_on_board"] == 4
    assert any(r["card_id"] == "copen" for r in out["results"])


def test_cli_backfill_links_accepts_a_board_id(sandbox, capsys) -> None:
    assert ki.main(["backfill-links", "--board", "b1", "--apply-closed"]) == 0
    out = json.loads(capsys.readouterr().out)

    assert out["board_id"] == "b1"
    assert out["apply_closed"] is True


# ── REPL surface ─────────────────────────────────────────────────────────────


class _Agent:
    """Only what the ``kanban`` command reads off the agent."""

    def __init__(self, mode: str = "build") -> None:
        self.mode = mode


def _repl(args: list[str], mode: str = "build") -> None:
    import asyncio

    from agent_core.commands.kanban_cmd import KanbanCommand

    asyncio.run(KanbanCommand().execute(args, _Agent(mode)))


def test_repl_backfill_links_repairs_and_reports(sandbox, capsys) -> None:
    """The REPL is how an agent actually runs this, so it must be wired."""
    capsys.readouterr()

    _repl(["backfill-links"])

    out = capsys.readouterr().out
    assert '"created": 1' in out
    assert kb.resolve_id("copen"), "the REPL path did not link the card"
    assert "cdone" in out, "the skipped finished card should be reported"


def test_repl_backfill_links_is_blocked_in_plan_mode(sandbox, capsys) -> None:
    """Read-only sessions must not write, matching ``sync-descriptions``."""
    capsys.readouterr()

    _repl(["backfill-links"], mode="plan")

    out = capsys.readouterr().out
    assert "plan mode" in out
    assert not kb.resolve_id("copen")
