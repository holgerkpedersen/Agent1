"""Bridge content propagation + the kanban inspector's read path.

``tests/test_kanban_roundtrip.py`` covers the idempotency of card↔issue
creation. This file covers what a card *says*, which is the half that used to
get dropped: Kanban→Agent1 moved titles and tags but not the card description,
so an issue arrived with an empty body and every status table lived only in
comments nobody read.

Everything here runs against temp files — never the real ``data/`` queue.
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


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Isolate the ledger, the queue and the id map in one temp directory."""
    data = tmp_path / "data"
    data.mkdir()
    # kanban_inspect reads through kanban_bridge, so patching the bridge's
    # module-level paths redirects both sides.
    for mod in (kb,):
        monkeypatch.setattr(mod, "DATA_DIR", data)
        monkeypatch.setattr(
            mod, "QUEUE_AGENT1_TO_KANBAN", data / "queue-agent1-to-kanban"
        )
        monkeypatch.setattr(
            mod, "QUEUE_KANBAN_TO_AGENT1", data / "queue-kanban-to-agent1"
        )
        monkeypatch.setattr(mod, "CARD_ID_MAP_PATH", data / ".card_id_map.json")
        monkeypatch.setattr(mod, "DEAD_LETTER_DIR", data / "dead-letter")
    monkeypatch.setattr(issue_store, "ISSUES_PATH", tmp_path / "issues.json")

    (data / "active.json").write_text(
        json.dumps({"active": "b1"}), encoding="utf-8"
    )
    return data


def _board(data: Path, cards: list[dict]) -> None:
    """Write the active board the inspector reads."""
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


# ── description propagation (kanban → agent1) ────────────────────────────────


def test_plain_card_description_reaches_the_issue(sandbox) -> None:
    """A card body with no tags is the issue's content, not a lost paragraph."""
    card = {
        "id": "c1",
        "title": "Queue stalls after 3 retries",
        "text": "Messages stop moving when the board app restarts mid-poll.",
        "frame_id": "f1",
        "tags": [],
    }
    kb._apply_card_create(card)

    issues = issue_store.load_issues()
    assert len(issues) == 1
    issue = issues[0]
    body = f"{issue['evidence']}\n{issue['suggested_approach']}"
    assert "Messages stop moving when the board app restarts mid-poll." in body


def test_tagged_split_still_wins_over_the_raw_body(sandbox) -> None:
    """The ``---`` split is explicit authoring; the raw body must not clobber it."""
    card = {
        "id": "c2",
        "title": "Tagged card",
        "text": "observed failure\n\n---\nfix the poll loop",
        "frame_id": "f1",
        "tags": ["[bug]", "[severity:high]"],
    }
    kb._apply_card_create(card)

    issue = issue_store.load_issues()[0]
    assert issue["evidence"] == "observed failure"
    assert issue["suggested_approach"] == "fix the poll loop"
    assert issue["category"] == "bug"
    assert issue["severity"] == "high"


def test_card_update_fills_an_empty_issue_without_duplicates(sandbox) -> None:
    """A description added to an existing card lands once, in one field only.

    Repeated edits used to append the same paragraph over and over; board sync
    must fill gaps, never duplicate, and never touch a field that has text.
    """
    card = {"id": "c3", "title": "Existing issue", "text": "", "frame_id": "f1"}
    kb._apply_card_create(card)
    issues = issue_store.load_issues()
    issue_id = issues[0]["id"]

    for _ in range(3):
        kb._apply_card_update(
            {
                "source_id": "c3",
                "payload": {
                    "id": "c3",
                    "title": "Existing issue",
                    "text": "the real description",
                    "frame_id": "f1",
                },
            }
        )

    updated = issue_store.load_issues()[0]
    assert updated["suggested_approach"] == "the real description"
    # The create path already carried the (empty) body; the update path fills
    # only a gap, so the description appears exactly once across both fields.
    combined = f"{updated['evidence']} {updated['suggested_approach']}"
    assert combined.count("the real description") == 1
    assert issue_id == updated["id"]


def test_card_update_never_overwrites_human_text(sandbox) -> None:
    """Board sync must not clobber analysis a human wrote into the ledger."""
    kb._apply_card_create(
        {"id": "c3h", "title": "Human text", "text": "", "frame_id": "f1"}
    )
    issues = issue_store.load_issues()
    issue_id = issues[0]["id"]
    issue_store.save_issues(
        [
            {**i, "suggested_approach": "human analysis", "evidence": "human notes"}
            if i["id"] == issue_id
            else i
            for i in issues
        ]
    )

    kb._apply_card_update(
        {
            "source_id": "c3h",
            "payload": {
                "id": "c3h",
                "title": "Human text",
                "text": "board body that must not clobber",
                "frame_id": "f1",
            },
        }
    )

    updated = issue_store.find_by_id(issue_store.load_issues(), issue_id)
    assert updated["suggested_approach"] == "human analysis"
    assert "board body that must not clobber" not in updated["evidence"]


# ── key anatomy: the map drift is the documented truth ───────────────────────


def test_key_anatomy_names_the_map_as_a_drifting_cache(sandbox) -> None:
    """The id map is a cache; ``locations`` is the durable link.

    An agent that trusts the map alone loses cards created directly on the
    board, so the inspector has to say which source is authoritative.
    """
    anatomy = ki.key_anatomy()
    keys = {item["key"]: item for item in anatomy}

    assert any("source_id" in k for k in keys)          # kanban card id
    assert any("issue id" in k for k in keys)
    assert ".card_id_map.json" in keys
    assert "locations: ['kanban:<card-id>']" in keys
    # The cache is not the source of truth, and the inspector says so.
    assert "stale" in keys[".card_id_map.json"]["meaning"].lower()
    assert "DURABLE" in keys["locations: ['kanban:<card-id>']"]["meaning"]


def test_card_status_and_column_tables_agree_with_the_bridge(sandbox) -> None:
    """One column→status table, shared with the bridge, not a second copy."""
    assert ki.COLUMN_TO_ISSUE == kb.FRAME_TO_STATUS
    assert ki.ISSUE_STATUS_TO_CARD["resolved"] == "done"


# ── resolving a key from the REPL ────────────────────────────────────────────


def test_issue_view_resolves_card_id_and_shows_the_description(sandbox) -> None:
    _board(
        sandbox,
        [
            {
                "id": "c9",
                "title": "Bridge drops card body",
                "text": "the full card description",
                "frame_id": "f2",
                "tags": [],
                "status": "in-progress",
            }
        ],
    )
    kb._apply_card_create(
        {
            "id": "c9",
            "title": "Bridge drops card body",
            "text": "the full card description",
            "frame_id": "f2",
            "tags": [],
        }
    )

    view = ki.issue_view("c9")
    link = view["link"]
    assert link["card_id"] == "c9"
    assert link["issue_id"]
    assert any(".card_id_map.json" in hop for hop in link["hops"])

    card = view["card"]
    assert card["frame_title"] == "On-going"
    assert card["status"] == "in-progress"

    composed = ki.composed_description(view["issue"], card)
    assert "the full card description" in composed


def test_issue_view_resolves_a_title_prefix(sandbox) -> None:
    """Title/prefix resolution is the only way to find a never-bridged card."""
    _board(
        sandbox,
        [
            {
                "id": "cA",
                "title": "Refactor the queue reader",
                "text": "never bridged",
                "frame_id": "f1",
                "tags": [],
            }
        ],
    )

    view = ki.issue_view("refactor the queue")
    assert view["link"]["card_id"] == "cA"
    assert view["issue"] is None
    assert any("never linked" in w for w in view["link"]["warnings"])


def test_issue_view_reports_an_unknown_key_cleanly(sandbox) -> None:
    _board(sandbox, [])
    view = ki.issue_view("deadbeef")
    assert not view["link"].get("card_id")
    assert not view["link"].get("issue_id")
    assert view["link"]["warnings"]


def test_locations_heal_a_missing_map_entry(sandbox) -> None:
    """The durable link must win when the cache is empty.

    Deleting the map simulates the drift the integration suffers from: an
    update for a card whose mapping was lost used to be a silent no-op.
    """
    _board(
        sandbox,
        [
            {
                "id": "cB",
                "title": "Healed by locations",
                "text": "body",
                "frame_id": "f1",
                "tags": [],
            }
        ],
    )
    kb._apply_card_create(
        {
            "id": "cB",
            "title": "Healed by locations",
            "text": "body",
            "frame_id": "f1",
            "tags": [],
        }
    )
    (sandbox / ".card_id_map.json").unlink(missing_ok=True)

    view = ki.issue_view("cB")
    assert view["link"]["issue_id"]
    assert any(
        "kanban:cB" in hop for hop in view["link"]["hops"]
    ), "the locations hop should be named in the trace"


# ── queue and board views ────────────────────────────────────────────────────


def test_queue_status_resolves_pending_messages_to_issue_ids(sandbox) -> None:
    _board(sandbox, [])
    kb._apply_card_create(
        {
            "id": "cD",
            "title": "Queued card",
            "text": "body",
            "frame_id": "f1",
            "tags": [],
        }
    )
    issue_id = ki.resolve("cD")["issue_id"]
    kb.enqueue(
        {"op": "card_update", "source_id": "cD", "payload": {"title": "Queued card"}},
        sandbox / "queue-kanban-to-agent1",
    )

    status = ki.queue_status(limit=5)
    inbound = next(q for q in status["queues"] if q["direction"] == "kanban→agent1")
    assert inbound["pending"] >= 1
    resolved = [m for m in inbound["messages"] if m.get("resolved_issue")]
    assert resolved, "a queued card id must be resolved to its issue id"
    assert resolved[0]["resolved_issue"] == issue_id


def test_board_view_lists_columns_with_their_status_mapping(sandbox) -> None:
    _board(
        sandbox,
        [
            {
                "id": "cE",
                "title": "Board card",
                "text": "t",
                "frame_id": "f3",
                "tags": [],
                "status": "done",
            }
        ],
    )
    view = ki.board_view()
    assert view["board_id"] == "b1"
    titles = [f["title"] for f in view["frames"]]
    assert {"Issues", "On-going", "Finished"} <= set(titles)

    finished = next(f for f in view["frames"] if f["title"] == "Finished")
    assert finished["issue_status"] == "resolved"
    assert finished["cards"][0]["id"] == "cE"


def test_status_summary_survives_a_missing_board(sandbox) -> None:
    """The inspector is a diagnostic: a missing file is a warning, never a raise."""
    summary = ki.status_summary()
    assert summary["active_board"] == "b1"
    assert summary["map"]["entries"] == 0
    assert len(summary["queues"]) == 2


def test_sync_descriptions_fills_only_empty_fields(sandbox) -> None:
    """The backfill command must be safe to re-run and never overwrite text."""
    _board(
        sandbox,
        [
            {
                "id": "cF",
                "title": "Needs a body",
                "text": "the description from the board",
                "frame_id": "f1",
                "tags": [],
            },
            {
                "id": "cG",
                "title": "Already written up",
                "text": "board text that must not clobber",
                "frame_id": "f1",
                "tags": [],
            },
        ],
    )
    kb._apply_card_create(
        {"id": "cF", "title": "Needs a body", "text": "", "frame_id": "f1"}
    )
    kb._apply_card_create(
        {"id": "cG", "title": "Already written up", "text": "", "frame_id": "f1"}
    )
    issues = issue_store.load_issues()
    filled = next(i for i in issues if i["title"] == "Needs a body")
    human = next(i for i in issues if i["title"] == "Already written up")
    issue_store.save_issues(
        [
            {**i, "suggested_approach": "human analysis", "evidence": "human notes"}
            if i["id"] == human["id"]
            else {**i, "suggested_approach": "", "evidence": ""}
            for i in issues
        ]
    )

    preview = ki.sync_descriptions(dry_run=True)
    assert any(r["action"] == "dry-run" for r in preview)
    after_preview = {i["id"]: i for i in issue_store.load_issues()}
    assert after_preview[filled["id"]]["suggested_approach"] == "", "dry run wrote"

    results = ki.sync_descriptions(dry_run=False)
    assert any(r["action"] == "updated" for r in results)

    after = {i["id"]: i for i in issue_store.load_issues()}
    assert (
        after[filled["id"]]["suggested_approach"] == "the description from the board"
    )
    assert after[human["id"]]["suggested_approach"] == "human analysis"
    assert after[human["id"]]["evidence"] == "human notes"

    # Idempotent: a second run changes nothing.
    again = ki.sync_descriptions(dry_run=False)
    assert all(r["action"] == "skipped" for r in again)
