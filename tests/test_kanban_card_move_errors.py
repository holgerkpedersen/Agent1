"""Regression tests for "Error when moving a Kanban Card".

Written AFTER reproducing each defect against the real bridge and the real
queue (``C:\\Dev\\Kanban\\data``) and BEFORE fixing it.

Reproduced root causes
----------------------
1. **The card's issue exists, but the id map lost the pair.**  ``_lookup_issue_for``
   resolved a card ONLY through ``.card_id_map.json``.  The live map holds 357
   keys, 280 of them card guids, and just 32 of those name a card that is still
   on the board (248 are stale), so a move for a card whose issue is in the
   ledger — reachable
   through the ``kanban:<card_id>`` location that ``_apply_card_create``
   writes — was classified "not tracked by the bridge" and consumed as a quiet
   no-op.  The move reported success (``True``) and changed nothing.  One of
   the five dead letters in the live inbound queue (``f01a7cb8``) is exactly
   this class; the other four are board-local cards, consumed as quiet no-ops
   since 2026-11-01.
2. **Moving a card reopened its issue.**  ``_resolve_frame_status`` returned
   ``"open"`` whenever the column title was not in ``FRAME_TO_STATUS`` (an
   unknown/renamed column, or a pre-fix message carrying only an opaque
   ``frame_id``), and ``_apply_card_move`` applied that unconditionally.  So
   "Finished" -> ``resolved`` followed by any unrecognised column -> ``open``,
   silently undoing the resolution.
3. **A ledger record without a ``status`` key raised ``KeyError``** in
   ``_apply_card_move``, which surfaced as a retry and then a dead letter
   instead of an applied move.  Latent today (all 15 ledger records carry the
   key) but reachable through the production ``flags="repair"`` path.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Point the bridge and the ledger at an isolated temp directory."""
    import harnessfix.kanban_bridge as kb
    from harnessfix import issues as issue_store

    data = tmp_path / "data"
    (data / "queue-agent1-to-kanban").mkdir(parents=True)
    (data / "queue-kanban-to-agent1").mkdir(parents=True)
    monkeypatch.setattr(kb, "DATA_DIR", data)
    monkeypatch.setattr(kb, "CARD_ID_MAP_PATH", data / ".card_id_map.json")
    monkeypatch.setattr(kb, "QUEUE_AGENT1_TO_KANBAN", data / "queue-agent1-to-kanban")
    monkeypatch.setattr(kb, "QUEUE_KANBAN_TO_AGENT1", data / "queue-kanban-to-agent1")
    monkeypatch.setattr(kb, "ENV_FILE_PATH", tmp_path / "absent.env")
    monkeypatch.delenv("KANBAN_SYNC_ENABLED", raising=False)
    monkeypatch.delenv("KANBAN_WORKING_DIR", raising=False)
    monkeypatch.setattr(issue_store, "ISSUES_PATH", tmp_path / ".issues.json")
    return kb, issue_store


CARD = "f01a7cb8fb514e7d8e48f8d224b66d24"


def _seed(store, status="open"):
    """Create a bridged issue, then drop its id-map entry (the live failure)."""
    kb, issue_store = store
    issue = issue_store.make_issue(
        category="general", title="clear command", locations=[f"kanban:{CARD}"]
    )
    issue["status"] = status
    ledger = issue_store.load_issues()
    issue_store.upsert(ledger, issue)
    issue_store.save_issues(ledger)
    # The live map lost this pair: the issue is in the ledger, the map is not.
    assert kb._load_id_map() == {}
    return issue


def _status(issue_store, issue_id):
    return issue_store.find_by_id(issue_store.load_issues(), issue_id)["status"]


def _move(store, frame_title=None, frame_id="opaque-uuid"):
    kb, _ = store
    payload = {"frame_id": frame_id}
    if frame_title is not None:
        payload["frame_title"] = frame_title
    return kb._apply_card_move(
        {"op": "card_move", "source_id": CARD, "payload": payload}
    )


# ---------------------------------------------------------------------------
# 1. A card whose issue is reachable through its kanban: location
# ---------------------------------------------------------------------------


class TestMoveResolvesCardWithoutIdMapEntry:
    """The move must find the issue recorded by ``_apply_card_create``."""

    def test_move_applies_when_id_map_entry_is_missing(self, store):
        """The reported symptom: the move succeeded but the issue never moved."""
        issue = _seed(store)
        _, issue_store = store

        assert _move(store, "Finished") is True
        assert _status(issue_store, issue["id"]) == "resolved", (
            "a card whose issue carries a 'kanban:<card_id>' location must be "
            "resolved by a move even when .card_id_map.json lost the pair"
        )

    def test_move_selfheals_the_id_map(self, store):
        """The recovered pair must be re-recorded, so the next move is cheap."""
        kb, _ = store
        issue = _seed(store)

        _move(store, "Finished")

        mapping = kb._load_id_map()
        assert mapping.get(CARD) == issue["id"]
        assert mapping.get(issue["id"]) == CARD, "the reverse lookup must heal too"

    def test_unmapped_card_with_no_issue_stays_a_quiet_noop(self, store):
        """Board-local cards (79 of them live) must not start dead-lettering."""
        kb, issue_store = store
        assert kb._apply_card_move(
            {"op": "card_move", "source_id": "board-local-card", "payload": {}}
        ) is True
        assert issue_store.load_issues() == []

    def test_mapped_but_missing_issue_is_still_an_anomaly(self, store):
        """A map entry pointing at an absent issue must keep failing loudly."""
        kb, issue_store = store
        issue = _seed(store)
        kb.link_ids(issue["id"], CARD)
        ledger = issue_store.load_issues()
        ledger[:] = [i for i in ledger if i["id"] != issue["id"]]
        issue_store.save_issues(ledger)

        assert _move(store, "Finished") is False


# ---------------------------------------------------------------------------
# 2. An unrecognised column must not reopen the issue
# ---------------------------------------------------------------------------


class TestUnknownColumnDoesNotReopen:
    """An unknown column gives no status, so the status must not change."""

    def test_unknown_column_leaves_a_resolved_issue_resolved(self, store):
        issue = _seed(store, status="resolved")
        _, issue_store = store

        assert _move(store, "Finished") is True
        assert _status(issue_store, issue["id"]) == "resolved"

        assert _move(store, "Some Renamed Column") is True
        assert _status(issue_store, issue["id"]) == "resolved", (
            "a move to a column the table does not know must not rewrite the "
            "status to 'open' — that silently reopened resolved issues"
        )

    def test_opaque_frame_id_alone_leaves_the_status_alone(self, store):
        """Pre-fix messages carry only the uuid; they must not reopen either."""
        issue = _seed(store, status="in-progress")
        _, issue_store = store

        assert _move(store, None, frame_id="6f1c-uuid-not-a-title") is True
        assert _status(issue_store, issue["id"]) == "in-progress"

    def test_resolve_frame_status_reports_no_opinion(self, store):
        kb, _ = store
        assert kb._resolve_frame_status({"payload": {"frame_id": "uuid"}}) is None
        assert (
            kb._resolve_frame_status({"payload": {"frame_title": "Not A Column"}})
            is None
        )
        assert (
            kb._resolve_frame_status({"payload": {"frame_title": "Finished"}})
            == "resolved"
        )


# ---------------------------------------------------------------------------
# 3. A ledger record without a status key must not raise
# ---------------------------------------------------------------------------


class TestMissingStatusKeyIsTolerated:
    def test_move_on_record_without_status_applies(self, store):
        issue = _seed(store)
        _, issue_store = store
        ledger = issue_store.load_issues()
        issue_store.find_by_id(ledger, issue["id"]).pop("status")
        issue_store.save_issues(ledger)

        assert _move(store, "Finished") is True, (
            "a record without a 'status' key must not raise KeyError"
        )
        assert _status(issue_store, issue["id"]) == "resolved"
