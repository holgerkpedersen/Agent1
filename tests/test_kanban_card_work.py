"""Regression: a Kanban "Work" request is consumed and claims the card's issue.

The Kanban side enqueues ``card_work`` (``POST /api/cards/<id>/work``).
Agent1 must:
  * claim the linked issue via ``claim_card_for_work`` when it is free;
  * refuse (and still consume the message) when a foreign session holds it;
  * consume an untracked card as a quiet no-op, never retry it.

Drives the real ``kanban_bridge.process_inbound`` path against a temp queue.
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


@pytest.fixture()
def data(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setattr(kb, "DATA_DIR", root)
    monkeypatch.setattr(kb, "QUEUE_AGENT1_TO_KANBAN", root / "queue-agent1-to-kanban")
    monkeypatch.setattr(kb, "QUEUE_KANBAN_TO_AGENT1", root / "queue-kanban-to-agent1")
    monkeypatch.setattr(kb, "CARD_ID_MAP_PATH", root / ".card_id_map.json")
    monkeypatch.setattr(kb, "DEAD_LETTER_DIR", root / "dead-letter")
    (root / "active.json").write_text(
        json.dumps({"active": "test-board"}), encoding="utf-8"
    )
    (root / "board-test-board.json").write_text(
        json.dumps({"frames": [], "cards": [], "sync": {"agent1": True}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(issue_store, "ISSUES_PATH", tmp_path / "issues.json")
    return root


def _queue(root: Path) -> Path:
    return root / "queue-kanban-to-agent1"


def _drain(root: Path, msg: dict) -> int:
    kb.enqueue(msg, _queue(root))
    return kb.process_inbound(_queue(root))


def _remaining(root: Path) -> list:
    """Messages the bridge's own reader still reports as pending.

    The queue never deletes lines; ``process_inbound`` advances the reader
    cursor. ``read_queue`` returns only what lies past that cursor, so an
    empty result means the message was consumed.
    """
    messages, _next_offset = kb.read_queue(_queue(root))
    return messages


def _linked_issue(card_id: str) -> str:
    issue = issue_store.make_issue(
        "bug", "boom", [f"kanban:{card_id}"], parent_card_id=card_id
    )
    issues = issue_store.load_issues()
    issue_store.upsert(issues, issue)
    issue_store.save_issues(issues)
    kb.link_ids(issue["id"], card_id)
    return issue["id"]


def test_card_work_claims_linked_issue_and_consumes(data, monkeypatch):
    calls: list[str] = []

    def fake_claim(issue_id):
        calls.append(issue_id)
        return True, "claimed"

    monkeypatch.setattr(issue_store, "claim_card_for_work", fake_claim)
    issue_id = _linked_issue("card-w1")

    applied = _drain(data, {
        "op": "card_work", "source_id": "card-w1", "payload": {"title": "t"},
    })
    assert applied == 1
    assert calls == [issue_id]
    assert _remaining(data) == []


def test_card_work_refused_claim_is_consumed_not_retried(data, monkeypatch):
    monkeypatch.setattr(
        issue_store, "claim_card_for_work",
        lambda issue_id: (False, "claimed by another session"),
    )
    _linked_issue("card-w2")

    applied = _drain(data, {
        "op": "card_work", "source_id": "card-w2", "payload": {},
    })
    assert applied == 1
    assert _remaining(data) == [], "a refused work request must not be re-queued"


def test_card_work_untracked_card_is_quiet_noop(data, monkeypatch):
    called: list[str] = []

    def fake_claim(issue_id):
        called.append(issue_id)
        return True, "claimed"

    monkeypatch.setattr(issue_store, "claim_card_for_work", fake_claim)

    applied = _drain(data, {
        "op": "card_work", "source_id": "never-bridged", "payload": {},
    })
    assert applied == 1
    assert called == [], "no claim may be attempted for an untracked card"
    assert _remaining(data) == []


def test_card_work_missing_source_id_is_rejected(data):
    applied = _drain(data, {"op": "card_work", "payload": {}})
    assert applied == 0
