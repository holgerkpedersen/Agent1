"""Regression tests for the child-card emission gating (Component 1).

The bridge mirror (Kanban→Agent1, created by ``kanban_bridge._apply_card_create``)
carries a ``parent_card_id`` + ``kanban:<card>`` location, so it looks
card-attributable. Emitting a ``child_card_create`` for the mirror would create a
spurious child card on every card-create echo — breaking the bidirectional
bridge (the cross-repo ``test_kanban_roundtrip`` echo guard asserts the board
stays at 1 card after an echo).

Child-card emission is therefore gated on a work-op flag (``make_issue(...,
emit_child_card)``): the bridge mirror passes ``False`` (not work), work issues
(``agent_core/commands/issue_cmd`` passes ``True``) emit when card-attributable,
and non-card issues never emit. Best-effort: a missing bridge degrades to no
child card while the parent issue op still syncs.
"""

from __future__ import annotations

import pytest

import harnessfix.issues as issue_store


class _FakeBridge:
    """Minimal Kanban bridge double that records enqueued ops."""

    def __init__(self) -> None:
        self.ops: list[dict] = []

    def enqueue(self, msg: dict) -> int:
        self.ops.append(msg)
        return len(self.ops)


def _enqueue_once(monkeypatch, bridge: _FakeBridge):
    """Enable sync and point the bridge at ``bridge``; returns nothing."""
    monkeypatch.setattr(issue_store, "_should_sync", lambda: True)
    monkeypatch.setattr(issue_store, "_get_kanban_bridge", lambda: bridge)


def _issue_ops(ops: list[dict], target: str) -> list[dict]:
    """Return the enqueued ops whose ``op`` matches ``target``."""
    return [msg for msg in ops if msg.get("op") == target]


def test_bridge_mirror_does_not_emit_child_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Kanban→Agent1 mirror is card-attributable but is NOT work, so it
    must not spawn a child card (this is the roundtrip echo regression guard)."""
    bridge = _FakeBridge()
    _enqueue_once(monkeypatch, bridge)

    mirror = issue_store.make_issue(
        "test",
        "Mirror issue",
        ["agent.py:1", "kanban:card-1"],
        parent_card_id="card-1",
    )
    assert mirror["parent_card_id"] == "card-1"

    assert len(_issue_ops(bridge.ops, "issue_create")) == 1
    assert _issue_ops(bridge.ops, "child_card_create") == [], (
        "bridge mirror must not emit a child card; it would create a spurious "
        "child on the card-create echo"
    )


def test_work_issue_with_card_emits_child_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A work issue (emit_child_card=True) attributable to a card emits a child
    card carrying parent_id + work metadata."""
    bridge = _FakeBridge()
    _enqueue_once(monkeypatch, bridge)

    issue = issue_store.make_issue(
        "test",
        "Work on card",
        ["agent.py:1"],
        parent_card_id="card-1",
        emit_child_card=True,
        evidence="fixed the bug",
        suggested_approach="patched",
    )

    child_ops = _issue_ops(bridge.ops, "child_card_create")
    assert len(child_ops) >= 1, (
        "a card-attributable work issue must emit at least one child card "
        "showing the work done on the parent card"
    )
    child = child_ops[0]["payload"]
    assert child["parent_id"] == "card-1"
    assert child["issue_id"] == issue["id"]
    assert child["title"], "child title must carry the work summary"
    assert child["text"], "child text must carry the work body"
    assert child["system"] == "harnessfix"


def test_non_card_issue_does_not_emit_child_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-card issue (no parent_card_id / kanban: location) stays as-is even
    with emit_child_card=True."""
    bridge = _FakeBridge()
    _enqueue_once(monkeypatch, bridge)

    issue_store.make_issue(
        "test", "Plain issue", ["agent.py:1"], emit_child_card=True
    )

    assert len(_issue_ops(bridge.ops, "issue_create")) == 1
    assert _issue_ops(bridge.ops, "child_card_create") == [], (
        "non-card issue must not emit a child card"
    )


def test_missing_bridge_degrades_to_no_child_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Best-effort: with no bridge, make_issue never raises and emits no
    child card (there is no bridge to enqueue the child to either); the parent
    op sync is skipped too because there is no bridge to enqueue to."""
    bridge = _FakeBridge()
    monkeypatch.setattr(issue_store, "_should_sync", lambda: True)
    monkeypatch.setattr(issue_store, "_get_kanban_bridge", lambda: None)

    issue_store.make_issue(
        "test", "Work", ["agent.py:1"], parent_card_id="card-1", emit_child_card=True
    )

    assert _issue_ops(bridge.ops, "child_card_create") == [], (
        "child-card emission must be skipped when no bridge is available"
    )


def test_child_card_payload_carries_work_summary_fields() -> None:
    """The child-card payload carries the work record so the board can show the
    work without the Agent1 ledger (Component 3 surface)."""
    payload = {
        "id": "iss-work",
        "status": "resolved",
        "evidence": "patched agent.py:12",
        "suggested_approach": "added guard",
    }
    # _work_summary_for_child_card builds title/text from the work record.
    summary = issue_store._work_summary_for_child_card(payload)
    assert "iss-work" in summary
    assert "resolved" in summary
    assert "patched agent.py:12" in summary
    assert "added guard" in summary
