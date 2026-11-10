"""Regression tests: resolving a card-attributable issue emits a child card.

Covers the new ``emit_child_card`` flag on ``issue_store.resolve`` and its
use by the human ``issue resolve`` command and the autonomous fix path.
"""

from __future__ import annotations

import harnessfix.issues as issue_store


class _FakeBridge:
    def __init__(self) -> None:
        self.ops: list[dict] = []

    def enqueue(self, msg: dict) -> int:
        self.ops.append(msg)
        return len(self.ops)


def _enable(monkeypatch, bridge):
    monkeypatch.setattr(issue_store, "_should_sync", lambda: True)
    monkeypatch.setattr(issue_store, "_get_kanban_bridge", lambda: bridge)


def test_resolve_default_does_not_emit_child_card(monkeypatch):
    bridge = _FakeBridge()
    _enable(monkeypatch, bridge)
    issues = []
    issue = issue_store.make_issue(
        "test", "Work", ["agent.py:1"], parent_card_id="card-1", emit_child_card=True
    )
    issues.append(issue)
    # drain create ops
    bridge.ops.clear()

    assert issue_store.resolve(issues, issue["id"], "resolved", note="done") is True
    child_ops = [m for m in bridge.ops if m.get("op") == "child_card_create"]
    assert child_ops == [], "default resolve must not emit a child card"


def test_resolve_emit_child_card_emits_when_card_attributable(monkeypatch):
    bridge = _FakeBridge()
    _enable(monkeypatch, bridge)
    issues = []
    issue = issue_store.make_issue(
        "test", "Work", ["agent.py:1"], parent_card_id="card-1", emit_child_card=True
    )
    issues.append(issue)
    bridge.ops.clear()

    assert issue_store.resolve(
        issues, issue["id"], "resolved", note="done", emit_child_card=True
    ) is True
    child_ops = [m for m in bridge.ops if m.get("op") == "child_card_create"]
    assert len(child_ops) >= 1
    child = child_ops[0]["payload"]
    assert child["parent_id"] == "card-1"
    assert child["issue_id"] == issue["id"]
    assert child["status"] == "resolved"
    # work summary must surface the note
    assert ("done" in child["text"]) or ("done" in child["title"])


def test_resolve_emit_child_card_noop_for_non_card_issue(monkeypatch):
    bridge = _FakeBridge()
    _enable(monkeypatch, bridge)
    issues = []
    issue = issue_store.make_issue(
        "test", "Plain", ["agent.py:1"], emit_child_card=True
    )
    issues.append(issue)
    bridge.ops.clear()

    issue_store.resolve(
        issues, issue["id"], "resolved", emit_child_card=True
    )
    child_ops = [m for m in bridge.ops if m.get("op") == "child_card_create"]
    assert child_ops == [], (
        "non-card issue must not emit a child card"
    )


def test_resolve_does_not_emit_on_already_resolved(monkeypatch):
    bridge = _FakeBridge()
    _enable(monkeypatch, bridge)
    issues = []
    issue = issue_store.make_issue(
        "test", "Work", ["agent.py:1"], parent_card_id="card-1", emit_child_card=True
    )
    issues.append(issue)
    # resolve once
    issue_store.resolve(issues, issue["id"], "resolved", emit_child_card=True)
    bridge.ops.clear()

    # resolve again (already resolved) must not spawn another child card
    issue_store.resolve(issues, issue["id"], "resolved", emit_child_card=True)
    child_ops = [m for m in bridge.ops if m.get("op") == "child_card_create"]
    assert child_ops == [], (
        "re-resolving must not emit a duplicate child card"
    )
