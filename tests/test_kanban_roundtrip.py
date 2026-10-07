"""Cross-repo regression: a Kanban card mirrored to Agent1 must not echo back.

Both halves of the sync run for real here — Kanban's ``routes_cards`` POST
enqueue hook, Agent1's ``kanban_bridge.process_inbound`` (which calls
``issues.make_issue``), and Kanban's ``kanban_sync.process_inbound``.

Agent1's ``_apply_card_create`` turns an inbound card into an issue via
``make_issue()``, and ``make_issue()`` enqueues ``issue_create`` straight back
to Kanban.  Before the fix, Kanban happily created a *second* card for the
same issue and mapped it to that same issue id, so the next poll echoed again
— an unbounded card-creation loop.  This test pins the idempotency guard that
absorbs the echo.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest

AGENT1_ROOT = Path(__file__).resolve().parents[1]
KANBAN_ROOT = AGENT1_ROOT.parent / "Kanban"
sys.path.insert(0, str(AGENT1_ROOT))

if not (KANBAN_ROOT / "src" / "agent1" / "kanban_sync.py").exists():
    pytest.skip(f"Kanban checkout not found at {KANBAN_ROOT}", allow_module_level=True)

sys.path.insert(0, str(KANBAN_ROOT))

from src.agent1 import kanban_sync as ks  # noqa: E402
from src.agent1.app import create_app  # noqa: E402
from src.agent1.models import Frame  # noqa: E402

from harnessfix import kanban_bridge as kb  # noqa: E402
from harnessfix import issues as issue_store  # noqa: E402


@pytest.fixture()
def shared_queue(tmp_path, monkeypatch):
    """Point both sides' queue and id-map globals at one temp directory.

    These paths are computed at import time, so a test that forgot to patch
    them would read and write the real ``data/`` queue shared with the running
    app.  One shared ``.card_id_map.json`` is what lets the two sides resolve
    each other's ids at all.
    """
    data = tmp_path / "data"
    data.mkdir()
    a1_to_kb = data / "queue-agent1-to-kanban"
    kb_to_a1 = data / "queue-kanban-to-agent1"
    id_map = data / ".card_id_map.json"
    for mod in (kb, ks):
        monkeypatch.setattr(mod, "DATA_DIR", data)
        monkeypatch.setattr(mod, "QUEUE_AGENT1_TO_KANBAN", a1_to_kb)
        monkeypatch.setattr(mod, "QUEUE_KANBAN_TO_AGENT1", kb_to_a1)
        monkeypatch.setattr(mod, "CARD_ID_MAP_PATH", id_map)
        monkeypatch.setattr(mod, "DEAD_LETTER_DIR", data / "dead-letter")
    monkeypatch.setattr(issue_store, "ISSUES_PATH", tmp_path / "issues.json")
    return data


def _app_with_card(monkeypatch, tmp_path, title="Synced card"):
    """Build a real Kanban app, add one column, POST one card via HTTP.

    Returns ``(store, card_id, client)``.  The app is created with sync
    *disabled* so no background thread starts; the env var is switched on
    afterwards because both enqueue hooks read it at call time.
    """
    monkeypatch.setenv("KANBAN_SYNC_ENABLED", "0")
    working = tmp_path / "kanban-working"
    app = create_app(
        working_dir=str(working),
        catalog_dir=str(working / "boards"),
        registry_path=str(working / "active.json"),
        uploads_dir=str(working / "uploads"),
    )
    client = app.test_client()

    # The first request binds the live store onto the app's holder.
    client.get("/api/cards")
    store = app.config["LIVE_STORE_HOLDER"]["store"]
    frame = Frame(id=uuid.uuid4().hex, title="Issues")
    store.add_frame(frame)

    monkeypatch.setenv("KANBAN_SYNC_ENABLED", "1")

    resp = client.post(
        "/api/cards",
        json={"frame_id": frame.id, "title": title, "text": "", "system": "harnessfix"},
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert resp.status_code == 201, resp.get_data(as_text=True)
    return store, resp.get_json()["id"], client


def test_card_create_echo_does_not_duplicate_the_card(
    shared_queue, tmp_path, monkeypatch
):
    store, card_id, _ = _app_with_card(monkeypatch, tmp_path)

    # Kanban → Agent1: the card becomes an issue, mapped back to the card.
    assert kb.process_inbound() == 1
    issue_id = kb.resolve_id(card_id)
    assert issue_id, "card must be mapped to the issue it created"

    # Agent1 → Kanban: make_issue() enqueued issue_create for that same issue.
    assert ks.process_inbound(store) == 1

    cards = store.get_all_cards()
    assert len(cards) == 1, f"echo duplicated the card: {[c.title for c in cards]}"
    assert cards[0].id == card_id


def test_repeated_poll_cycles_do_not_grow_the_board(
    shared_queue, tmp_path, monkeypatch
):
    store, card_id, _ = _app_with_card(monkeypatch, tmp_path)

    for cycle in range(3):
        kb.process_inbound()
        ks.process_inbound(store)
        cards = store.get_all_cards()
        assert len(cards) == 1, f"board grew on cycle {cycle}: {len(cards)} cards"

    assert store.get_all_cards()[0].id == card_id


def test_round_trip_produces_no_dead_letters(shared_queue, tmp_path, monkeypatch):
    """A healthy round trip must not dead-letter anything on either side."""
    store, _, _ = _app_with_card(monkeypatch, tmp_path)

    kb.process_inbound()
    ks.process_inbound(store)

    dead_dir = shared_queue / "dead-letter"
    dead = list(dead_dir.glob("**/*")) if dead_dir.exists() else []
    assert dead == [], f"unexpected dead letters: {dead}"
