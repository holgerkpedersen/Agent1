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

import json
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

# Kanban's app enforces a double-submit CSRF token on every /api/* write
# (POST/PUT/DELETE).  Under TESTING it uses the fixed token from the Kanban
# test conftest, so this cross-repo harness sends that header — the guard
# itself stays live and keeps rejecting requests without it.
KANBAN_TEST_CSRF_TOKEN = "test-csrf-token"

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

    # The shared temp dir stands in for the Kanban app's data directory. Give
    # it an active board that has opted into agent1 sync — otherwise Agent1's
    # per-board gate (issues._should_sync layer 2) fails closed and no echo is
    # ever enqueued, which would collapse the round trip under test.
    (data / "active.json").write_text(
        json.dumps({"active": "test-board"}), encoding="utf-8"
    )
    (data / "board-test-board.json").write_text(
        json.dumps({"frames": [], "cards": [], "sync": {"agent1": True}}),
        encoding="utf-8",
    )

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
    # The CSRF guard compares the X-CSRF-Token header against the per-request
    # token in g.csrf_token; under TESTING that is app.config["TEST_CSRF_TOKEN"]
    # (the fixed value Kanban's own tests use).  Without these two config keys
    # every write here would be rejected with 403.
    app.config["TESTING"] = True
    app.config["TEST_CSRF_TOKEN"] = KANBAN_TEST_CSRF_TOKEN
    client = app.test_client()

    # The first request binds the live store onto the app's holder.
    client.get("/api/cards")
    store = app.config["LIVE_STORE_HOLDER"]["store"]

    # Opt this board in to agent1 sync so the Kanban-side outbound gate opens
    # (mirrors what the UI toggle persists). Without it routes_cards._should_
    # sync() keeps every enqueue closed and no card ever reaches Agent1.
    store.set_sync_settings({"agent1": True})

    frame = Frame(id=uuid.uuid4().hex, title="Issues")
    store.add_frame(frame)

    monkeypatch.setenv("KANBAN_SYNC_ENABLED", "1")

    resp = client.post(
        "/api/cards",
        json={"frame_id": frame.id, "title": title, "text": "", "system": "harnessfix"},
        headers={
            "X-Requested-With": "XMLHttpRequest",
            "X-CSRF-Token": KANBAN_TEST_CSRF_TOKEN,
        },
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


def test_card_create_persists_the_issue_it_maps(shared_queue, tmp_path, monkeypatch):
    """An inbound card must land in the ledger, not just in the id map.

    ``issues.make_issue()`` only builds the dict; the repo-scan collector does
    the ``upsert``.  ``_apply_card_create`` used to link the card to that
    unsaved issue, leaving a phantom mapping: every later card_update/move
    resolved to an id present nowhere and logged "no matching issue".
    """
    _, card_id, _ = _app_with_card(monkeypatch, tmp_path)

    assert kb.process_inbound() == 1
    issue_id = kb.resolve_id(card_id)
    assert issue_id, "card must be mapped to the issue it created"

    ledger = issue_store.load_issues()
    assert issue_store.find_by_id(ledger, issue_id) is not None, (
        f"id map claims {issue_id} <-> {card_id} but the issue is absent "
        f"from the ledger ({len(ledger)} entries)"
    )


def test_move_to_finished_resolves_the_issue(shared_queue, tmp_path, monkeypatch):
    """Moving a card to the "Finished" column must resolve its issue.

    The Kanban side sends an opaque frame *id*; ``FRAME_TO_STATUS`` is keyed by
    the column *title*, so resolving the id alone never hit and every move fell
    back to "open" — silently reopening resolved issues.
    """
    store, card_id, client = _app_with_card(monkeypatch, tmp_path)

    assert kb.process_inbound() == 1
    issue_id = kb.resolve_id(card_id)
    assert issue_id

    finished = Frame(id=uuid.uuid4().hex, title="Finished")
    store.add_frame(finished)

    resp = client.post(
        f"/api/cards/{card_id}/move",
        json={"frame_id": finished.id},
        headers={
            "X-Requested-With": "XMLHttpRequest",
            "X-CSRF-Token": KANBAN_TEST_CSRF_TOKEN,
        },
    )
    assert resp.status_code == 200, resp.get_data(as_text=True)

    assert kb.process_inbound() == 1
    issue = issue_store.find_by_id(issue_store.load_issues(), issue_id)
    assert issue is not None
    assert issue["status"] == "resolved", (
        f"move to Finished left status={issue['status']!r} (expected 'resolved')"
    )
    assert issue["resolved_at"], "a resolved issue must carry resolved_at"


# --- Untracked cards: permanent misses are not failures -------------------------
#
# A card created directly on the Kanban board was never bridged to an Agent1
# issue, so it has no entry in .card_id_map.json.  Its update/move/delete
# messages can therefore NEVER resolve — a permanent condition.  The bridge
# used to treat every miss the same: warning + False, which re-queued each
# message three times toward dead-letter (the churn behind issue
# "Error when moving a Kanban Card").  A permanent, benign miss is consumed
# as a quiet no-op; only mapped-but-missing stays a loud anomaly.

_UNTRACKED = "02e7042697b64b34af4fb78adf387a27"  # the real-world card guid


def _drain_inbound(shared_queue, msg: dict) -> int:
    """Enqueue one inbound message, process it once, return applied count."""
    kb.enqueue(msg, shared_queue / "queue-kanban-to-agent1")
    return kb.process_inbound(shared_queue / "queue-kanban-to-agent1")


@pytest.mark.parametrize(
    "op,payload",
    [
        ("card_update", {"title": "edited on the board"}),
        ("card_move", {"frame_title": "On-going"}),
        ("card_delete", {}),
    ],
    ids=["update", "move", "delete"],
)
def test_untracked_card_op_is_a_quiet_noop(
    shared_queue, tmp_path, monkeypatch, op, payload
):
    """A follow-up for a never-bridged card is consumed, not churned."""
    applied = _drain_inbound(
        shared_queue, {"op": op, "source_id": _UNTRACKED, "payload": payload}
    )
    assert applied == 1, f"{op}: untracked card must be consumed as a no-op"

    queue_dir = shared_queue / "queue-kanban-to-agent1"
    lines = [
        ln
        for ln in (queue_dir / "messages.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert len(lines) == 1, f"{op}: message was re-queued for retry: {lines}"

    dead_dir = shared_queue / "dead-letter"
    dead = list(dead_dir.glob("**/*")) if dead_dir.exists() else []
    assert dead == [], f"{op}: untracked-card message dead-lettered: {dead}"


def test_mapped_card_move_still_resolves_the_issue(shared_queue, tmp_path, monkeypatch):
    """The no-op path must not swallow tracked cards — the move still lands."""
    store, card_id, client = _app_with_card(monkeypatch, tmp_path)

    assert kb.process_inbound() == 1
    issue_id = kb.resolve_id(card_id)
    assert issue_id

    finished = Frame(id=uuid.uuid4().hex, title="Finished")
    store.add_frame(finished)
    resp = client.post(
        f"/api/cards/{card_id}/move",
        json={"frame_id": finished.id},
        headers={
            "X-Requested-With": "XMLHttpRequest",
            "X-CSRF-Token": KANBAN_TEST_CSRF_TOKEN,
        },
    )
    assert resp.status_code == 200, resp.get_data(as_text=True)

    assert kb.process_inbound() == 1
    issue = issue_store.find_by_id(issue_store.load_issues(), issue_id)
    assert issue is not None and issue["status"] == "resolved"


def test_mapped_but_missing_issue_is_a_loud_failure(
    shared_queue, tmp_path, monkeypatch
):
    """A mapping whose issue vanished from the ledger is a real anomaly: it
    must keep failing loudly (warning + retry), never be silently consumed."""
    store, card_id, _ = _app_with_card(monkeypatch, tmp_path)
    assert kb.process_inbound() == 1
    issue_id = kb.resolve_id(card_id)
    assert issue_id

    # Sabotage the ledger: the mapping now points at an absent issue.
    issues = issue_store.load_issues()
    issues[:] = [i for i in issues if i["id"] != issue_id]
    issue_store.save_issues(issues)

    applied = _drain_inbound(
        shared_queue,
        {"op": "card_move", "source_id": card_id, "payload": {"frame_title": "Issues"}},
    )
    assert applied == 0, (
        "mapped-but-missing is a genuine anomaly — it must not be consumed "
        "as a quiet no-op"
    )
