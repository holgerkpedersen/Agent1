"""Regression: GET /api/cards/linked returns exactly the linked card ids.

The endpoint derives its answer from the shared issue<->card id map
(``kanban_sync.CARD_ID_MAP_PATH``).  The map is bidirectional and carries
several kinds of entries at once:

* forward pairs ``iss-... -> card id`` (the links that count),
* the reverse keys ``card id -> iss-...`` (must not be double-counted),
* non-issue pairs (e.g. ``card:<title> -> card id`` written by the inspect
  helpers) whose *values* are card ids but which are not issue links, and
* stale entries pointing at cards that no longer exist on the board.

The endpoint must return each linked card id exactly once — reverse keys and
non-issue pairs excluded — and an empty list when the map file is missing.

The endpoint does not exist yet (no ``/api/cards/linked`` route anywhere in
the Kanban checkout), so every test here fails today; that is the point.
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

from src.agent1 import kanban_sync as ks  # noqa: E402
from src.agent1.app import create_app  # noqa: E402
from src.agent1.models import Card, Frame  # noqa: E402


@pytest.fixture()
def kanban_client(tmp_path, monkeypatch):
    """Real Kanban app on a temp working dir, with the id map redirected.

    ``CARD_ID_MAP_PATH`` is computed at import time from the module-level
    ``DATA_DIR``, so an unpatched test would read (and, on write paths,
    clobber) the real shared ``data/.card_id_map.json``.  Sync stays disabled
    so no poller thread starts and nothing is enqueued behind the tests'
    backs — these tests only exercise the read-only linked endpoint.
    """
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(ks, "DATA_DIR", data)
    monkeypatch.setattr(ks, "CARD_ID_MAP_PATH", data / ".card_id_map.json")
    monkeypatch.setenv("KANBAN_SYNC_ENABLED", "0")

    working = tmp_path / "kanban-working"
    app = create_app(
        working_dir=str(working),
        catalog_dir=str(working / "boards"),
        registry_path=str(working / "active.json"),
        uploads_dir=str(working / "uploads"),
    )
    client = app.test_client()

    # Seed the board with cards through the live store (the first request
    # binds it, mirroring tests/test_kanban_roundtrip.py).
    client.get("/api/cards")
    store = app.config["LIVE_STORE_HOLDER"]["store"]
    frame = Frame(id=uuid.uuid4().hex, title="Issues")
    store.add_frame(frame)
    for card_id in (
        "card-linked-a",
        "card-linked-b",
        "card-unlinked",
        "card-stale-target",  # linked in the map, deleted below
    ):
        store.add_card(
            Card(
                id=card_id,
                title=card_id,
                text="",
                frame_id=frame.id,
                tags=[],
                system="harnessfix",
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:00:00+00:00",
                status="open",
            ),
            frame.id,
        )
    store.delete_card("card-stale-target")

    # Mixed map: forward issue links, reverse keys, a non-issue pair whose
    # value is also a card id, and a stale link to the deleted card.
    (data / ".card_id_map.json").write_text(
        json.dumps(
            {
                "iss-alpha": "card-linked-a",
                "card-linked-a": "iss-alpha",          # reverse key
                "iss-beta": "card-linked-b",
                "card-linked-b": "iss-beta",           # reverse key
                "card:Unlinked title": "card-unlinked",  # not an issue link
                "iss-ghost": "card-stale-target",      # card no longer exists
            }
        ),
        encoding="utf-8",
    )
    return client


def test_linked_returns_exactly_linked_card_ids(kanban_client):
    resp = kanban_client.get("/api/cards/linked")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    body = resp.get_json()
    assert isinstance(body, list), f"expected a JSON list, got {type(body)}: {body!r}"
    assert sorted(body) == ["card-linked-a", "card-linked-b"], (
        "linked endpoint must return each issue-linked card id exactly once: "
        "reverse keys, non-issue pairs and stale links must be excluded"
    )


def test_linked_ids_are_deduplicated(kanban_client):
    """A card linked under two issue keys appears exactly once."""
    map_path = ks.CARD_ID_MAP_PATH
    mapping = json.loads(map_path.read_text(encoding="utf-8"))
    mapping["iss-duplicate-of-alpha"] = "card-linked-a"
    map_path.write_text(json.dumps(mapping), encoding="utf-8")

    body = kanban_client.get("/api/cards/linked").get_json()
    assert isinstance(body, list)
    assert sorted(body) == ["card-linked-a", "card-linked-b"]
    assert len(body) == len(set(body)), f"duplicate card ids in response: {body!r}"


def test_linked_is_empty_when_map_file_missing(kanban_client):
    ks.CARD_ID_MAP_PATH.unlink()
    assert ks.CARD_ID_MAP_PATH.exists() is False

    resp = kanban_client.get("/api/cards/linked")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert resp.get_json() == []
