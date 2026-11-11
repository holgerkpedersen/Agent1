"""Read-only Kanban ↔ Agent1 inspector — the key-resolution and queue view.

``kanban_bridge`` moves messages; this module *explains* them. It answers the
three questions the bridge itself never reports:

* **Which card is this issue, and which issue is this card?**  The link lives
  in two places (``.card_id_map.json`` as a cache, ``locations["kanban:<id>"]``
  on the ledger record as the durable copy) and the two drift.  ``resolve()``
  reports every hop it used, so a human can see *why* a card resolved the way
  it did — or why it resolved to nothing.
* **What is on the board right now?**  Frames (columns), their titles, and the
  cards in them, read straight from ``board-<id>.json``.
* **What is sitting in the queues, unprocessed?**  Both directions, with
  dead-letter counts.

It also fills the one real gap in the inbound mapping: a Kanban card's
``text`` (description) never reached an Agent1 issue, because the bridge reads
the description out of ``[component:]``/``[note:]``/``[status:]`` tags and a
plain description has none.  ``issue_view()`` composes the **full** view —
ledger fields plus the card's raw description and column — and
``sync_descriptions()`` writes that text into issues whose ledger fields are
empty, so ``kanban → agent1`` carries the description instead of dropping it.

Nothing here writes to a queue.  The only mutation is the explicit, opt-in
:func:`sync_descriptions`.

CLI::

    python -m harnessfix.kanban_inspect status
    python -m harnessfix.kanban_inspect card <card-id-or-issue-id> [--json]
    python -m harnessfix.kanban_inspect queue [--limit N]
    python -m harnessfix.kanban_inspect board [board-id]
    python -m harnessfix.kanban_inspect sync-descriptions [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import logging

from . import kanban_bridge as kb
from . import issues as issue_store

logger = logging.getLogger(__name__)

# --- Kanban-side status table -------------------------------------------------
#
# The Kanban app validates card status against its own frozenset
# (``src/agent1/models.py:VALID_STATUSES``):
#     open, planned, in-progress, reviewing, done
# Agent1's ledger uses the same words except ``resolved`` for the terminal
# state.  The table below is the bridge's contract; it is asserted against the
# Kanban source by tests/test_kanban_status_contract.py so a rename on either
# side fails a test instead of silently dropping statuses.

#: Kanban card status → Agent1 issue status.
CARD_STATUS_TO_ISSUE: dict[str, str] = {
    "open": "open",
    "planned": "planned",
    "in-progress": "in-progress",
    "reviewing": "reviewing",
    "done": "resolved",
}

#: Agent1 issue status → Kanban card status (the inverse, for outbound reads).
ISSUE_STATUS_TO_CARD: dict[str, str] = {
    "open": "open",
    "planned": "planned",
    "in-progress": "in-progress",
    "reviewing": "reviewing",
    "resolved": "done",
    "wontfix": "done",
}

#: Kanban column title → Agent1 issue status.  Mirrors
#: ``kanban_bridge.FRAME_TO_STATUS``; asserted equal by the contract test.
COLUMN_TO_ISSUE: dict[str, str] = dict(kb.FRAME_TO_STATUS)


# --- key anatomy --------------------------------------------------------------


def key_anatomy() -> list[dict[str, str]]:
    """The keys that flow between the two repos, and what each one means.

    Hard-coded on purpose: this is documentation of a stable contract, and a
    test asserts every described path exists as a constant in the bridge.
    """
    return [
        {
            "key": "card id (source_id)",
            "where": "Kanban",
            "shape": "uuid4 hex, 32 chars",
            "meaning": "A card on a Kanban board. Travels as ``source_id`` in "
            "every inbound message and inside ``locations`` as "
            "``kanban:<card-id>``.",
        },
        {
            "key": "issue id",
            "where": "Agent1 (.issues.json)",
            "shape": "iss-<category>-<digest>, e.g. iss-9, iss-kanban-a1b2c3d4",
            "meaning": "An Agent1 work item. Travels as ``source_id`` in every "
            "outbound message (issue_create / issue_update / issue_resolve).",
        },
        {
            "key": ".card_id_map.json",
            "where": f"{kb.DATA_DIR}",
            "shape": "flat dict, BOTH directions: card→issue and issue→card",
            "meaning": "A CACHE of the link. Written by ``link_ids`` (both "
            "keys at once), read by ``resolve_id``. It drifts — stale keys "
            "outlive deleted cards — so it is never the only lookup.",
        },
        {
            "key": "locations: ['kanban:<card-id>']",
            "where": "Agent1 (.issues.json)",
            "shape": "list of strings on the issue record",
            "meaning": "The DURABLE copy of the link. Survives map loss; a hit "
            "here heals the map (see ``_lookup_issue_for``).",
        },
        {
            "key": "frame id / frame_title",
            "where": "Kanban board file",
            "shape": "uuid4 hex / human column title",
            "meaning": "A column. Status is keyed by TITLE, never by id — a "
            "uuid can't be looked up in FRAME_TO_STATUS, so ``card_move`` "
            "carries the title next to the id.",
        },
        {
            "key": "seq / offset",
            "where": "<queue>/seq.txt, <queue>/offset.txt",
            "shape": "integer counters",
            "meaning": "``seq.txt`` is the WRITER's counter; ``offset.txt`` is "
            "the READER's cursor. They are separate files on purpose: sharing "
            "them made each side overwrite the other's progress.",
        },
        {
            "key": "active.json → board-<id>.json",
            "where": f"{kb.DATA_DIR}",
            "shape": '{"active": "<catalog-id>"} / {"frames":[…], "sync":{…}}',
            "meaning": "Which board is live, and whether it opted in to "
            "agent1 sync (``sync.agent1``). Both sides read these; sync is "
            "fail-closed when they are missing.",
        },
    ]


# --- board reading ------------------------------------------------------------


def _map_issue_id(key: str) -> Optional[str]:
    """The id map's answer only — no board fallback, so it cannot recurse."""
    if not key:
        return None
    try:
        return kb.resolve_id(key, kb._load_id_map())
    except Exception as exc:  # noqa: BLE001 — inspection must never raise
        logger.warning("kanban_inspect: id map lookup failed for %s: %s", key, exc)
        return None


def _board_root() -> Path:
    return kb.DATA_DIR


def active_board_id() -> Optional[str]:
    """Catalog id of the board the Kanban app has open (``active.json``)."""
    try:
        data = json.loads((_board_root() / "active.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    active = data.get("active") if isinstance(data, dict) else None
    return active if isinstance(active, str) and active else None


def board_ids() -> list[str]:
    """Catalog board ids: the seed files in ``data/boards`` plus working copies."""
    root = _board_root()
    found: set[str] = set()
    for path in sorted((root / "boards").glob("*.json")):
        found.add(path.stem)
    for path in sorted(root.glob("board-*.json")):
        found.add(path.name[len("board-") : -len(".json")])
    return sorted(found)


def load_board(board_id: str) -> dict[str, Any]:
    """A board's working copy (``data/board-<id>.json``), or ``{}`` if absent."""
    try:
        data = json.loads(
            (_board_root() / f"board-{board_id}.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def board_view(board_id: Optional[str] = None) -> dict[str, Any]:
    """Frames with their titles and cards — the shape a human reads a board in."""
    bid = board_id or active_board_id()
    board = load_board(bid) if bid else {}
    frames = [f for f in board.get("frames", []) if isinstance(f, dict)]
    cards = board.get("cards")
    card_map: dict[str, dict[str, Any]] = {}
    if isinstance(cards, dict):
        card_map = {k: v for k, v in cards.items() if isinstance(v, dict)}
    elif isinstance(cards, list):
        card_map = {
            c["id"]: c
            for c in cards
            if isinstance(c, dict) and isinstance(c.get("id"), str)
        }

    out_frames = []
    for frame in frames:
        ids = frame.get("card_ids")
        owned = (
            [card_map[cid] for cid in ids if cid in card_map]
            if isinstance(ids, list)
            else [c for c in card_map.values() if c.get("frame_id") == frame.get("id")]
        )
        out_frames.append(
            {
                "frame_id": frame.get("id", ""),
                "title": frame.get("title", ""),
                "issue_status": COLUMN_TO_ISSUE.get(frame.get("title", "")),
                "cards": [
                    {
                        "id": c.get("id", ""),
                        "title": c.get("title", ""),
                        "status": c.get("status", ""),
                        "text": c.get("text", ""),
                        # ``_map_issue_id`` only, NOT resolve(): resolve()
                        # falls back to a board title match, which calls
                        # board_view() again — an infinite loop.
                        "linked_issue": _map_issue_id(c.get("id", "")),
                    }
                    for c in owned
                ],
            }
        )

    return {
        "board_id": bid,
        "active_board_id": active_board_id(),
        "path": str(_board_root() / f"board-{bid}.json") if bid else "",
        "sync_enabled": bool((board.get("sync") or {}).get("agent1")),
        "frames": out_frames,
        "card_count": len(card_map),
    }


def find_card(board: dict[str, Any], key: str) -> Optional[dict[str, Any]]:
    """A card from a board view by id, title (exact), or unique title prefix.

    Matching is case-insensitive on the title. The key that arrives here is
    what a human typed at a prompt — ``"refactor the queue reader"``, not a
    guid — and a capitalised card title must not be the reason a lookup fails.
    """
    cards = [c for frame in board["frames"] for c in frame["cards"]]
    for card in cards:
        if card.get("id") == key:
            return card
    needle = str(key).strip().lower()
    if not needle:
        return None
    exact = [c for c in cards if str(c.get("title", "")).strip().lower() == needle]
    if len(exact) == 1:
        return exact[0]
    prefix = [
        c
        for c in cards
        if str(c.get("title", "")).strip().lower().startswith(needle)
    ]
    if len(prefix) == 1:
        return prefix[0]
    return None


# --- key resolution -----------------------------------------------------------


def resolve(key: str) -> dict[str, Any]:
    """Resolve any bridge key to both sides, reporting every hop that was used.

    Accepts a Kanban card id, an Agent1 issue id, a card title, or a unique
    title prefix (the last three only for the active board — ids are what the
    queues carry).
    """
    report: dict[str, Any] = {
        "input": key,
        "kind": None,
        "card_id": None,
        "issue_id": None,
        "hops": [],
        "warnings": [],
    }
    if not key:
        report["warnings"].append("empty key")
        return report

    issues = issue_store.load_issues()
    id_map = kb._load_id_map()

    direct_issue = issue_store.find_by_id(issues, key)
    if direct_issue is not None:
        report["kind"] = "issue"
        report["issue_id"] = key
        mapped_card = kb.resolve_id(key, id_map)
        report["hops"].append(f"ledger record {key} exists")
        if mapped_card:
            report["card_id"] = mapped_card
            report["hops"].append(
                f".card_id_map.json: issue {key} → card {mapped_card}"
            )
        else:
            locs = [
                loc[len("kanban:") :]
                for loc in (direct_issue.get("locations") or [])
                if isinstance(loc, str) and loc.startswith("kanban:")
            ]
            if locs:
                report["card_id"] = locs[0]
                report["hops"].append(
                    f"issue locations → card {locs[0]} (no map entry — map is stale)"
                )
        return report

    mapped_issue = kb.resolve_id(key, id_map)
    if mapped_issue:
        report["kind"] = "card"
        report["card_id"] = key
        report["issue_id"] = mapped_issue
        report["hops"].append(f".card_id_map.json: card {key} → issue {mapped_issue}")
        if issue_store.find_by_id(issues, mapped_issue) is None:
            report["warnings"].append(
                f"map points at issue {mapped_issue}, which is NOT in .issues.json "
                "(broken link — inbound messages for this card will fail)"
            )
        return report

    recovered = kb._find_issue_by_card_location(issues, key)
    if recovered is not None:
        report["kind"] = "card"
        report["card_id"] = key
        report["issue_id"] = recovered["id"]
        report["hops"].append(
            f"ledger locations 'kanban:{key}' → issue {recovered['id']}"
        )
        report["warnings"].append(
            "the link was recovered from the ledger; .card_id_map.json has no "
            "entry for this card (run `issue`/bridge traffic to heal it)"
        )
        return report

    board = board_view()
    card = find_card(board, key)
    if card is not None:
        report["kind"] = "card"
        report["card_id"] = card["id"]
        report["card_title"] = card.get("title", "")
        report["hops"].append(f"active board title match for {key!r}")
        report["warnings"].append(
            "the card is on the board but the bridge has never linked it to an "
            "issue — inbound update/move/delete messages for it are no-ops"
        )
        return report

    report["warnings"].append(
        f"{key!r} is not a card id, an issue id, a linked card, or a card title "
        "on the active board"
    )
    return report


# --- combined views -----------------------------------------------------------


def _card_from_board(card_id: str) -> Optional[dict[str, Any]]:
    board = board_view()
    for frame in board["frames"]:
        for card in frame["cards"]:
            if card.get("id") == card_id:
                return {**card, "frame_title": frame.get("title", "")}
    return None


def issue_view(key: str) -> dict[str, Any]:
    """Everything known about one work item: ledger record + card + link + queues.

    The Kanban card's ``text`` is the description a human typed on the board.
    The bridge maps it to ``evidence``/``suggested_approach`` only when tagged;
    this view shows the raw text too, so nothing is hidden from the reader.
    """
    link = resolve(key)
    out: dict[str, Any] = {
        "link": link,
        "issue": None,
        "card": None,
        "queue": [],
        "work_log": [],
    }

    if link.get("issue_id"):
        issue = issue_store.find_by_id(issue_store.load_issues(), link["issue_id"])
        if issue is not None:
            out["issue"] = issue
            out["issue"]["column_status"] = COLUMN_TO_ISSUE.inverse.get(
                issue.get("status", ""), None
            ) if hasattr(COLUMN_TO_ISSUE, "inverse") else _card_status_for(issue)

    card_id = link.get("card_id")
    if card_id:
        card = _card_from_board(card_id)
        if card is not None:
            out["card"] = card
        else:
            link["warnings"].append(
                f"card {card_id} is linked but is not on the active board "
                "(deleted, or another board is active)"
            )
        out["work_log"] = work_log(card_id)

    for direction, queue_dir in (
        ("kanban→agent1", kb.QUEUE_KANBAN_TO_AGENT1),
        ("agent1→kanban", kb.QUEUE_AGENT1_TO_KANBAN),
    ):
        for msg in pending_messages(queue_dir):
            if msg.get("source_id") in (card_id, link.get("issue_id")):
                out["queue"].append({"direction": direction, "message": msg})
    return out


def work_log(card_id: str) -> list[dict[str, Any]]:
    """The chronological trail of work done on *card_id*, from the ledger.

    Each entry is a compact record — id, title, status, evidence, approach,
    timestamps — ordered oldest first, so ``kanban card <id>`` can answer
    "what did the Agent1 session do to this card?" without reading
    ``.issues.json`` by hand.  Empty when the card has no linked work.
    """
    if not card_id:
        return []
    records = issue_store.work_log_for_card(issue_store.load_issues(), card_id)
    out: list[dict[str, Any]] = []
    for it in records:
        out.append(
            {
                "issue_id": it.get("id"),
                "title": it.get("title"),
                "status": it.get("status"),
                "evidence": it.get("evidence") or "",
                "suggested_approach": it.get("suggested_approach") or "",
                "created_at": it.get("created_at"),
                "resolved_at": it.get("resolved_at"),
            }
        )
    return out


def _card_status_for(issue: dict[str, Any]) -> Optional[str]:
    return ISSUE_STATUS_TO_CARD.get(str(issue.get("status", "")))


def composed_description(issue: dict[str, Any], card: dict[str, Any] | None) -> str:
    """The full description of a work item: card text + every ledger field.

    This is the answer to "what does the card say?" — the bridge stores card
    ``text`` as ``evidence`` only when tagged, so reading one field is not
    enough.  Deduplicates lines that appear in both places.
    """
    parts: list[str] = []
    if card and card.get("text"):
        parts.append(str(card["text"]).strip())
    for label, field in (
        ("suggested approach", "suggested_approach"),
        ("evidence", "evidence"),
        ("decision ref", "decision_ref"),
    ):
        value = str(issue.get(field) or "").strip()
        if not value:
            continue
        for existing in list(parts):
            if value == existing:
                value = ""
                break
        if value:
            parts.append(f"{label}: {value}")
    return "\n".join(p for p in parts if p)


# --- queue inspection ---------------------------------------------------------


def pending_messages(queue_dir: Path, *, limit: int = 50) -> list[dict[str, Any]]:
    """Messages the reader of *queue_dir* has not consumed yet."""
    try:
        messages, _offset = kb.read_queue(Path(queue_dir))
    except Exception as exc:  # noqa: BLE001 — inspection must never raise
        logger.warning("kanban_inspect: cannot read %s: %s", queue_dir, exc)
        return []
    return messages[:limit]


def dead_letters(queue_dir: Path) -> list[dict[str, Any]]:
    """Quarantined messages for *queue_dir* (dead-letter directory)."""
    out: list[dict[str, Any]] = []
    try:
        dead_dir = kb._dead_letter_dir(Path(queue_dir))
    except Exception:  # noqa: BLE001
        return out
    if not dead_dir.exists():
        return out
    for path in sorted(dead_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                out.append({"unparsed_line": line, "file": str(path)})
    return out


def queue_status(limit: int = 10) -> dict[str, Any]:
    """Both queues: pending counts, unread messages, dead letters."""
    directions = (
        ("kanban→agent1", kb.QUEUE_KANBAN_TO_AGENT1),
        ("agent1→kanban", kb.QUEUE_AGENT1_TO_KANBAN),
    )
    out: dict[str, Any] = {"data_dir": str(kb.DATA_DIR), "queues": []}
    for name, queue_dir in directions:
        pending = pending_messages(queue_dir, limit=limit)
        dead = dead_letters(queue_dir)
        all_pending = pending_messages(queue_dir, limit=10_000)
        out["queues"].append(
            {
                "direction": name,
                "path": str(queue_dir),
                "pending": len(all_pending),
                "retries_pending": sum(
                    1 for m in all_pending if int(m.get("retry_count") or 0) > 0
                ),
                "messages": [
                    {
                        "seq": m.get("seq"),
                        "op": m.get("op"),
                        "source_id": m.get("source_id"),
                        "resolved_issue": resolve(str(m.get("source_id", "")))[
                            "issue_id"
                        ],
                        "retry_count": m.get("retry_count", 0),
                        "ts": m.get("ts"),
                    }
                    for m in pending
                ],
                "dead_letters": len(dead),
            }
        )
    return out


def map_status() -> dict[str, Any]:
    """Health of ``.card_id_map.json`` against the ledger and the active board."""
    id_map = kb._load_id_map()
    issues = issue_store.load_issues()
    issue_ids = {i.get("id") for i in issues}
    board = board_view()
    live_cards = {c["id"] for frame in board["frames"] for c in frame["cards"]}

    card_keys = [k for k in id_map if not str(k).startswith("iss-")]
    issue_keys = [k for k in id_map if str(k).startswith("iss-")]
    return {
        "path": str(kb.CARD_ID_MAP_PATH),
        "entries": len(id_map),
        "card_keys": len(card_keys),
        "issue_keys": len(issue_keys),
        "stale_card_keys": sum(1 for k in card_keys if k not in live_cards),
        "issue_keys_missing_from_ledger": sum(
            1 for k in issue_keys if k not in issue_ids
        ),
        "issues_linked_by_location_only": sum(
            1
            for i in issues
            if any(
                isinstance(loc, str) and loc.startswith("kanban:")
                for loc in (i.get("locations") or [])
            )
            and kb.resolve_id(str(i.get("id")), id_map) is None
        ),
    }


# --- description sync ---------------------------------------------------------


def sync_descriptions(*, dry_run: bool = False) -> list[dict[str, Any]]:
    """Copy the Kanban card description into issues that have none.

    The inbound bridge maps a card's ``text`` to ledger fields only through
    tags (``[component:]`` / ``[note:]`` / ``[status:]``), so a plain
    description written on the board never reaches the issue — the gap this
    closes.  Rules, in order:

    * an issue is only touched when it is linked to a card that is on the
      active board;
    * the card's raw ``text`` goes into ``suggested_approach`` when that field
      is empty (it is the "what to do" field), otherwise into ``evidence`` when
      that is empty;
    * an issue with BOTH fields filled is left alone — human text wins, and a
      re-run stays idempotent;
    * nothing is written when ``dry_run`` is set.

    Returns one entry per issue considered, with the action taken.
    """
    board = board_view()
    cards_by_id = {
        c["id"]: {**c, "frame_title": f.get("title", "")}
        for f in board["frames"]
        for c in f["cards"]
    }
    issues = issue_store.load_issues()
    results: list[dict[str, Any]] = []
    changed = 0

    for issue in issues:
        locations = [
            loc[len("kanban:") :]
            for loc in (issue.get("locations") or [])
            if isinstance(loc, str) and loc.startswith("kanban:")
        ]
        card_id = locations[0] if locations else kb.resolve_id(str(issue.get("id")))
        card = cards_by_id.get(card_id) if card_id else None
        if card is None:
            continue
        text = str(card.get("text") or "").strip()
        if not text:
            continue

        target: Optional[str] = None
        if not str(issue.get("suggested_approach") or "").strip():
            target = "suggested_approach"
        elif not str(issue.get("evidence") or "").strip():
            target = "evidence"
        if target is None:
            results.append(
                {"id": issue.get("id"), "action": "skipped", "reason": "fields full"}
            )
            continue
        if text in (
            str(issue.get("suggested_approach") or ""),
            str(issue.get("evidence") or ""),
        ):
            # Already carried — a second run must be a no-op, not a copy into
            # the next empty field.
            results.append(
                {
                    "id": issue.get("id"),
                    "action": "skipped",
                    "reason": "description already present",
                }
            )
            continue

        if not dry_run:
            issue[target] = text
        changed += 1
        results.append(
            {
                "id": issue.get("id"),
                "card_id": card_id,
                "action": "dry-run" if dry_run else "updated",
                "field": target,
                "chars": len(text),
            }
        )

    if changed and not dry_run:
        issue_store.save_issues(issues)
    return results


# --- human summary ------------------------------------------------------------


def status_summary() -> dict[str, Any]:
    """One screen: config, board, map health, both queues."""
    return {
        "sync_enabled": kb.sync_enabled(),
        "active_board_allows_agent1": kb.active_board_allows_agent1(),
        "data_dir": str(kb.DATA_DIR),
        "active_board": active_board_id(),
        "boards": board_ids(),
        "inbound_processor_running": kb.inbound_processor_running(),
        "issues_total": len(issue_store.load_issues()),
        "map": map_status(),
        "queues": queue_status()["queues"],
    }


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnessfix.kanban_inspect",
        description="Read-only Kanban ↔ Agent1 inspector (keys, board, queues).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="sync config, board, map health, queue depths")

    p_card = sub.add_parser(
        "card", help="resolve a card/issue key and show both sides"
    )
    p_card.add_argument(
        "key", help="card id, issue id, or card title (prefix ok)"
    )
    p_card.add_argument(
        "--full", action="store_true", help="include the full ledger record"
    )

    p_queue = sub.add_parser("queue", help="pending messages in both directions")
    p_queue.add_argument("--limit", type=int, default=10)

    p_board = sub.add_parser("board", help="frames, titles and cards of a board")
    p_board.add_argument("board_id", nargs="?")

    sub.add_parser(
        "keys", help="explain every key that flows between the repos"
    )

    p_sync = sub.add_parser(
        "sync-descriptions", help="copy card descriptions into issues that have none"
    )
    p_sync.add_argument("--dry-run", action="store_true")

    args, rest = parser.parse_known_args(argv)

    if args.cmd == "status":
        _print(status_summary())
    elif args.cmd == "keys":
        _print(key_anatomy())
    elif args.cmd == "card":
        view = issue_view(args.key)
        if not args.full and view.get("issue"):
            view["issue"] = {
                k: v
                for k, v in view["issue"].items()
                if k in ("id", "title", "status", "severity", "autonomy_level")
            }
        view["description"] = composed_description(
            view.get("issue") or {}, view.get("card")
        )
        _print(view)
    elif args.cmd == "queue":
        _print(queue_status(limit=args.limit))
    elif args.cmd == "board":
        _print(board_view(args.board_id))
    elif args.cmd == "sync-descriptions":
        results = sync_descriptions(dry_run=args.dry_run)
        _print(
            {
                "dry_run": args.dry_run,
                "updated": sum(1 for r in results if r["action"] not in ("skipped",)),
                "results": results,
            }
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
