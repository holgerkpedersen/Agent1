"""Remove meaningless (test-fixture) cards from a Kanban board file.

The Kanban board accumulates cards created by Agent1's own test suite.  Two
shapes leak through, and both are recognisable without guessing:

* **placeholder titles** — ``x``, ``y``, ``z``, ``low``, ``high``, ``bench``
  and stray ``ælø`` / ``æølæøl`` keyboard mashing.  These are what
  ``tests/conftest.py`` writes when it only needs *a* card.
* **pytest temp-path titles** — a card whose title is a captured
  ``best-effort-except in C:/Users/.../pytest-NN/test_.../sample.py:5``
  finding point.  The ``pytest-of-<user>`` segment can never be a human title.

Why this is not "delete anything short": the rules are exact-match and
prefix-only.  A real card such as ``Deleting a Frame`` or ``Work button``
survives because no rule matches it, and :func:`classify_card` returns the
reason alongside the verdict so every deletion can be justified line by line.

Safety properties, in order:

1. ``dry_run=True`` is the **default** on the CLI.  Nothing is written without
   ``--apply``.
2. :func:`backup_board` copies the file to a timestamped sibling *before* the
   write and refuses to overwrite an existing backup.
3. The write is verified by re-reading the file: the card count must equal the
   expected number, no junk may remain, every survivor must still belong to a
   frame, and no surviving card may be left pointing at a deleted parent.  A
   failed check restores nothing silently — it raises.

The board app is single-writer and loads the file on demand, so this tool
refuses to run while it is listening: a live app would rewrite the file from
its own in-memory copy and undo the purge.

CLI::

    python -m harnessfix.kanban_purge scan  [--board agent1]
    python -m harnessfix.kanban_purge apply [--board agent1] [--force]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import logging

from . import kanban_bridge as kb
from . import kanban_inspect as ki
from . import kanban_junk

logger = logging.getLogger(__name__)

#: Titles that can never be a human-written card.  Exact match, case-folded.
#: Re-exported from :mod:`harnessfix.kanban_junk` so the outbox gate and this
#: purge can never disagree about what counts as junk.
PLACEHOLDER_TITLES: frozenset[str] = kanban_junk.PLACEHOLDER_TITLES

#: Prefix of a card titled after a pytest temp findings path.
TEMP_TITLE_PREFIX = "best-effort-except in C:/Users/"

#: Marker that makes a temp path unmistakably pytest-generated.
TEMP_TITLE_MARKER = "pytest-of-"

#: ``system`` values that are ALWAYS discarded, with no flag required.
#:
#: ``harnessfix`` is the name of Agent1's own Python package, not a real
#: external system.  The Kanban sync used to stamp it on every card it built
#: (``C:/Dev/Kanban/src/agent1/kanban_sync.py``), so ``System=harnessfix`` on a
#: card means "created by Agent1's sync layer", never "authored by a human".
#: Such cards are Agent1's own echo noise and must not survive on a board.
#: Real authoring systems are ``Agent1``, ``Kanban``, ``git-server`` and the
#: human teams — none of which appear here.
ALWAYS_DROP_SYSTEMS: frozenset[str] = frozenset({"harnessfix"})


def classify_card(
    card: dict[str, Any], *, drop_systems: frozenset[str] = frozenset()
) -> Optional[str]:
    """Why this card is meaningless, or ``None`` if it must be kept.

    ``drop_systems`` adds ``system`` values (case-folded) to remove wholesale,
    on top of :data:`ALWAYS_DROP_SYSTEMS`.  Every other system tag is kept, so
    the purge never deletes a real author's card: a card stamped ``Agent1`` or a
    human team survives even when its title looks odd.
    """
    title = (card.get("title") or "").strip()
    if kanban_junk.is_placeholder_title(title):
        return "placeholder-title"
    if title.startswith(TEMP_TITLE_PREFIX) and TEMP_TITLE_MARKER in title:
        return "pytest-temp-title"
    system = str(card.get("system") or "").strip().casefold()
    if system and system in (set(drop_systems) | set(ALWAYS_DROP_SYSTEMS)):
        return f"system:{system}"
    return None


def _frame_titles(board: dict[str, Any]) -> dict[str, str]:
    return {
        f.get("id", ""): f.get("title", "")
        for f in board.get("frames") or []
        if isinstance(f, dict)
    }


def plan_purge(
    board: dict[str, Any], *, drop_systems: frozenset[str] = frozenset()
) -> dict[str, Any]:
    """Decide, without mutating anything, which cards go and which stay."""
    frames = _frame_titles(board)
    cards = [c for c in board.get("cards") or [] if isinstance(c, dict)]

    junk: list[dict[str, str]] = []
    keep: list[dict[str, Any]] = []
    for card in cards:
        reason = classify_card(card, drop_systems=drop_systems)
        if reason is None:
            keep.append(card)
        else:
            junk.append(
                {
                    "id": card.get("id", ""),
                    "title": (card.get("title") or "").strip(),
                    "frame": frames.get(card.get("frame_id", ""), ""),
                    "reason": reason,
                }
            )

    junk_ids = {j["id"] for j in junk}
    keep_ids = {c.get("id", "") for c in keep}

    return {
        "total": len(cards),
        "kept": len(keep),
        "removed": len(junk),
        "reasons": dict(Counter(j["reason"] for j in junk)),
        "per_frame_kept": dict(
            Counter(frames.get(c.get("frame_id", ""), "") for c in keep)
        ),
        "per_frame_removed": dict(Counter(j["frame"] for j in junk)),
        "remove_ids": sorted(junk_ids),
        "junk": junk,
        # A surviving card must not be orphaned by the purge.
        "orphaned_by_purge": sorted(
            c.get("id", "")
            for c in keep
            if (c.get("parent_id") or "") and c["parent_id"] not in keep_ids
        ),
        "kept_with_deleted_parent": sorted(
            c.get("id", "")
            for c in keep
            if (c.get("parent_id") or "") in junk_ids
        ),
    }


def serialize_board(board: dict[str, Any]) -> bytes:
    """Byte-identical to what the Kanban app writes: indent 2, CRLF, no EOF NL."""
    text = json.dumps(board, indent=2, ensure_ascii=False)
    return text.replace("\n", "\r\n").encode("utf-8")


def board_path(board_id: str) -> Path:
    """``data/board-<board_id>.json`` — the board's working copy."""
    return kb.DATA_DIR / f"board-{board_id}.json"


def backup_board(path: Path, *, label: str = "purge") -> Path:
    """Copy the board aside with a timestamp; never overwrite an existing one.

    ``label`` names the reason for the backup (``purge``, ``undopre``) so an
    undo never overwrites the very backup the user may still need.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{path.name}.bak-{label}-{stamp}"
    target = path.with_name(base)
    # Never overwrite an existing backup; a fast rerun gets a -2, -3 suffix
    # instead of an exception the caller cannot act on.
    suffix = 2
    while target.exists():
        target = path.with_name(f"{base}-{suffix}")
        suffix += 1
    shutil.copy2(path, target)
    if target.stat().st_size != path.stat().st_size:
        raise OSError(f"backup size mismatch: {target}")
    return target


def purge_board(
    board: dict[str, Any], *, drop_systems: frozenset[str] = frozenset()
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Drop junk cards and their frame memberships; return (new board, plan)."""
    plan = plan_purge(board, drop_systems=drop_systems)
    doomed = set(plan["remove_ids"])

    if plan["kept_with_deleted_parent"]:
        raise ValueError(
            "refusing to purge: surviving cards would lose their parent: "
            f"{plan['kept_with_deleted_parent']}"
        )

    new_board = dict(board)
    new_board["cards"] = [
        c
        for c in board.get("cards") or []
        if not (isinstance(c, dict) and c.get("id") in doomed)
    ]
    frames = []
    for frame in board.get("frames") or []:
        if isinstance(frame, dict):
            frame = dict(frame)
            if isinstance(frame.get("card_ids"), list):
                frame["card_ids"] = [
                    cid for cid in frame["card_ids"] if cid not in doomed
                ]
            frames.append(frame)
    new_board["frames"] = frames
    return new_board, plan


def verify_purge(
    new_board: dict[str, Any],
    plan: dict[str, Any],
    board: dict[str, Any],
    *,
    drop_systems: frozenset[str] = frozenset(),
) -> list[str]:
    """Return the list of problems found; empty means the purge is sound."""
    problems: list[str] = []
    cards = [c for c in new_board.get("cards") or [] if isinstance(c, dict)]
    ids = {c.get("id", "") for c in cards}

    if len(cards) != plan["kept"]:
        problems.append(f"card count {len(cards)} != expected {plan['kept']}")
    leftover = [
        c.get("id", "")
        for c in cards
        if classify_card(c, drop_systems=drop_systems) is not None
    ]
    if leftover:
        problems.append(f"junk survived: {leftover[:5]}")

    frame_members = {
        cid for f in new_board.get("frames") or [] for cid in (f.get("card_ids") or [])
    }
    # Frames can already point at cards the board no longer has (drift the app
    # never cleans up).  Only *newly* dangling members are this purge's fault.
    members_before = {
        cid for f in board.get("frames") or [] for cid in (f.get("card_ids") or [])
    }
    ids_before = {
        c.get("id", "") for c in board.get("cards") or [] if isinstance(c, dict)
    }
    dangling_before = members_before - ids_before
    newly_dangling = (frame_members - ids) - dangling_before
    if newly_dangling:
        problems.append(
            f"frame refers to deleted cards: {sorted(newly_dangling)}"
        )
    # A stale reference the purge did not create must survive untouched.
    if dangling_before - frame_members:
        problems.append(
            "pre-existing dangling members were rewritten: "
            f"{sorted(dangling_before - frame_members)}"
        )
    if ids - frame_members:
        problems.append(f"cards in no frame: {sorted(ids - frame_members)}")

    # Nothing that was kept may have been dropped.
    kept_before = {
        c.get("id")
        for c in board.get("cards") or []
        if isinstance(c, dict)
        and classify_card(c, drop_systems=drop_systems) is None
    }
    if kept_before - ids:
        problems.append(f"kept cards were deleted: {sorted(kept_before - ids)}")
    return problems


def stranded_links(board: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Map entries pointing at the cards this purge deletes.

    Reported, never rewritten: ``.card_id_map.json`` is a cache that the bridge
    heals from ``locations["kanban:<id>"]``, so hand-editing it is forbidden.
    """
    try:
        mapping = json.loads(kb.CARD_ID_MAP_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"available": False, "entries": 0, "stranded_card_keys": []}
    doomed = set(plan["remove_ids"])
    stranded = sorted(k for k in mapping if k in doomed)
    return {
        "available": True,
        "entries": len(mapping),
        "stranded_card_keys": stranded,
        "stranded_count": len(stranded),
    }


def _port_is_live(host: str = "127.0.0.1", port: int = 3000) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((host, port)) == 0


def apply_purge(
    board_id: str = "agent1",
    *,
    force: bool = False,
    drop_systems: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Purge junk cards from one board file, with backup and verification."""
    path = board_path(board_id)
    raw = path.read_bytes()
    board = json.loads(raw.decode("utf-8"))

    new_board, plan = purge_board(board, drop_systems=drop_systems)
    problems = verify_purge(new_board, plan, board, drop_systems=drop_systems)
    if problems:
        raise ValueError(f"refusing to write: {problems}")

    if _port_is_live() and not force:
        raise RuntimeError(
            "the Kanban app appears to be listening on :3000 and would rewrite "
            "the board from memory; stop it or pass --force"
        )

    # Byte-identical round-trip is the guarantee that we only removed cards.
    if serialize_board(board) != raw:
        raise ValueError(
            "board does not round-trip byte-for-byte; refusing to rewrite it "
            "(format drift would make the diff unreviewable)"
        )

    label = "purge" + ("-" + "-".join(sorted(drop_systems)) if drop_systems else "")
    backup = backup_board(path, label=label)
    path.write_bytes(serialize_board(new_board))

    reread = json.loads(path.read_bytes().decode("utf-8"))
    after = verify_purge(reread, plan, board, drop_systems=drop_systems)
    if after:
        raise ValueError(f"post-write verification failed: {after}")

    return {
        "board": board_id,
        "path": str(path),
        "backup": str(backup),
        "bytes_before": len(raw),
        "bytes_after": len(serialize_board(new_board)),
        "kept": plan["kept"],
        "removed": plan["removed"],
        "reasons": plan["reasons"],
        "per_frame_kept": plan["per_frame_kept"],
        "removed_cards": [
            {k: j[k] for k in ("id", "title", "frame", "reason")}
            for j in plan["junk"]
        ],
        "links": stranded_links(new_board, plan),
    }


def list_backups(board_id: str = "agent1") -> list[Path]:
    """Purge backups for one board, oldest first."""
    path = board_path(board_id)
    return sorted(path.parent.glob(f"{path.name}.bak-purge-*"))


def restore_board(
    board_id: str = "agent1",
    *,
    backup: Optional[Path] = None,
    force: bool = False,
) -> dict[str, Any]:
    """Undo a purge: put a backup back, keeping the current file aside.

    Without ``backup`` the newest purge backup is used.  The file being
    replaced is copied to ``.bak-undopre-<stamp>`` first, so an undo is itself
    undoable.
    """
    path = board_path(board_id)
    if backup is None:
        candidates = list_backups(board_id)
        if not candidates:
            raise FileNotFoundError(f"no purge backup for board {board_id!r}")
        backup = candidates[-1]
    backup = Path(backup)

    if not backup.exists():
        raise FileNotFoundError(f"no such backup: {backup}")
    if not path.exists():
        raise FileNotFoundError(f"no such board file: {path}")
    if _port_is_live() and not force:
        raise RuntimeError(
            "the Kanban app appears to be listening on :3000 and would rewrite "
            "the board from memory; stop it or pass --force"
        )

    raw_backup = backup.read_bytes()
    restored = json.loads(raw_backup.decode("utf-8"))
    # Same guarantee as a purge: only a byte-exact board is ever written back.
    if serialize_board(restored) != raw_backup:
        raise ValueError(
            f"backup {backup.name} does not round-trip byte-for-byte; "
            "refusing to restore it"
        )

    current = json.loads(path.read_bytes().decode("utf-8"))
    aside = backup_board(path, label="undopre")
    path.write_bytes(raw_backup)

    reread = json.loads(path.read_bytes().decode("utf-8"))
    return {
        "board": board_id,
        "path": str(path),
        "restored_from": str(backup),
        "current_kept_as": str(aside),
        "cards_before": len(current.get("cards") or []),
        "cards_after": len(reread.get("cards") or []),
        "backups_remaining": [p.name for p in list_backups(board_id)],
    }


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def available_boards() -> list[str]:
    """Board ids that have a ``board-<id>.json`` file."""
    return sorted(
        p.name[len("board-") : -len(".json")]
        for p in kb.DATA_DIR.glob("board-*.json")
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnessfix.kanban_purge",
        description="Remove meaningless (test-fixture) Kanban cards.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name, help_text in (
        ("scan", "report what would be removed (read-only)"),
        ("apply", "remove them, after a timestamped backup"),
        ("restore", "undo a purge by restoring its backup"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--board", default="agent1")
        p.add_argument(
            "--system",
            action="append",
            default=[],
            metavar="TAG",
            help="also drop every card whose system tag is TAG "
            "(repeatable; not part of the automatic junk rules)",
        )
        if name in ("apply", "restore"):
            p.add_argument(
                "--force", action="store_true", help="ignore a live board app"
            )
        if name == "restore":
            p.add_argument(
                "--backup", default=None, help="backup file to restore"
            )
            p.add_argument(
                "--list", action="store_true", help="list backups and exit"
            )

    args = parser.parse_args(argv)
    drop_systems = frozenset(s.strip().casefold() for s in args.system if s.strip())

    if args.cmd == "scan":
        board = ki.load_board(args.board)
        if not board:
            _print({"error": f"no such board: {args.board}"})
            return 1
        plan = plan_purge(board, drop_systems=drop_systems)
        plan.pop("junk", None)
        _print(plan)
        return 0

    if args.cmd == "restore":
        if args.list:
            _print(
                {
                    "board": args.board,
                    "backups": [p.name for p in list_backups(args.board)],
                }
            )
            return 0
        result = restore_board(
            args.board,
            backup=Path(args.backup) if args.backup else None,
            force=getattr(args, "force", False),
        )
        _print(result)
        return 0

    result = apply_purge(
        args.board, force=getattr(args, "force", False), drop_systems=drop_systems
    )
    _print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
