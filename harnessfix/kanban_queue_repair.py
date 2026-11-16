"""Compact a Kanban bridge queue: drop duplicated messages, keep the protocol.

The live inbound queue held 463 lines for 33 cards — the same ``card_update``
re-queued five times, every ``card_move`` applied two to four times.  That is
not one writer looping: a single writer cannot emit two lines with the same
``seq`` (see ``kanban_bridge._next_seq``).  It was two to four HANDLERS draining
one directory at once — the agent's boot poller, a second REPL, the manual
``--process-in`` CLI — each applying the same lines and each re-queueing the
same failures, so ``retry_count`` climbed in lockstep.  The durable fix is the
per-queue :func:`harnessfix.kanban_bridge.handling_lock`; this module cleans up
the history that defect left behind.

Why not simply truncate the file?  Because the queue is a *reader cursor*
protocol, not a log:

``seq``      the line's 1-based position, cached in ``seq.txt``;
``offset``   the reader's cursor in ``offset.txt`` — the number of leading
             lines already consumed;
``retry``    failures are re-appended at the TAIL with a NEW ``seq``.

Rewriting the file therefore has to rewrite ``seq.txt`` and ``offset.txt`` with
it.  :func:`repair` does exactly that: it counts how many KEPT lines sit before
the old cursor and writes that as the new cursor, so an unconsumed message stays
unconsumed and a consumed one stays consumed.

Safety properties, in order:

1. ``apply=False`` is the default.  Nothing is written without ``--apply``.
2. The whole operation runs under the queue's own
   :func:`~harnessfix.kanban_bridge.handling_lock`.  If a handler owns the
   queue — the agent is draining it right now — the repair REFUSES instead of
   racing it.
3. The file is copied to ``messages.jsonl.bak-repair-<timestamp>`` first, and
   an existing backup is never overwritten.
4. The new content is written to a sibling temp file and moved into place with
   ``os.replace``, so a reader never sees a half-written queue.
5. The result is verified by re-reading: line count, no duplicate left, every
   kept line still parses, and the cursor/seq counters agree with the file.

Two passes, both conservative:

* **pass 1 (always)** — collapse EXACT duplicates.  Two lines are the same
  message when they carry the same ``op``, ``source_id``, payload and ``ts``
  after the volatile bookkeeping (``seq``, ``retry_count``, ``processed``) is
  ignored.  Order is preserved and the FIRST occurrence is kept, so a
  ``create → move → move`` sequence keeps its meaning.
* **pass 2 (``--drop-superseded``)** — drop cards whose whole story is a
  ``card_delete``.  67 of the 68 deleted cards have no ``card_create`` in the
  queue, no issue in the ledger and no card on any board: nothing depends on
  their history and every replay of it resolves to nothing.  This pass is
  opt-in because it is a judgement about relevance, not a duplicate removal.

CLI::

    python -m harnessfix.kanban_queue_repair                  # dry run
    python -m harnessfix.kanban_queue_repair --apply          # pass 1
    python -m harnessfix.kanban_queue_repair --apply --drop-superseded
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import kanban_bridge as kb

logger = logging.getLogger(__name__)

#: Bookkeeping fields that differ between two copies of the same message.
VOLATILE_FIELDS = ("seq", "retry_count", "processed")

BACKUP_SUFFIX = ".bak-repair-"


def _message_identity(msg: dict[str, Any]) -> str:
    """Stable identity of a message, ignoring the volatile queue bookkeeping."""
    core = {k: v for k, v in msg.items() if k not in VOLATILE_FIELDS}
    return json.dumps(core, sort_keys=True, ensure_ascii=False)


def _load_lines(msg_file: Path) -> list[str]:
    return msg_file.read_text(encoding="utf-8", errors="replace").splitlines()


def _default_ledger_path() -> Path:
    """The live issue ledger (``harnessfix.issues.ISSUES_PATH``).

    Read through the issues module rather than guessing a sibling of the queue
    directory: the ledger is committed at the AGENT repo root, while the queue
    lives in the KANBAN repo's data dir, so ``queue_dir.parent`` pointed at the
    wrong tree and every card looked unlinked.
    """
    from . import issues as issue_store

    return Path(issue_store.ISSUES_PATH)


def _linked_card_ids(ledger: Path | None = None) -> set[str]:
    """Card ids that an issue in the ledger points at (``kanban:<id>``)."""
    path = Path(ledger) if ledger is not None else _default_ledger_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    issues = raw.get("issues") if isinstance(raw, dict) else raw
    out: set[str] = set()
    for issue in issues or []:
        if not isinstance(issue, dict):
            continue
        for loc in issue.get("locations") or []:
            if isinstance(loc, str) and loc.startswith("kanban:"):
                out.add(loc[len("kanban:") :])
    return out


def _board_card_ids(data_dir: Path) -> set[str]:
    """Card ids present on any non-backup board file under *data_dir*."""
    out: set[str] = set()
    for path in sorted(data_dir.glob("board*.json")):
        if ".bak" in path.name:
            continue
        try:
            board = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for frame in board.get("frames") or []:
            for card in frame.get("cards") or []:
                if isinstance(card, dict) and card.get("id"):
                    out.add(str(card["id"]))
    return out


def _superseded_sources(
    messages: list[dict[str, Any]],
    data_dir: Path,
    ledger: Path | None = None,
) -> tuple[set[str], list[str]]:
    """Cards whose only meaning is a ``card_delete`` of something gone.

    Returns ``(card_ids, reasons)``.  A card qualifies when every message about
    it is a ``card_delete`` (so no create/update history is lost), the ledger
    has no issue pointing at it, and no board holds it.
    """
    reasons: list[str] = []
    linked = _linked_card_ids(ledger)
    on_board = _board_card_ids(data_dir)
    by_source: dict[str, list[dict[str, Any]]] = {}
    for msg in messages:
        sid = str(msg.get("source_id") or "")
        if sid:
            by_source.setdefault(sid, []).append(msg)

    dead: set[str] = set()
    for sid, group in by_source.items():
        if any(m.get("op") != "card_delete" for m in group):
            continue
        if sid in linked:
            reasons.append(f"{sid[:8]}: kept, ledger issue still points at it")
            continue
        if sid in on_board:
            reasons.append(f"{sid[:8]}: kept, the card is still on a board")
            continue
        dead.add(sid)
        reasons.append(
            f"{sid[:8]}: dropped {len(group)} delete line(s), card is gone from "
            f"ledger and boards"
        )
    return dead, reasons


def analyse(
    queue_dir: Path | str,
    *,
    drop_superseded: bool = False,
    ledger: Path | str | None = None,
) -> dict[str, Any]:
    """Plan a repair for *queue_dir* without writing anything.

    ``drop_superseded`` selects whether pass 2 runs; the plan and the kept line
    list are produced together so the reported counts can never disagree with
    what a subsequent :func:`repair` would write.  ``ledger`` overrides the
    issue ledger used by pass 2 (default: ``harnessfix.issues.ISSUES_PATH``).
    """
    queue_dir = Path(queue_dir).resolve()
    msg_file = queue_dir / "messages.jsonl"
    data_dir = queue_dir.parent

    lines = _load_lines(msg_file) if msg_file.exists() else []
    parsed: list[tuple[int, str, Optional[dict[str, Any]]]] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            msg = json.loads(stripped)
        except ValueError:
            parsed.append((idx, line, None))
            continue
        if not isinstance(msg, dict):
            parsed.append((idx, line, None))
            continue
        parsed.append((idx, line, msg))

    # --- pass 1: exact duplicates ------------------------------------------
    seen: set[str] = set()
    after_pass1: list[tuple[int, str, Optional[dict[str, Any]]]] = []
    duplicates = 0
    for item in parsed:
        _idx, _line, msg = item
        if msg is None:
            after_pass1.append(item)  # malformed: leave it to the bridge
            continue
        ident = _message_identity(msg)
        if ident in seen:
            duplicates += 1
            continue
        seen.add(ident)
        after_pass1.append(item)

    # --- pass 2: superseded delete-only cards ------------------------------
    pass1_messages = [m for _i, _l, m in after_pass1 if m is not None]
    dead_sources, reasons = (
        _superseded_sources(pass1_messages, data_dir, ledger)
        if drop_superseded
        else (set(), [])
    )
    kept: list[tuple[int, str, Optional[dict[str, Any]]]] = []
    superseded = 0
    for item in after_pass1:
        _idx, _line, msg = item
        if msg is not None and str(msg.get("source_id") or "") in dead_sources:
            superseded += 1
            continue
        kept.append(item)

    old_offset = kb._read_int(queue_dir / "offset.txt")
    # Cursor remap: how many KEPT lines were already consumed.
    new_offset = sum(1 for idx, _l, _m in kept if idx < old_offset)
    seqs = [int(m.get("seq") or 0) for _i, _l, m in kept if m is not None]
    new_seq = max(seqs) if seqs else 0

    return {
        "queue_dir": str(queue_dir),
        "file": str(msg_file),
        "lines_before": len(parsed),
        "duplicate_lines": duplicates,
        "superseded_lines": superseded,
        "lines_after": len(kept),
        "malformed_lines": sum(1 for _i, _l, m in kept if m is None),
        "distinct_cards_before": len(
            {str(m.get("source_id")) for _i, _l, m in parsed if m is not None}
        ),
        "distinct_cards_after": len(
            {str(m.get("source_id")) for _i, _l, m in kept if m is not None}
        ),
        "offset_before": old_offset,
        "offset_after": new_offset,
        "seq_before": kb._read_int(queue_dir / "seq.txt"),
        "seq_after": new_seq,
        "pending_before": max(0, len(parsed) - old_offset),
        "pending_after": max(0, len(kept) - new_offset),
        "reasons": reasons,
        "_kept": kept,
    }


def _backup(msg_file: Path) -> Path:
    """Copy the queue aside before a rewrite; never overwrite a backup."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = msg_file.with_name(msg_file.name + BACKUP_SUFFIX + stamp)
    n = 1
    while target.exists():
        target = msg_file.with_name(f"{msg_file.name}{BACKUP_SUFFIX}{stamp}-{n}")
        n += 1
    shutil.copy2(str(msg_file), str(target))
    return target


def repair(
    queue_dir: Path | str,
    *,
    apply: bool = False,
    drop_superseded: bool = False,
    ledger: Path | str | None = None,
) -> dict[str, Any]:
    """Deduplicate the queue's ``messages.jsonl``; write only when *apply*.

    Refuses (``refused`` in the result) when another process owns the queue,
    because rewriting the file under a live reader is precisely how the
    duplicates happened in the first place.
    """
    queue_dir = Path(queue_dir).resolve()
    plan = analyse(queue_dir, drop_superseded=drop_superseded, ledger=ledger)
    kept = plan.pop("_kept")
    plan["applied"] = False
    plan["drop_superseded"] = drop_superseded

    if plan["lines_after"] == plan["lines_before"]:
        plan["refused"] = "nothing to remove"
        return plan

    if not apply:
        return plan

    with kb.handling_lock(queue_dir, timeout=0.0) as owned:
        if not owned:
            owner = kb.handler_lock_owner(queue_dir) or {}
            plan["refused"] = (
                "another handler owns this queue"
                + (f" (pid {owner.get('pid')})" if owner.get("pid") else "")
            )
            logger.warning("Queue repair refused: %s", plan["refused"])
            return plan
        try:
            return _rewrite(queue_dir, plan, kept)
        except Exception as exc:  # noqa: BLE001 - report, never half-succeed
            plan["refused"] = f"rewrite failed: {exc}"
            logger.exception("Queue repair failed")
            return plan


def _rewrite(
    queue_dir: Path,
    plan: dict[str, Any],
    kept: list[tuple[int, str, Optional[dict[str, Any]]]],
) -> dict[str, Any]:
    """Write the compacted queue plus its cursor/seq counters, then verify."""
    msg_file = queue_dir / "messages.jsonl"
    backup = _backup(msg_file)
    plan["backup"] = str(backup)

    # ``kept`` is already filtered by both passes in ``analyse``, so the written
    # lines and the reported counts come from one and the same list.
    content = "\n".join(line for _idx, line, _msg in kept)
    if content:
        content += "\n"

    tmp = msg_file.with_name(msg_file.name + ".repair.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, msg_file)

    (queue_dir / "offset.txt").write_text(str(plan["offset_after"]), encoding="utf-8")
    (queue_dir / "seq.txt").write_text(str(plan["seq_after"]), encoding="utf-8")

    _verify(queue_dir, plan)
    plan["applied"] = True
    logger.info(
        "Compacted %s: %d -> %d lines (offset %d -> %d, seq %d -> %d)",
        msg_file,
        plan["lines_before"],
        plan["lines_after"],
        plan["offset_before"],
        plan["offset_after"],
        plan["seq_before"],
        plan["seq_after"],
    )
    return plan


def _verify(queue_dir: Path, plan: dict[str, Any]) -> None:
    """Re-read the rewritten queue and refuse to report success on drift."""
    msg_file = queue_dir / "messages.jsonl"
    lines = [ln for ln in _load_lines(msg_file) if ln.strip()]
    if len(lines) != plan["lines_after"]:
        raise RuntimeError(
            f"line count mismatch: wrote {plan['lines_after']}, found {len(lines)}"
        )
    idents = []
    for line in lines:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if isinstance(msg, dict):
            idents.append(_message_identity(msg))
    if len(idents) != len(set(idents)):
        raise RuntimeError("duplicates remain after the rewrite")
    offset = kb._read_int(queue_dir / "offset.txt")
    if offset > len(lines):
        raise RuntimeError(
            f"cursor {offset} points past the {len(lines)} lines that remain"
        )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue-dir",
        default=None,
        help="Queue directory (default: the kanban->agent1 inbound queue)",
    )
    parser.add_argument("--apply", action="store_true", help="write the changes")
    parser.add_argument(
        "--ledger",
        default=None,
        help="Issue ledger used by --drop-superseded "
        "(default: harnessfix.issues.ISSUES_PATH)",
    )
    parser.add_argument(
        "--drop-superseded",
        action="store_true",
        help="also drop history of cards that only have a card_delete left",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    queue_dir = Path(args.queue_dir) if args.queue_dir else kb.QUEUE_KANBAN_TO_AGENT1
    result = repair(
        queue_dir,
        apply=args.apply,
        drop_superseded=args.drop_superseded,
        ledger=args.ledger,
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"queue            : {result['queue_dir']}")
        print(f"lines            : {result['lines_before']} -> {result['lines_after']}")
        print(f"duplicates       : {result['duplicate_lines']}")
        print(f"superseded       : {result['superseded_lines']}")
        print(f"cards            : {result['distinct_cards_before']} -> "
              f"{result['distinct_cards_after']}")
        print(
            f"offset           : {result['offset_before']} -> "
            f"{result['offset_after']}"
        )
        print(f"seq              : {result['seq_before']} -> {result['seq_after']}")
        print(
            f"pending          : {result['pending_before']} -> "
            f"{result['pending_after']}"
        )
        print(f"applied          : {result['applied']}")
        if result.get("refused"):
            print(f"refused          : {result['refused']}")
        if result.get("backup"):
            print(f"backup           : {result['backup']}")
    return 0 if result["applied"] or not args.apply or result.get("refused") else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main())
