"""Issue ledger — systematic, autonomous-handled work items.

This is deliberately *separate* from ``.decisions.json``: decisions record the
*why* (architectural choices), issues record the *what* (concrete, locatable
work to fix or improve). An issue may reference a decision via ``decision_ref``
but the two ledgers never overlap.

The ledger is committed at the repo root as ``.issues.json`` so the autonomous
agent and the human reviewer see the identical work list on every machine.
Writes are atomic (temp file + rename) because three writers touch it: the
collector (seeds new findings), the ``issue`` REPL command (triage/status), and
the autonomous driver (outcome).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
ISSUES_PATH = REPO_ROOT / ".issues.json"

# autonomy_level: 0 = human-only, 1 = auto-safe (tests + security gates),
# 2 = benchmark-required (explicit human promotion). New issues default to 1.
DEFAULT_AUTONOMY_LEVEL = 1
VALID_AUTONOMY_LEVELS = (0, 1, 2)

RESOLVED_STATUSES = ("resolved", "wontfix")

# Lazy import for Kanban sync — only loaded when enabled via env var.
_kanban_bridge = None


def _get_kanban_bridge():
    """Lazy-load kanban_bridge to avoid circular imports and unnecessary deps."""
    global _kanban_bridge
    if _kanban_bridge is None:
        try:
            from . import kanban_bridge  # type: ignore[import-not-found,unused-import]

            _kanban_bridge = kanban_bridge
        except ImportError:
            pass
    return _kanban_bridge


def _should_sync() -> bool:
    """Check if Kanban sync is enabled.

    The master switch is resolved by ``kanban_bridge.sync_enabled()`` (process
    env first, then the repo ``.env``) rather than by reading ``os.environ``
    here: ``.env.example`` documents ``KANBAN_SYNC_ENABLED`` as an ``.env``
    setting, and nothing bridges ``.env`` into ``os.environ``, so an
    os.environ-only gate made the documented switch silently dead.

    Two independent gates must both pass (layered control):

      1. the process-level master switch, resolved by ``sync_enabled()``
         (process env first, then the repo ``.env``); and
      2. the *per-board* opt-in on the Kanban side: the ACTIVE board's working
         copy must carry ``"sync": {"agent1": true}``. While a non-synced board
         is active we enqueue nothing; its queue is held by the Kanban
         processor until a synced board becomes active (fail closed — see
         ``kanban_bridge.active_board_allows_agent1``).

    Best-effort: any failure while resolving the bridge (e.g. the module is
    missing or a broken install) is treated as "sync disabled" rather than
    propagating — issue creation must never fail because of the sync path.
    """
    try:
        bridge = _get_kanban_bridge()
        if bridge is None:
            return False
        # Layer 1: process-level master switch (env / repo .env).
        if not bridge.sync_enabled():
            return False
        # Layer 2: per-board opt-in, read from disk on every call so a board
        # switched or toggled in the Kanban UI takes effect immediately.
        return bool(bridge.active_board_allows_agent1())
    except Exception:
        return False


def _enqueue_sync(
    op: str, source_id: str, payload: dict[str, Any], emit_child_card: bool = False
) -> None:
    """Best-effort Kanban enqueue. Never raises into the caller.

    Syncing to Kanban is a side-effect of the issue ledger; a broken bridge
    or a full disk must not abort the primary operation (issue create/resolve).

    A card-attributable issue op (one whose payload carries ``parent_card_id``
    or a ``kanban:<card_id>`` location) additionally enqueues a child-card
    payload so the work performed on that card is visible on the board as a
    child card under the parent (see :func:`_emit_child_card_for`). The
    ``emit_child_card`` flag is set only for genuine work issues; the bridge
    mirror (Kanban→Agent1) must not emit a child card, or the echo path
    would create a spurious child card on every card-create round trip.
    """
    try:
        if not _should_sync():
            return
        kb = _get_kanban_bridge()
        if kb is not None:
            kb.enqueue({"op": op, "source_id": source_id, "payload": payload})
            if emit_child_card:
                _emit_child_card_for(payload)
    except Exception as exc:  # noqa: BLE001 — deliberately swallow, sync is optional
        import logging

        logging.getLogger(__name__).warning(
            "Kanban sync failed for %s (%s): %s", op, source_id, exc
        )


def _card_id_from_locations(payload: dict[str, Any]) -> str | None:
    """Extract a card id from a payload's ``kanban:<card_id>`` location.

    The bridge records ``kanban:<card_id>`` locations on issues it created from
    a Kanban card, so this is the fallback when ``parent_card_id`` is absent.
    Returns ``None`` when the payload carries no such location.
    """
    for loc in payload.get("locations") or []:
        if isinstance(loc, str) and loc.startswith("kanban:"):
            return loc[len("kanban:"):]
    return None


def _emit_child_card_for(payload: dict[str, Any]) -> None:
    """Best-effort: emit a child-card payload for work on a synchronized card.

    Only work-on-card issues (those attributable to a card via ``parent_card_id``
    or a ``kanban:<card>`` location) get a child-card payload; non-card issues
    are left as-is.  Any failure degrades to a no-op so it can never abort the
    primary issue op that triggered the emission.

    The child-card payload carries the work record (title/text = work summary,
    evidence/approach, plus issue id/status/op/ts metadata) and ``parent_id``
    = the card it belongs to, so the Kanban reader (Component 2) can nest the
    child card under the parent.  The work itself is also captured on the Agent1
    issue ledger (``.issues.json``); the payload just surfaces it on the board.
    """
    card_id = payload.get("parent_card_id") or _card_id_from_locations(payload)
    if not card_id:
        return

    issue_id = payload.get("id")
    issue_status = payload.get("status", "open")
    summary = _work_summary_for_child_card(payload)

    kb = _get_kanban_bridge()
    if kb is None:
        return
    try:
        kb.enqueue(
            {
                "op": "child_card_create",
                "source_id": issue_id or card_id,
                "payload": {
                    "parent_id": card_id,
                    "title": summary,
                    "text": summary,
                    "issue_id": issue_id,
                    "op": payload.get("op", ""),
                    "status": issue_status,
                    "created_at": payload.get("created_at") or _now(),
                    "system": "harnessfix",
                },
            }
        )
    except Exception:  # noqa: BLE001 — child-card emission must never break the op
        import logging

        logging.getLogger(__name__).warning(
            "Child-card emission failed for issue %s (card %s); parent op still synced",
            issue_id,
            card_id,
        )


def _work_summary_for_child_card(payload: dict[str, Any]) -> str:
    """Build the child-card title/text from the work record.

    Title carries the work id + status (the work identity); text carries the
    evidence/approach body so the work is visible on the board alone.
    """
    parts: list[str] = []
    if payload.get("evidence"):
        parts.append(payload["evidence"])
    if payload.get("suggested_approach"):
        parts.append(f"Approach:\n{payload['suggested_approach']}")
    body = "\n\n---\n".join(parts)
    title = f"[work] {payload.get('id')} ({payload.get('status', 'open')})"
    return title + (f"\n\n---\n{body}" if body else "")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(category: str, locations: list[str]) -> str:
    """Stable id from (category, locations) so the collector is idempotent."""
    loc = "+".join(sorted(locations))
    cat = re.sub(r"[^a-z0-9]+", "-", category.lower()).strip("-")
    digest = re.sub(r"[^A-Za-z0-9]", "_", loc)[:80]
    return f"iss-{cat}-{digest}"


def load_issues(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or ISSUES_PATH
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return []
    return data if isinstance(data, list) else []


def save_issues(issues: list[dict[str, Any]], path: Path | None = None) -> None:
    """Atomic write: temp file + os.replace so a crash can't corrupt the ledger."""
    p = path or ISSUES_PATH
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(issues, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(p)


def find_by_id(issues: list[dict[str, Any]], issue_id: str) -> dict[str, Any] | None:
    for it in issues:
        if it.get("id") == issue_id:
            return it
    return None


def make_issue(
    category: str,
    title: str,
    locations: list[str],
    *,
    severity: str = "low",
    evidence: str = "",
    suggested_approach: str = "",
    decision_ref: str | None = None,
    autonomy_level: int = DEFAULT_AUTONOMY_LEVEL,
    parent_card_id: str | None = None,
    emit_child_card: bool = False,
) -> dict[str, Any]:
    issue = {
        "id": _slug(category, locations),
        "title": title,
        "category": category,
        "severity": severity,
        "locations": list(locations),
        "status": "open",
        "autonomy_level": autonomy_level,
        "evidence": evidence,
        "suggested_approach": suggested_approach,
        "decision_ref": decision_ref,
        "repair_ref": None,
        "created_at": _now(),
        "resolved_at": None,
    }
    if parent_card_id:
        issue["parent_card_id"] = parent_card_id

    # Enqueue creation to Kanban if sync is enabled (best-effort side-effect)
    payload = {k: v for k, v in issue.items() if k != "repair_ref"}
    _enqueue_sync(
        "issue_create", issue["id"], payload, emit_child_card=emit_child_card
    )

    return issue


def upsert(issues: list[dict[str, Any]], issue: dict[str, Any]) -> bool:
    """Add ``issue`` only if its id is absent.

    Never overwrites a human-set ``status``/``autonomy_level`` — idempotent for
    repeated collector runs. Returns True if a new issue was added.
    """
    if find_by_id(issues, issue["id"]) is not None:
        return False
    issues.append(issue)
    return True


def resolve(
    issues: list[dict[str, Any]],
    issue_id: str,
    disposition: str = "resolved",
    note: str = "",
    emit_child_card: bool = False,
) -> bool:
    it = find_by_id(issues, issue_id)
    if it is None:
        return False
    prior_status = it.get("status")
    # Only emit a work card on a genuine transition into a resolved state.
    should_emit = bool(
        emit_child_card
        and disposition in RESOLVED_STATUSES
        and prior_status not in RESOLVED_STATUSES
    )
    it["status"] = disposition
    it["resolved_at"] = _now() if disposition in RESOLVED_STATUSES else None
    if note:
        it["evidence"] = f"{it.get('evidence', '')}\n[note] {note}".strip()

    # Enqueue resolution to Kanban if sync is enabled (best-effort side-effect)
    payload = dict(it)  # copy of the resolved issue
    payload["disposition"] = disposition
    payload["op"] = "issue_resolve"
    _enqueue_sync(
        "issue_resolve", issue_id, payload, emit_child_card=should_emit
    )

    return True


def promote(
    issues: list[dict[str, Any]],
    issue_id: str,
    level: int,
) -> tuple[bool, str]:
    if level not in VALID_AUTONOMY_LEVELS:
        return (
            False,
            f"invalid autonomy_level {level!r}; expected {VALID_AUTONOMY_LEVELS}",
        )
    it = find_by_id(issues, issue_id)
    if it is None:
        return False, f"no such issue {issue_id!r}"
    it["autonomy_level"] = level

    # Enqueue the new level to Kanban if sync is enabled (best-effort
    # side-effect).  Without this the card keeps showing the level it had when
    # it was created, so a promoted issue looks un-promoted on the board.
    _enqueue_sync("issue_update", issue_id, dict(it))

    return True, f"promoted {issue_id} to autonomy_level={level}"


def open_issues(
    issues: list[dict[str, Any]],
    *,
    max_level: int | None = None,
) -> list[dict[str, Any]]:
    """Open issues eligible for the autonomous driver.

    ``max_level`` caps by autonomy_level (the driver's ``AGENT_AUTONOMY_LEVEL``).
    Deferred/wontfix are excluded; resolved are excluded.
    """
    out = []
    for it in issues:
        if it.get("status") in ("resolved", "wontfix", "deferred"):
            continue
        lvl = int(it.get("autonomy_level", DEFAULT_AUTONOMY_LEVEL))
        if max_level is not None and lvl > max_level:
            continue
        out.append(it)
    return out
