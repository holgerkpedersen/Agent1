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


def _agent_identity():
    """Lazy-load the session identity module (stdlib-only; no import cycle).

    Returns ``None`` when the module is unavailable — identity stamping is a
    best-effort side-effect and must never break an issue op.
    """
    try:
        from agent_core.session import identity  # type: ignore[import-not-found,unused-import]

        return identity
    except Exception:  # noqa: BLE001 — sync-side metadata must not break the ledger
        return None


def _stamp_identity(payload: dict[str, Any]) -> dict[str, Any]:
    """Copy *payload* with the acting session's identity fields added.

    Field names are the cross-repo contract (``agent1_session``,
    ``agent1_label``, ``claimed_at``) — see
    ``agent_core/session/identity.py``.  The stamp is applied to every
    outbound Kanban payload so the board shows *which* Agent1 session did the
    work; failures degrade to the un-stamped payload.
    """
    ident = _agent_identity()
    if ident is None:
        return dict(payload)
    try:
        stamped = dict(payload)
        stamped.update(ident.identity_payload())
        return stamped
    except Exception:  # noqa: BLE001 — identity metadata must not break the op
        return dict(payload)


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
            kb.enqueue(
                {"op": op, "source_id": source_id, "payload": _stamp_identity(payload)}
            )
            if emit_child_card:
                _emit_child_card_for(payload, op=op)
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


def _emit_child_card_for(payload: dict[str, Any], op: str = "") -> None:
    """Best-effort: emit a child-card payload for work on a synchronized card.

    Only work-on-card issues (those attributable to a card via ``parent_card_id``
    or a ``kanban:<card>`` location) get a child-card payload; non-card issues
    are left as-is.  Any failure degrades to a no-op so it can never abort the
    primary issue op that triggered the emission.

    The child-card payload carries the work record (title/text = work summary,
    evidence/approach, plus issue id/status/op metadata) and ``parent_id``
    = the card it belongs to, so the Kanban reader (Component 2) can nest the
    child card under the parent.  The work itself is also captured on the Agent1
    issue ledger (``.issues.json``); the payload just surfaces it on the board.

    **One child card per work item**: the op is ``child_card_create`` with
    ``source_id`` = the issue id, so one work item maps to exactly one child
    card.  The Kanban reader (Component 2) creates it on first sight and
    updates it in place on every later emission for the same ``source_id``
    (milestones keep the same card).  The payload carries the acting session's
    identity (``agent1_session``/``agent1_label``) so the board shows who did
    the work.

    The op name is part of the **cross-repo contract**: the reader dispatches
    ``child_card_create`` and ignores ops it does not know, so renaming this
    op silently drops every child card.  Change it only together with
    ``src/agent1/kanban_sync.py`` in the Kanban repo.
    """
    card_id = payload.get("parent_card_id") or _card_id_from_locations(payload)
    if not card_id:
        return

    issue_id = payload.get("id")
    issue_status = payload.get("status", "open")
    summary = _work_summary_for_child_card(payload)
    op_val = (op or payload.get("op") or "").strip()

    kb = _get_kanban_bridge()
    if kb is None:
        return
    ident = _agent_identity()
    sess = ""
    label = ""
    if ident is not None:
        try:
            sess, label = ident.session_id(), ident.agent_label()
        except Exception:  # noqa: BLE001 — badge metadata is optional
            pass
    sess_tag = f"[{sess}]" if sess else ""
    try:
        kb.enqueue(
            {
                "op": "child_card_create",
                "source_id": issue_id or card_id,
                "payload": {
                    "parent_id": card_id,
                    "title": f"[Agent1 work]{sess_tag} {issue_id} ({issue_status})",
                    "text": summary,
                    "issue_id": issue_id,
                    "op": op_val,
                    "status": issue_status,
                    "created_at": payload.get("created_at") or _now(),
                    "system": "harnessfix",
                    # Session identity for the board (cross-repo contract).
                    "agent1_session": sess,
                    "agent1_label": label,
                    # Badge tags so the child card is identifiable as a child of
                    # the parent (and by its author) in flat views.
                    "tags": [f"[card:{card_id}]"] + ([sess_tag] if sess_tag else []),
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
    """Build the child-card body from the work record.

    The body leads with a header line naming the work item and its status, so
    the card text is self-describing on the board (a reader sees which issue and
    how far it got without opening the Agent1 ledger).  Evidence and approach
    follow.  Empty fields are skipped — a missing fact stays absent, never a
    placeholder.
    """
    parts: list[str] = []
    header = " ".join(
        str(v) for v in (payload.get("id"), payload.get("status")) if v
    )
    if header:
        parts.append(header)
    if payload.get("evidence"):
        parts.append(payload["evidence"])
    if payload.get("suggested_approach"):
        parts.append(f"Approach:\n{payload['suggested_approach']}")
    body = "\n\n---\n".join(parts)
    return body


def _claim_for_card(card_id: str, payload: dict[str, Any]) -> tuple[bool, dict]:
    """First-writer-wins claim on *card_id* for this session (best-effort).

    Returns ``(acquired, record)``.  When the card is live-claimed by a foreign
    session, returns ``(False, their_record)`` and enqueues a ``card_claimed``
    message naming the FOREIGN owner so the board records who claimed it first
    (the claim protocol: Kanban-side assignment emits the same op).
    """
    ident = _agent_identity()
    if ident is None:
        return True, {}
    try:
        acquired, record = ident.claim_card(card_id)
    except Exception:  # noqa: BLE001 — claim bookkeeping must not break the op
        return True, {}
    if not acquired:
        kb = _get_kanban_bridge()
        if kb is not None:
            try:
                kb.enqueue(
                    {
                        "op": "card_claimed",
                        "source_id": card_id,
                        "payload": {
                            "card_id": card_id,
                            "owner": record.get("owner"),
                            "label": record.get("label"),
                            "claimed_at": record.get("claimed_at"),
                            "attempted_by": ident.session_id(),
                            "issue_id": payload.get("id"),
                            "system": "harnessfix",
                        },
                    }
                )
            except Exception:  # noqa: BLE001 — notification is best-effort
                pass
    return acquired, record


def claim_card_for_work(issue_id: str) -> tuple[bool, str]:
    """Claim-before-work entry point for card-attributed issues (Task 3).

    Call this BEFORE starting work on an issue that came from a Kanban card.
    It resolves the issue's card id, applies the first-writer-wins claim, and
    enqueues a ``card_claimed`` message so the board records the owner.

    Returns ``(acquired, message)``:
      * ``(True,  "claimed card <id> ...")`` — this session owns the card; proceed.
      * ``(True,  "not card-attributed ...")`` — no card link; nothing to claim.
      * ``(False, "card <id> is claimed by ...")`` — a foreign live session
        holds it; the caller MUST NOT start work (hard block).

    Sync-off is a no-op success: with no board there is nothing to claim.
    """
    if not _should_sync():
        return True, "sync disabled — nothing to claim"
    issues = load_issues()
    it = find_by_id(issues, issue_id)
    if it is None:
        return False, f"no issue {issue_id} in the ledger"
    card_id = it.get("parent_card_id") or _card_id_from_locations(it)
    if not card_id:
        return True, "not card-attributed — nothing to claim"
    ident = _agent_identity()
    if ident is None:
        return True, "identity unavailable — claim skipped"
    acquired, record = _claim_for_card(card_id, it)
    if not acquired:
        owner = record.get("owner") or record.get("label") or "unknown"
        return (
            False,
            f"card {card_id} is claimed by session '{owner}' "
            f"(claimed_at {record.get('claimed_at')}) — do not start work",
        )
    kb = _get_kanban_bridge()
    if kb is not None:
        try:
            kb.enqueue(
                {
                    "op": "card_claimed",
                    "source_id": card_id,
                    "payload": {
                        "card_id": card_id,
                        "owner": ident.session_id(),
                        "label": ident.agent_label(),
                        "claimed_at": record.get("claimed_at"),
                        "issue_id": issue_id,
                        "system": "harnessfix",
                    },
                }
            )
        except Exception:  # noqa: BLE001 — claim notification is best-effort
            pass
    return True, f"claimed card {card_id} for session {ident.session_id()}"


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


def _release_claim_for_resolved(issue: dict[str, Any]) -> None:
    """Release the card claim held by this session for *issue*'s card.

    Best-effort and idempotent: a no-op when the issue is not card-attributed,
    when identity is unavailable, or when this session holds no claim.  Only
    the owner's claim is released (``identity.release_card`` checks ownership),
    so resolving an issue can never steal a foreign session's claim.
    """
    card_id = issue.get("parent_card_id") or _card_id_from_locations(issue)
    if not card_id:
        return
    ident = _agent_identity()
    if ident is None:
        return
    try:
        ident.release_card(card_id)
    except Exception:  # noqa: BLE001 — claim bookkeeping must not break the op
        pass


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

    # Release this issue's claim on the card it belongs to.  Without this the
    # claim outlives the work: ``release_card`` has no other automatic caller,
    # so a completed card stays claimed and every later visit to the same card
    # is refused as "claimed by <old session>".  The lease must end with the
    # work; the TTL in ``identity.claim_owner`` stays as the crash safety net.
    if disposition in RESOLVED_STATUSES:
        _release_claim_for_resolved(it)

    # Enqueue resolution to Kanban if sync is enabled (best-effort side-effect)
    payload = dict(it)  # copy of the resolved issue
    payload["disposition"] = disposition
    payload["op"] = "issue_resolve"
    _enqueue_sync(
        "issue_resolve", issue_id, payload, emit_child_card=should_emit
    )

    return True


def update_issue(
    issues: list[dict[str, Any]],
    issue_id: str,
    *,
    evidence: str | None = None,
    suggested_approach: str | None = None,
    status: str | None = None,
    note: str = "",
    emit_child_card: bool = False,
) -> tuple[bool, str]:
    """Mutate an existing issue record (intermediate work milestone).

    Only fields explicitly provided (not ``None``) are touched, so a partial
    update never clobbers a field the caller did not intend to change.  A
    ``note`` is appended to ``evidence`` as a ``[note]`` line (matching
    ``resolve``), so the trail stays chronological.

    When ``emit_child_card`` is set and the issue is card-attributable, a
    ``child_card_create`` is enqueued describing this milestone, so the board
    shows intermediate work, not just intake and resolution.  The parent op
    is ``issue_update`` (the existing Kanban contract).

    Returns ``(ok, message)``.
    """
    it = find_by_id(issues, issue_id)
    if it is None:
        return False, f"no such issue {issue_id!r}"

    if evidence is not None:
        it["evidence"] = evidence
    if suggested_approach is not None:
        it["suggested_approach"] = suggested_approach
    if status is not None:
        it["status"] = status
        if status in RESOLVED_STATUSES:
            it["resolved_at"] = _now()
            _release_claim_for_resolved(it)
        elif it.get("resolved_at") is not None:
            # Re-opening a previously resolved item clears the terminal stamp.
            it["resolved_at"] = None
    if note:
        it["evidence"] = f"{it.get('evidence', '')}\n[note] {note}".strip()
    it["updated_at"] = _now()

    # The child card only makes sense for work-on-card issues; _emit_child_card_for
    # gates on parent_card_id / kanban: location, so pass the flag only when the
    # issue is card-attributable (the parent issue_update op always syncs).
    card_id = it.get("parent_card_id") or _card_id_from_locations(it)
    should_emit = bool(emit_child_card and card_id)

    payload = dict(it)
    payload["op"] = "issue_update"
    _enqueue_sync("issue_update", issue_id, payload, emit_child_card=should_emit)

    return True, f"updated {issue_id}"


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


def work_log_for_card(
    issues: list[dict[str, Any]],
    card_id: str,
) -> list[dict[str, Any]]:
    """Every ledger record whose work belongs to *card_id*, oldest first.

    This is the Agent1-side answer to "what did the session do to this Kanban
    card?".  The issue ledger already IS the work ledger — it carries status,
    evidence, suggested_approach and resolved_at per work item — so no second
    store is needed; what was missing is the card→issues query.

    A record belongs to the card when ``parent_card_id`` matches (set by the
    bridge when a card is linked) or a ``kanban:<card_id>`` location matches
    (the fallback used for cards bridged before ``parent_card_id`` existed).
    The parent issue itself is included first when it is the card's own record,
    followed by any child work items. Ordering is by ``created_at`` then id so
    the trail reads chronologically and stays stable across runs.
    """
    if not card_id:
        return []
    out = [
        it
        for it in issues
        if it.get("parent_card_id") == card_id
        or _card_id_from_locations(it) == card_id
    ]
    out.sort(key=lambda it: (str(it.get("created_at") or ""), str(it.get("id") or "")))
    return out


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
