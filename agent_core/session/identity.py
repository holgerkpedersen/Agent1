"""Stable agent identity for cross-system attribution (Kanban sync).

The Kanban bridge enqueues work records that the board renders.  For a human
reading the board, "an Agent1 did this" is not enough: they need to know
*which* session did it, and two sessions must be able to see who claimed a
card first.  This module resolves that identity once per process:

* :func:`session_id` — the active session slug (``AGENT1_SESSION_ID`` env /
  repo ``.env`` wins; otherwise the process-active session; otherwise
  ``default``).
* :func:`agent_label` — ``<slug>@<host>``, the human-readable owner stamp
  carried in every outbound payload.
* :func:`claim_owner` — who holds a claim on a card id (first writer wins),
  with liveness: a claim held by a session that is locked by a live local
  process stays valid; a stale claim expires after ``CLAIM_TTL_SECONDS`` so a
  crashed machine cannot block the board forever.

Claim records live in ``.card_claims.json`` at the repo root — one JSON
object keyed by card id, written atomically (tmp + ``os.replace``).  It is a
local mirror of the claim stamped on the outbound queue; the Kanban side is
the authority once it applies the payload (see the contract in
``.docs/2026-11-10_23-06-43/plan_proposed.md``).

Import discipline: this module imports ONLY stdlib at module level, so
``harnessfix`` can use it without pulling the agent_core session machinery
(no circular-import risk; ``agent_core.session.context`` imports the lock
layer, which is heavier and workspace-bound).
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Local claim mirror: one JSON object keyed by card id.  Module-level (not a
#: call inlined into functions) so tests can redirect it, same pattern as
#: ``kanban_bridge.CARD_ID_MAP_PATH``.
CLAIMS_PATH = REPO_ROOT / ".card_claims.json"

#: Claim records expire after this many seconds unless the holder is provably
#: alive (a live local session lock).  Chosen long enough to survive a normal
#: work session, short enough that a crashed machine releases its cards.
CLAIM_TTL_SECONDS = 24 * 3600

#: Grace applied when judging a claim we cannot probe for liveness — i.e. a
#: claim whose ``host`` is not this machine, because ``_holder_alive_local``
#: can only see session locks on the local box.  A remote worker that runs
#: longer than the TTL would otherwise be taken over mid-work, so a foreign-
#: host claim is treated as live while it is younger than TTL * this factor.
REMOTE_LIVENESS_GRACE = 2

_CLAIMS_LOCK = threading.RLock()

_SLUG_RE = re.compile(r"[^a-z0-9._-]+")


def _slugify(name: str) -> str:
    raw = (name or "").strip().lower()
    slug = _SLUG_RE.sub("-", raw).strip("-._")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug


def _read_env_file_value(key: str) -> str | None:
    """Value of *key* in the repo ``.env`` (same dialect as kanban_bridge)."""
    try:
        text = (REPO_ROOT / ".env").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        env_key, _, value = stripped.partition("=")
        if env_key.strip() == key:
            return value.strip().strip("\"'")
    return None


def _host() -> str:
    return os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or ""


def session_id() -> str:
    """The identity this process attributes its Kanban work to.

    Resolution order (first hit wins):

    1. ``AGENT1_SESSION_ID`` in the process env, else in the repo ``.env``
       (the operator-set per-process identity — a deployed session on another
       machine has no local session registry to consult);
    2. the process-active session id when one is activated;
    3. ``default`` (legacy single-session installs).
    """
    value = os.environ.get("AGENT1_SESSION_ID")
    if value is None:
        value = _read_env_file_value("AGENT1_SESSION_ID")
    slug = _slugify(value or "")
    if slug:
        return slug
    try:  # best-effort: the agent_core session layer may not be importable
        from agent_core.session.context import active_session, current_session_id

        if active_session() is not None:
            return current_session_id()
    except Exception:  # noqa: BLE001 — identity must never break the caller
        pass
    return "default"


def agent_label() -> str:
    """Human-readable owner stamp: ``<session-slug>[@<host>]``."""
    sid = session_id()
    host = _host()
    return f"{sid}@{host}" if host else sid


def claims_path() -> Path:
    """The local claim mirror (module-level constant so tests can redirect)."""
    return CLAIMS_PATH


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None = None) -> str:
    return (dt or _now()).isoformat(timespec="seconds")


def _load_claims(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_claims(claims: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(
            json.dumps(claims, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
    except OSError:
        pass


def _holder_alive_local(session_slug: str) -> bool:
    """True when a live local process holds the session lock for *session_slug*."""
    try:
        from agent_core.session.lock import probe_owner
        from agent_core.session.store import paths_for

        return probe_owner(paths_for(session_slug)) is not None
    except Exception:  # noqa: BLE001 — no registry (other machine) => not provable
        return False


def claim_owner(
    card_id: str, path: Path | None = None
) -> dict[str, Any] | None:
    """The live claim on *card_id*, or ``None`` when unclaimed/expired.

    A claim is live when the holder is this process's session, or a live local
    session lock proves the holder alive, or the claim is younger than
    :data:`CLAIM_TTL_SECONDS`.  Liveness cannot be probed across machines, so
    a claim stamped on another host gets :data:`REMOTE_LIVENESS_GRACE` times
    the TTL: we prefer to wait out an unprovable owner rather than take a card
    over while a remote session is still working on it.
    """
    p = path or claims_path()
    entry = _load_claims(p).get(card_id)
    if not isinstance(entry, dict):
        return None
    owner = str(entry.get("owner") or "")
    if not owner:
        return None
    if owner == session_id():
        return entry
    if _holder_alive_local(owner):
        return entry
    ttl = CLAIM_TTL_SECONDS
    if str(entry.get("host") or "") and str(entry.get("host")) != _host():
        ttl = CLAIM_TTL_SECONDS * REMOTE_LIVENESS_GRACE
    claimed_at = entry.get("claimed_at")
    if isinstance(claimed_at, str):
        try:
            ts = datetime.fromisoformat(claimed_at)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if _now() - ts <= timedelta(seconds=ttl):
                return entry
        except ValueError:
            pass
    return None


def claim_card(
    card_id: str, path: Path | None = None, *, force: bool = False
) -> tuple[bool, dict[str, Any]]:
    """First-writer-wins claim for *card_id* by this process's session.

    Returns ``(acquired, record)``.  When the card is live-claimed by another
    session, ``(False, their_record)`` — the caller decides whether to block.
    ``force=True`` re-stamps the claim (used by the REPL takeover flow only).
    """
    p = path or claims_path()
    with _CLAIMS_LOCK:
        existing = claim_owner(card_id, path=p)
        if existing is not None and not force:
            return False, existing
        record = {
            "owner": session_id(),
            "label": agent_label(),
            "claimed_at": _iso(),
            "host": _host(),
        }
        claims = _load_claims(p)
        claims[card_id] = record
        _save_claims(claims, p)
        return True, record


def release_card(card_id: str, path: Path | None = None) -> bool:
    """Drop this session's claim on *card_id* (no-op when held by another)."""
    p = path or claims_path()
    with _CLAIMS_LOCK:
        claims = _load_claims(p)
        entry = claims.get(card_id)
        if not isinstance(entry, dict):
            return False
        if str(entry.get("owner") or "") != session_id():
            return False
        claims.pop(card_id, None)
        _save_claims(claims, p)
        return True


def identity_payload() -> dict[str, Any]:
    """The fields stamped on every outbound Kanban payload.

    Field names are the cross-repo contract (Kanban reader persists them and
    renders the owner): ``agent1_session``, ``agent1_label``, ``claimed_at``.
    """
    return {
        "agent1_session": session_id(),
        "agent1_label": agent_label(),
        "claimed_at": _iso(),
    }
