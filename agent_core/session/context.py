"""Process-global active session.

An Agent1 process runs exactly one active session.  The session is selected
before the :class:`Agent` is constructed and can be switched between turns by
the REPL ``session`` command.

Path lookup falls back to the historical ``agent`` module globals when no
session is active.  That fallback is deliberate: the test suite redirects
runtime state by monkeypatching ``agent.CHAT_HISTORY_JSON_PATH`` (see
``conftest.py``), and every existing code path must keep honouring it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agent_core.session.lock import SessionLock, live_peers
from agent_core.session.models import DEFAULT_SESSION_ID, SessionMeta, SessionPaths
from agent_core.session.store import (
    load_meta,
    paths_for,
    save_meta,
    session_dir,
    slugify,
)

_ACTIVE_PATHS: SessionPaths | None = None
_ACTIVE_META: SessionMeta | None = None
_ACTIVE_LOCK: SessionLock | None = None


def current_paths() -> SessionPaths:
    """Return the active session's paths, or the legacy defaults."""
    if _ACTIVE_PATHS is not None:
        return _ACTIVE_PATHS
    return paths_for(DEFAULT_SESSION_ID)


def active_session() -> SessionMeta | None:
    """Return the active session metadata, or ``None`` for legacy/tests."""
    return _ACTIVE_META


def active_paths() -> SessionPaths | None:
    return _ACTIVE_PATHS


def current_session_id() -> str:
    return _ACTIVE_META.id if _ACTIVE_META is not None else DEFAULT_SESSION_ID


def is_active() -> bool:
    return _ACTIVE_META is not None


def activate_session(
    session_id: str,
    *,
    root: str | Path | None = None,
    workspace: str = "",
) -> SessionMeta:
    """Load and lock a session, making it the process-wide active session.

    Raises :class:`SessionError` when the session does not exist or is already
    locked by another live process.  The previous active session is released
    only after the new one has been acquired successfully.
    """
    global _ACTIVE_PATHS, _ACTIVE_META, _ACTIVE_LOCK

    sid = slugify(session_id) if session_id != DEFAULT_SESSION_ID else session_id
    meta = load_meta(sid, root)
    paths = paths_for(sid, root)
    if sid != DEFAULT_SESSION_ID:
        session_dir(sid, root).mkdir(parents=True, exist_ok=True)
    else:
        paths.state_dir.mkdir(parents=True, exist_ok=True)

    lock = SessionLock(paths, workspace)
    lock.acquire()

    old_lock = _ACTIVE_LOCK
    _ACTIVE_PATHS = paths
    _ACTIVE_META = meta
    _ACTIVE_LOCK = lock
    if old_lock is not None:
        old_lock.release()
    return meta


def deactivate_session() -> None:
    """Release the active session lock and clear process-global state."""
    global _ACTIVE_PATHS, _ACTIVE_META, _ACTIVE_LOCK
    lock, _ACTIVE_LOCK = _ACTIVE_LOCK, None
    _ACTIVE_PATHS = None
    _ACTIVE_META = None
    if lock is not None:
        lock.release()


def update_active_meta(**fields: Any) -> None:
    """Persist selected metadata fields for the active session."""
    global _ACTIVE_META
    if _ACTIVE_META is None:
        return
    for key, value in fields.items():
        if hasattr(_ACTIVE_META, key):
            setattr(_ACTIVE_META, key, value)
    save_meta(_ACTIVE_META)


def peers(*, exclude_self: bool = True) -> list[dict[str, Any]]:
    """Return other live sessions on the active workspace."""
    if _ACTIVE_META is None:
        return []
    workspace = _ACTIVE_META.workspace
    if not workspace:
        workspace = os.getcwd()
    return live_peers(
        workspace,
        root=None,
        exclude=_ACTIVE_META.id if exclude_self else None,
    )


def reset() -> None:
    """Release the lock and forget the active session (tests / shutdown)."""
    deactivate_session()


__all__ = [
    "activate_session",
    "active_paths",
    "active_session",
    "current_paths",
    "current_session_id",
    "deactivate_session",
    "is_active",
    "peers",
    "reset",
    "update_active_meta",
]
