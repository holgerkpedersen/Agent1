"""Session metadata storage and state-path resolution.

Sessions are directories under ``.sessions/`` at the repository root
(override with ``AGENT_SESSION_ROOT``).  Every session owns its chat history,
cross-session memory, turn log, and derived state files.  The ``default``
session is a compatibility alias for the historical repo-root files, so
existing installations keep their conversation without a migration step.

With the worktree layer enabled (``AGENT_SESSION_WORKTREES=1``) a named
session also owns a git worktree (``meta.worktree``): creation binds one,
deletion unbinds it (dirty trees are refused without ``force``), and
relocation moves it.  The store keeps the metadata honest — a session's
recorded ``workspace`` is the live worktree path when one is bound.

Storage is intentionally index-free: listing scans the session directories,
so concurrent processes never race on one shared index file.  Metadata writes
are atomic (tmp file + ``os.replace``).
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from agent_core.session import worktrees
from agent_core.session.models import (
    DEFAULT_SESSION_ID,
    SessionMeta,
    SessionPaths,
    utc_now,
)

_SLUG_RE = re.compile(r"[^a-z0-9._-]+")
_MAX_SLUG = 64


class SessionError(RuntimeError):
    """Base class for session-layer errors."""


class SessionNotFound(SessionError):
    """Raised when a requested session does not exist."""


class SessionExists(SessionError):
    """Raised when creating a session whose id is already taken."""


def repo_root() -> Path:
    """Return the Agent1 repository root."""
    return Path(__file__).resolve().parents[2]


def session_root(root: str | Path | None = None) -> Path:
    """Return the directory that holds session folders."""
    if root is None:
        env = os.environ.get("AGENT_SESSION_ROOT", "").strip()
        root = env or (repo_root() / ".sessions")
    return Path(root).expanduser().resolve()


def slugify(name: str) -> str:
    """Normalize a user-supplied session name into a safe directory id."""
    raw = (name or "").strip().lower()
    slug = _SLUG_RE.sub("-", raw).strip("-._")
    slug = re.sub(r"-{2,}", "-", slug)
    if not slug:
        raise SessionError("Session name must contain at least one letter or digit.")
    if len(slug) > _MAX_SLUG:
        raise SessionError(
            f"Session name is too long (max {_MAX_SLUG} characters after normalizing)."
        )
    if slug in (".", ".."):
        raise SessionError("Invalid session name.")
    return slug


def session_dir(session_id: str, root: str | Path | None = None) -> Path:
    return session_root(root) / session_id


def session_meta_path(session_id: str, root: str | Path | None = None) -> Path:
    return session_dir(session_id, root) / "session.json"


def _live_legacy_paths() -> dict[str, str]:
    """Resolve the historical repo-root state paths through the agent shim."""
    from agent_core.runtime_globals import live

    chat = str(
        live("CHAT_HISTORY_JSON_PATH")
        or (repo_root() / "chat_history.json")
    )
    memory = str(
        live("AGENT_MEMORY_JSON_PATH")
        or (repo_root() / "agent_memory.json")
    )
    turn_log = str(live("TURN_LOG_PATH") or (repo_root() / "turn_log.jsonl"))
    return {"chat_history": chat, "agent_memory": memory, "turn_log": turn_log}


def paths_for(
    session_id: str, root: str | Path | None = None
) -> SessionPaths:
    """Return the state paths for *session_id*.

    ``default`` keeps the legacy repo-root files.  Named sessions live under
    ``<session_root>/<id>/``.
    """
    sid = slugify(session_id) if session_id != DEFAULT_SESSION_ID else session_id
    if sid == DEFAULT_SESSION_ID:
        legacy = _live_legacy_paths()
        return SessionPaths(
            session_id=sid,
            state_dir=session_dir(sid, root),
            chat_history_path=legacy["chat_history"],
            agent_memory_path=legacy["agent_memory"],
            turn_log_path=legacy["turn_log"],
        )
    state = session_dir(sid, root)
    return SessionPaths(
        session_id=sid,
        state_dir=state,
        chat_history_path=str(state / "chat_history.json"),
        agent_memory_path=str(state / "agent_memory.json"),
        turn_log_path=str(state / "turn_log.jsonl"),
    )


def _read_meta_file(path: Path, fallback_id: str) -> SessionMeta:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return SessionMeta(id=fallback_id)
    except (OSError, json.JSONDecodeError):
        return SessionMeta(id=fallback_id)
    if not isinstance(data, dict):
        return SessionMeta(id=fallback_id)
    return SessionMeta.from_dict(data, fallback_id=fallback_id)


def load_meta(session_id: str, root: str | Path | None = None) -> SessionMeta:
    """Load session metadata, synthesizing it for a missing ``default``."""
    sid = slugify(session_id) if session_id != DEFAULT_SESSION_ID else session_id
    if sid != DEFAULT_SESSION_ID and not session_meta_path(sid, root).is_file():
        raise SessionNotFound(f"Session '{sid}' does not exist.")
    return _read_meta_file(session_meta_path(sid, root), sid)


def save_meta(meta: SessionMeta, root: str | Path | None = None) -> None:
    """Write ``session.json`` atomically."""
    target = session_meta_path(meta.id, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    meta.updated_at = utc_now()
    payload = json.dumps(meta.to_dict(), ensure_ascii=False, indent=2)
    tmp = target.with_name(target.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, target)
    except OSError as exc:
        raise SessionError(f"Could not save session metadata: {exc}") from exc


def _git_repo_for(root: str | Path | None, meta: SessionMeta) -> Path:
    """Repo root (git registry) to run worktree plumbing in for this session.

    Prefer the recorded main-checkout path (``meta.repo``); the session
    root's parent is the production fallback.  The worktree path is never
    used as cwd: on Windows, running git with the worktree as its own cwd
    locks the directory and makes the move/remove deny.
    """
    if meta.repo:
        return Path(meta.repo)
    return Path(session_root(root)).parent


def create_session(
    name: str,
    *,
    root: str | Path | None = None,
    workspace: str = "",
    model: str = "",
    provider: str = "",
    mode: str = "build",
    title: str = "",
    worktree: str = "",
    repo: str = "",
) -> SessionMeta:
    """Create a new session directory and metadata record.

    ``worktree`` is the absolute path of the session's dedicated git
    worktree (the caller binds it BEFORE this runs — a bind failure means
    no session), and ``repo`` the main checkout it is registered in.  When
    bound, the session's ``workspace`` is the worktree path; an empty
    ``worktree`` keeps the shared-workspace behaviour.
    """
    sid = slugify(name)
    if sid == DEFAULT_SESSION_ID:
        raise SessionExists("The 'default' session already exists.")
    if session_meta_path(sid, root).exists():
        raise SessionExists(f"Session '{sid}' already exists.")
    meta = SessionMeta(
        id=sid,
        title=title or sid,
        workspace=workspace,
        worktree=worktree,
        repo=repo,
        model=model,
        provider=provider,
        mode=mode or "build",
    )
    session_dir(sid, root).mkdir(parents=True, exist_ok=True)
    save_meta(meta, root)
    return meta


def update_session(
    session_id: str, root: str | Path | None = None, **fields: Any
) -> SessionMeta:
    """Update selected metadata fields and persist the result."""
    meta = load_meta(session_id, root)
    for key, value in fields.items():
        if hasattr(meta, key):
            setattr(meta, key, value)
    meta.updated_at = utc_now()
    save_meta(meta, root)
    return meta


def list_sessions(root: str | Path | None = None) -> list[SessionMeta]:
    """Return every session, newest ``updated_at`` first."""
    base = session_root(root)
    metas: list[SessionMeta] = []
    if base.is_dir():
        for child in base.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            meta_file = child / "session.json"
            if not meta_file.is_file():
                continue
            meta = _read_meta_file(meta_file, child.name)
            if meta.id != child.name:
                meta.id = child.name
            metas.append(meta)
    if not any(m.id == DEFAULT_SESSION_ID for m in metas):
        metas.append(load_meta(DEFAULT_SESSION_ID, root))
    metas.sort(key=lambda m: (m.updated_at, m.id), reverse=True)
    return metas


def rename_session(
    old_name: str,
    new_name: str,
    root: str | Path | None = None,
    *,
    move_worktree: bool = True,
) -> SessionMeta:
    """Rename a session directory and id (the default session is fixed).

    With ``move_worktree`` (default) the session's git worktree and its
    ``session/<id>`` branch are relocated FIRST — a refused move (dirty
    tree) aborts before the state directory is touched.
    """
    old_id = slugify(old_name) if old_name != DEFAULT_SESSION_ID else old_name
    new_id = slugify(new_name)
    if old_id == DEFAULT_SESSION_ID or new_id == DEFAULT_SESSION_ID:
        raise SessionError("The 'default' session cannot be renamed.")
    if not session_meta_path(old_id, root).is_file():
        raise SessionNotFound(f"Session '{old_id}' does not exist.")
    if session_meta_path(new_id, root).exists():
        raise SessionExists(f"Session '{new_id}' already exists.")
    moved = None
    repo = _git_repo_for(root, load_meta(old_id, root))
    if move_worktree and worktrees.worktree_path(old_id, repo).exists():
        try:
            moved = worktrees.move_worktree(old_id, new_id, repo=repo)
        except worktrees.WorktreeError as exc:
            raise SessionError(f"Worktree move refused: {exc}") from exc
    src = session_dir(old_id, root)
    dst = session_dir(new_id, root)
    src.rename(dst)
    meta = load_meta(new_id, root)
    meta.id = new_id
    meta.title = new_id
    if moved is not None:
        meta.worktree = str(moved)
        if meta.workspace:
            meta.workspace = str(moved)
    save_meta(meta, root)
    return meta


def delete_session(
    name: str,
    root: str | Path | None = None,
    *,
    allow_default: bool = False,
    remove_worktree: bool = True,
    force: bool = False,
) -> None:
    """Delete a session directory and all of its state files.

    The session's git worktree (if bound) is unbound FIRST: a dirty
    worktree is refused without ``force`` and the state directory stays
    untouched (fail closed).
    """
    sid = slugify(name) if name != DEFAULT_SESSION_ID else name
    if sid == DEFAULT_SESSION_ID and not allow_default:
        raise SessionError("The 'default' session cannot be deleted.")
    target = session_dir(sid, root)
    if not target.is_dir():
        raise SessionNotFound(f"Session '{sid}' does not exist.")
    if remove_worktree:
        repo = _git_repo_for(root, load_meta(sid, root))
        try:
            worktrees.unbind(sid, repo=repo, force=force)
        except worktrees.WorktreeError as exc:
            raise SessionError(f"Worktree removal refused: {exc}") from exc
    shutil.rmtree(target)


__all__ = [
    "SessionError",
    "SessionExists",
    "SessionNotFound",
    "create_session",
    "delete_session",
    "list_sessions",
    "load_meta",
    "paths_for",
    "rename_session",
    "repo_root",
    "save_meta",
    "session_dir",
    "session_meta_path",
    "session_root",
    "slugify",
    "update_session",
]
