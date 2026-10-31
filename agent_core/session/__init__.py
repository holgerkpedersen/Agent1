"""Per-process session layer for Agent1.

A session owns its chat history, cross-session memory, turn log, model choice,
and mode.  Session files are stored under ``.sessions/<id>/``; the ``default``
session aliases the historical repo-root files.  One live process may hold a
session at a time (enforced with an OS-level lock), which is what makes
side-by-side Agent1 instances safe to run.

With ``AGENT_SESSION_WORKTREES=1`` a new named session is additionally bound
to its own git worktree on a ``session/<id>`` branch (see ``worktrees``), so
concurrent sessions edit separate checkouts instead of one shared workspace.
"""

from agent_core.session import worktrees
from agent_core.session.context import (
    activate_session,
    active_paths,
    active_session,
    current_paths,
    current_session_id,
    deactivate_session,
    is_active,
    peers,
    reset,
    update_active_meta,
)
from agent_core.session.lock import (
    SessionLock,
    SessionLocked,
    live_peers,
    probe_owner,
    read_owner,
)
from agent_core.session.models import (
    DEFAULT_SESSION_ID,
    SESSION_VERSION,
    SessionMeta,
    SessionPaths,
    utc_now,
)
from agent_core.session.store import (
    SessionError,
    SessionExists,
    SessionNotFound,
    create_session,
    delete_session,
    list_sessions,
    load_meta,
    paths_for,
    rename_session,
    repo_root,
    save_meta,
    session_dir,
    session_meta_path,
    session_root,
    slugify,
    update_session,
)

__all__ = [
    "DEFAULT_SESSION_ID",
    "SESSION_VERSION",
    "SessionError",
    "SessionExists",
    "SessionLock",
    "SessionLocked",
    "SessionMeta",
    "SessionNotFound",
    "SessionPaths",
    "activate_session",
    "active_paths",
    "active_session",
    "create_session",
    "current_paths",
    "current_session_id",
    "deactivate_session",
    "delete_session",
    "is_active",
    "list_sessions",
    "live_peers",
    "load_meta",
    "paths_for",
    "peers",
    "probe_owner",
    "read_owner",
    "rename_session",
    "repo_root",
    "reset",
    "save_meta",
    "session_dir",
    "session_meta_path",
    "session_root",
    "slugify",
    "update_active_meta",
    "update_session",
    "utc_now",
    "worktrees",
]
