"""Typed session metadata and resolved per-session state paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: v2 added the ``worktree`` field (per-session git worktree binding).
SESSION_VERSION = 2
DEFAULT_SESSION_ID = "default"


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp (seconds precision)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class SessionPaths:
    """Every mutable state file owned by one session.

    ``default`` points at the pre-session-layer repo-root files so existing
    chat history keeps loading unchanged.  Named sessions point inside their
    own directory under ``.sessions/``.
    """

    session_id: str
    state_dir: Path
    chat_history_path: str
    agent_memory_path: str
    turn_log_path: str

    @property
    def chat_history(self) -> Path:
        return Path(self.chat_history_path)

    @property
    def agent_memory(self) -> Path:
        return Path(self.agent_memory_path)

    @property
    def chat_history_tmp(self) -> str:
        return self.chat_history_path + ".tmp"

    @property
    def agent_memory_tmp(self) -> str:
        return self.agent_memory_path + ".tmp"

    @property
    def memory_db_path(self) -> str:
        return self.agent_memory_path.replace(".json", ".db")

    @property
    def prior_history_path(self) -> Path:
        return Path(self.chat_history_path).parent / "chat_history_prior.json"

    @property
    def meta_policy_path(self) -> Path:
        return Path(self.chat_history_path).parent / "meta_policy.json"

    @property
    def line_history_path(self) -> Path:
        return self.state_dir / ".agent_history"

    @property
    def lock_path(self) -> Path:
        return self.state_dir / ".lock"

    @property
    def owner_path(self) -> Path:
        return self.state_dir / ".lock.owner.json"


@dataclass
class SessionMeta:
    """Persistent description of a session (``session.json``)."""

    id: str
    title: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    workspace: str = ""
    #: Absolute path of the session's dedicated git worktree (v2), or ""
    #: when the session shares the main checkout (every pre-v2 session).
    worktree: str = ""
    #: Absolute path of the MAIN checkout (git registry root) the session's
    #: worktree is registered in (v2).  All worktree plumbing runs with this
    #: as cwd — never the worktree being moved/removed, whose open handle
    #: makes Windows deny the operation.
    repo: str = ""
    model: str = ""
    provider: str = ""
    mode: str = "build"
    turn_count: int = 0

    def __post_init__(self) -> None:
        if not self.title:
            self.title = self.id

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SESSION_VERSION,
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "workspace": self.workspace,
            "worktree": self.worktree,
            "repo": self.repo,
            "model": self.model,
            "provider": self.provider,
            "mode": self.mode,
            "turn_count": self.turn_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], fallback_id: str = "") -> "SessionMeta":
        session_id = str(data.get("id") or fallback_id).strip()
        if not session_id:
            session_id = DEFAULT_SESSION_ID
        try:
            turn_count = int(data.get("turn_count") or 0)
        except (TypeError, ValueError):
            turn_count = 0
        return cls(
            id=session_id,
            title=str(data.get("title") or session_id),
            created_at=str(data.get("created_at") or utc_now()),
            updated_at=str(data.get("updated_at") or utc_now()),
            workspace=str(data.get("workspace") or ""),
            worktree=str(data.get("worktree") or ""),
            repo=str(data.get("repo") or ""),
            model=str(data.get("model") or ""),
            provider=str(data.get("provider") or ""),
            mode=str(data.get("mode") or "build"),
            turn_count=max(0, turn_count),
        )


__all__ = [
    "DEFAULT_SESSION_ID",
    "SESSION_VERSION",
    "SessionMeta",
    "SessionPaths",
    "utc_now",
]
