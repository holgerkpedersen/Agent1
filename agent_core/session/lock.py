"""Exclusive per-session process locks.

Two Agent1 processes must never write the same session's state files.  Each
session acquires an OS-level lock for the lifetime of the process; the OS
releases it automatically when the process exits, so a crashed process cannot
leave a permanently stuck session.

The lock file is empty and is only used for locking.  Human-readable owner
metadata lives in a sidecar ``.lock.owner.json`` so readers can report which
PID owns a session without touching the locked region.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from agent_core.session.models import SessionPaths, utc_now
from agent_core.session.store import SessionError, list_sessions, paths_for

#: Process-wide guard for lock bookkeeping (not the OS lock itself).
_BOOKKEEPING = threading.RLock()


class SessionLocked(SessionError):
    """Raised when another live process already owns the session."""

    def __init__(self, session_id: str, owner: dict[str, Any] | None = None):
        self.session_id = session_id
        self.owner = owner or {}
        pid = self.owner.get("pid")
        detail = f" (PID {pid})" if pid else ""
        super().__init__(
            f"Session '{session_id}' is already in use by another Agent1 "
            f"process{detail}."
        )


def _lock_fd(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(  # type: ignore[attr-defined]
            fd, fcntl.LOCK_EX | fcntl.LOCK_NB  # type: ignore[attr-defined]
        )


def _unlock_fd(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)  # type: ignore[attr-defined]


def _open_lock_file(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
    if os.fstat(fd).st_size == 0:
        os.write(fd, b"\0")
    return fd


def read_owner(paths: SessionPaths) -> dict[str, Any]:
    """Read the owner sidecar; empty dict when absent or unreadable."""
    try:
        with open(paths.owner_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_owner(paths: SessionPaths, workspace: str) -> None:
    payload = {
        "session": paths.session_id,
        "pid": os.getpid(),
        "workspace": workspace,
        "started_at": utc_now(),
        "host": os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "",
    }
    tmp = paths.owner_path.with_name(paths.owner_path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, paths.owner_path)
    except OSError:
        pass


class SessionLock:
    """Hold an exclusive lock for one session until :meth:`release`."""

    def __init__(self, paths: SessionPaths, workspace: str = ""):
        self.paths = paths
        self.workspace = workspace
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self) -> None:
        if self._fd is not None:
            return
        fd = _open_lock_file(self.paths.lock_path)
        try:
            _lock_fd(fd)
        except OSError as exc:
            owner = read_owner(self.paths)
            os.close(fd)
            raise SessionLocked(self.paths.session_id, owner) from exc
        self._fd = fd
        _write_owner(self.paths, self.workspace)

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            _unlock_fd(fd)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            self.paths.owner_path.unlink()
        except OSError:
            pass

    def __enter__(self) -> "SessionLock":
        self.acquire()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.release()


def probe_owner(paths: SessionPaths) -> dict[str, Any] | None:
    """Return owner metadata when *paths* is locked by a live process."""
    if not paths.lock_path.is_file():
        return None
    fd = _open_lock_file(paths.lock_path)
    try:
        _lock_fd(fd)
    except OSError:
        try:
            os.close(fd)
        except OSError:
            pass
        return read_owner(paths) or {"session": paths.session_id}
    try:
        _unlock_fd(fd)
    except OSError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass
    return None


def live_peers(
    workspace: str,
    *,
    root: str | Path | None = None,
    exclude: str | None = None,
) -> list[dict[str, Any]]:
    """Return live session owners on the same workspace."""
    peers: list[dict[str, Any]] = []
    target = os.path.normcase(os.path.abspath(workspace))
    with _BOOKKEEPING:
        for meta in list_sessions(root):
            if exclude and meta.id == exclude:
                continue
            try:
                paths = paths_for(meta.id, root)
            except Exception:
                continue
            owner = probe_owner(paths)
            if not owner:
                continue
            owner_ws = str(owner.get("workspace") or meta.workspace or "")
            if owner_ws and os.path.normcase(os.path.abspath(owner_ws)) != target:
                continue
            owner["session"] = owner.get("session") or meta.id
            peers.append(owner)
    return peers


__all__ = [
    "SessionLock",
    "SessionLocked",
    "live_peers",
    "probe_owner",
    "read_owner",
]
