"""Per-session git worktrees: one checkout per named session.

Why this module exists
----------------------
Named sessions already isolate conversation, memory, model and mode, but
every session shares ONE working tree: concurrent sessions on one repo
edit the same files, last write wins.  A git *worktree* gives each
session its own checkout on a dedicated ``session/<id>`` branch: sessions
then edit physically separate trees while sharing the repository's object
database.  ``prune()`` cleans up dangling registrations; the owner-sidecar
peers model collapses to "sessions sharing one tree" (normally zero).

The module is pure git plumbing (no ``agent`` imports): it owns argv
construction and runs every git command the way ``tools/git_merge.py``
does — pinned cwd, the non-interactive ``merge_environment`` (no editor,
no credential prompt), ``stdin=DEVNULL``, hard timeouts, capped messages.

Policy
------
* Worktree creation is **opt-in**: ``AGENT_SESSION_WORKTREES=1`` binds
  new named sessions to their own worktree.  Everything here works
  regardless of the gate; the gate lives in the callers.
* Where worktrees live: ``AGENT_WORKTREE_ROOT`` when set, else
  ``<repo>/.worktrees``.  The root directory lives INSIDE the repo, so
  every recursive workspace walker must skip ``.worktrees/`` (it contains
  a full second checkout of the repository).
* Deletion is fail-closed: a dirty worktree refuses to be removed
  without ``force=True``.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

from agent_core.tools.git_merge import merge_environment

#: Opt-in gate: bind new named sessions to their own worktree.
WORKTREES_ENV = "AGENT_SESSION_WORKTREES"

#: Optional override for the directory that holds session worktrees.
WORKTREE_ROOT_ENV = "AGENT_WORKTREE_ROOT"

#: Default worktree container directory (under the main repo root).
WORKTREE_DIR_NAME = ".worktrees"

#: Every session branch is namespaced so it can never collide with user
#: branches: ``session/<id>``.
BRANCH_PREFIX = "session/"

#: What a fresh session branch starts from.
DEFAULT_BASE_REF = "HEAD"

#: Wall-clock budgets.  Read-only queries are probes; add/move/remove do
#: real I/O and get the longer budget.
LIST_TIMEOUT = 30.0
MUTATE_TIMEOUT = 120.0

_MAX_MSG = 2000

__all__ = [
    "BRANCH_PREFIX",
    "DEFAULT_BASE_REF",
    "LIST_TIMEOUT",
    "MUTATE_TIMEOUT",
    "WORKTREE_DIR_NAME",
    "WORKTREE_ROOT_ENV",
    "WORKTREES_ENV",
    "WorktreeError",
    "bind",
    "branch_name",
    "current_branch",
    "dirty_count",
    "is_dirty",
    "is_git_repo",
    "is_registered",
    "list_worktrees",
    "main_repo_root",
    "move_worktree",
    "prune",
    "unbind",
    "worktree_path",
    "worktree_root",
    "worktrees_enabled",
]


class WorktreeError(RuntimeError):
    """A git worktree operation failed (bad request, git error, timeout)."""


# ---------------------------------------------------------------------------
# Environment gates
# ---------------------------------------------------------------------------


def worktrees_enabled() -> bool:
    """True when ``AGENT_SESSION_WORKTREES`` asks for worktree binding."""
    raw = os.environ.get(WORKTREES_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def main_repo_root() -> Path:
    """The Agent1 repository root (the main worktree)."""
    return Path(__file__).resolve().parents[2]


def worktree_root(repo: str | Path | None = None) -> Path:
    """The directory that holds session worktrees.

    ``AGENT_WORKTREE_ROOT`` overrides; the default is ``<repo>/.worktrees``.
    """
    env = os.environ.get(WORKTREE_ROOT_ENV, "").strip()
    base = Path(env) if env else Path(repo or main_repo_root())
    return Path(base).expanduser()


def worktree_path(session_id: str, repo: str | Path | None = None) -> Path:
    """Where the worktree for *session_id* lives (registered or not)."""
    return worktree_root(repo) / session_id


def branch_name(session_id: str) -> str:
    """The dedicated branch for *session_id* (``session/<id>``)."""
    return f"{BRANCH_PREFIX}{session_id}"


def _validate_session_id(session_id: str) -> str:
    """Reject ids that could not be a directory name, never silently fix."""
    sid = str(session_id or "").strip()
    if not sid or sid in (".", "..") or "/" in sid or "\\" in sid:
        raise WorktreeError(f"Invalid session id for a worktree: {session_id!r}")
    if sid.startswith("-"):
        raise WorktreeError(
            f"Session id {sid!r} starts with '-' and would look like a git flag."
        )
    return sid


# ---------------------------------------------------------------------------
# Git plumbing
# ---------------------------------------------------------------------------


def _bounded(text: str) -> str:
    text = (text or "").strip()
    return text if len(text) <= _MAX_MSG else text[:_MAX_MSG] + " ...[truncated]"


def _run_git(
    repo: str | os.PathLike[str] | None,
    argv: list[str],
    timeout: float,
) -> tuple[int, str, str]:
    """Run *argv* in *repo* non-interactively, never blocking.

    Same contract as ``tools/git_merge._run_git``: closed stdin, every
    editor/prompt disabled, the timeout reported as a message (rc -1)
    rather than an exception.
    """
    cwd = str(repo) if repo is not None else str(main_repo_root())
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=merge_environment(),
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired:
        return -1, "", f"git timed out after {timeout:g}s"
    except OSError as e:  # git missing, cwd gone, ...
        return -1, "", f"git could not be started: {e}"


def _run_git_write(
    repo: str | os.PathLike[str] | None,
    argv: list[str],
) -> tuple[int, str, str]:
    """Run a worktree MUTATING command, retrying Windows transient locks.

    Windows raises ``Permission denied`` when a directory being moved or
    removed has a transient handle open (antivirus, indexer, a lingering
    shell).  The handle releases in milliseconds, so a few quick retries
    ride it out; a real refusal (dirty tree) is a deterministic git
    message, not a permission error, and fails immediately.
    """
    attempts = 3
    rc = out = err = None
    for attempt in range(attempts):
        rc, out, err = _run_git(repo, argv, MUTATE_TIMEOUT)
        if rc == 0 or "permission denied" not in (out + err).lower():
            return rc, out, err
        if attempt + 1 < attempts:
            time.sleep(0.3 * (attempt + 1))
    return rc, out, err  # type: ignore[return-value]


def _same_path(a: str | os.PathLike[str], b: str | os.PathLike[str]) -> bool:
    return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(
        os.path.abspath(str(b))
    )


def is_git_repo(path: str | os.PathLike[str]) -> bool:
    """True when *path* is inside a git work tree (main or linked)."""
    rc, out, _err = _run_git(
        path,
        ["git", "rev-parse", "--is-inside-work-tree"],
        LIST_TIMEOUT,
    )
    return rc == 0 and "true" in out.lower()


def list_worktrees(repo: str | os.PathLike[str] | None = None) -> list[dict[str, Any]]:
    """Parse ``git worktree list --porcelain`` into records.

    Each record: ``{"path", "head", "branch", "bare"}`` (``branch`` is the
    short name with ``refs/heads/`` stripped, ``""`` for detached).
    """
    rc, out, err = _run_git(
        repo, ["git", "worktree", "list", "--porcelain"], LIST_TIMEOUT
    )
    if rc != 0:
        raise WorktreeError(f"git worktree list failed: {_bounded(out or err)}")
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in out.splitlines():
        if not line.strip():
            continue
        key, _, value = line.partition(" ")
        if key == "worktree":
            if current is not None:
                entries.append(current)
            current = {"path": value.strip(), "head": "", "branch": "", "bare": False}
        elif current is None:
            continue
        elif key == "HEAD":
            current["head"] = value.strip()
        elif key == "branch":
            branch = value.strip()
            if branch.startswith("refs/heads/"):
                branch = branch[len("refs/heads/") :]
            current["branch"] = branch
        elif key == "bare":
            current["bare"] = True
    if current is not None:
        entries.append(current)
    return entries


def is_registered(
    session_id: str, repo: str | os.PathLike[str] | None = None
) -> bool:
    """True when the session's worktree path is a registered worktree."""
    target = worktree_path(session_id, repo)
    return any(_same_path(w["path"], target) for w in list_worktrees(repo))


def current_branch(path: str | os.PathLike[str]) -> str:
    """Short branch name checked out in the worktree at *path* ("" on error)."""
    rc, out, _err = _run_git(
        path, ["git", "rev-parse", "--abbrev-ref", "HEAD"], LIST_TIMEOUT
    )
    branch = out.strip() if rc == 0 else ""
    return "" if branch == "HEAD" else branch


def is_dirty(path: str | os.PathLike[str]) -> bool:
    """True when the worktree at *path* has uncommitted changes."""
    return dirty_count(path) > 0


def dirty_count(path: str | os.PathLike[str]) -> int:
    """Number of uncommitted changes (git status --porcelain lines)."""
    rc, out, _err = _run_git(path, ["git", "status", "--porcelain"], LIST_TIMEOUT)
    if rc != 0:
        return 0
    return len([ln for ln in out.splitlines() if ln.strip()])


def prune(repo: str | os.PathLike[str] | None = None) -> str:
    """Run ``git worktree prune -v`` and return its report ("" when quiet)."""
    rc, out, err = _run_git(repo, ["git", "worktree", "prune", "-v"], LIST_TIMEOUT)
    if rc != 0:
        raise WorktreeError(f"git worktree prune failed: {_bounded(out or err)}")
    return _bounded((out + "\n" + err).strip())


# ---------------------------------------------------------------------------
# Lifecycle actions
# ---------------------------------------------------------------------------


def bind(
    session_id: str,
    repo: str | os.PathLike[str] | None = None,
    base_ref: str = DEFAULT_BASE_REF,
) -> Path:
    """Create ``<root>/<id>`` as a worktree on a fresh ``session/<id>`` branch.

    The branch always starts from *base_ref* (default: the main checkout's
    HEAD), so a session begins from the shared, committed baseline.  Raises
    :class:`WorktreeError` when the directory exists, git fails, or git is
    unreachable — callers must treat a failure as "session not created".
    """
    sid = _validate_session_id(session_id)
    root = worktree_root(repo)
    root.mkdir(parents=True, exist_ok=True)
    dest = root / sid
    if dest.exists():
        raise WorktreeError(
            f"Worktree directory already exists: {dest} (session was "
            "previously created or the id collides)."
        )
    base = str(base_ref or DEFAULT_BASE_REF).strip() or DEFAULT_BASE_REF
    if base.startswith("-"):
        raise WorktreeError(f"Invalid base ref: {base!r}")
    argv = ["git", "worktree", "add", "-b", branch_name(sid), str(dest), base]
    rc, out, err = _run_git_write(repo, argv)
    if rc != 0:
        raise WorktreeError(
            f"git worktree add failed (exit {rc}): {_bounded(out or err)}"
        )
    return dest.resolve()


def unbind(
    session_id: str,
    repo: str | os.PathLike[str] | None = None,
    force: bool = False,
) -> bool:
    """Remove the session's worktree.  Returns False when it never existed.

    A dirty worktree refuses removal without *force* (fail closed: the
    session's uncommitted work is the user's, not ours to discard).
    """
    sid = _validate_session_id(session_id)
    dest = worktree_path(sid, repo)
    if not dest.exists():
        return False
    argv = ["git", "worktree", "remove"]
    if force:
        argv.append("--force")
    argv.append(str(dest))
    rc, out, err = _run_git_write(repo, argv)
    if rc != 0:
        raise WorktreeError(
            f"git worktree remove failed (exit {rc}): {_bounded(out or err)} "
            "— pass --force to discard uncommitted changes."
        )
    return True


def move_worktree(
    old_id: str, new_id: str, repo: str | os.PathLike[str] | None = None
) -> Path:
    """Relocate the worktree from ``<root>/<old>`` to ``<root>/<new>``.

    Best-effort renames the session branch in the same call.  Raises when
    the old worktree is missing, the target exists, git refuses, or the
    branch rename fails (a session whose worktree moved under a branch it
    cannot find is a session that will be pruned).
    """
    old = worktree_path(_validate_session_id(old_id), repo)
    new = worktree_path(_validate_session_id(new_id), repo)
    if not old.exists():
        raise WorktreeError(f"Worktree does not exist: {old}")
    if new.exists():
        raise WorktreeError(f"Worktree target already exists: {new}")
    new.parent.mkdir(parents=True, exist_ok=True)
    rc, out, err = _run_git_write(
        repo, ["git", "worktree", "move", str(old), str(new)]
    )
    if rc != 0:
        raise WorktreeError(
            f"git worktree move failed (exit {rc}): {_bounded(out or err)}"
        )
    rc, out, err = _run_git(
        repo,
        ["git", "branch", "-m", branch_name(old_id), branch_name(new_id)],
        LIST_TIMEOUT,
    )
    if rc != 0:
        raise WorktreeError(
            f"worktree moved, but the branch rename failed (exit {rc}): "
            f"{_bounded(out or err)}"
        )
    return new.resolve()
