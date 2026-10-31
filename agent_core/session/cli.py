"""Command-line session selection for the Agent1 REPL.

The effective *workspace* follows the selected session: with the worktree
layer enabled (``AGENT_SESSION_WORKTREES=1``), a new named session is
bound to its own git worktree on a ``session/<id>`` branch, and resuming
a session lands in its worktree — each session edits its own checkout.
Without the layer (or without git), sessions share the caller's
workspace exactly as before.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from agent_core.session import worktrees
from agent_core.session.context import activate_session
from agent_core.session.lock import SessionLocked, probe_owner
from agent_core.session.models import DEFAULT_SESSION_ID
from agent_core.session.store import (
    SessionError,
    SessionExists,
    SessionNotFound,
    create_session,
    list_sessions,
    load_meta,
    paths_for,
    repo_root,
    session_root,
    slugify,
)


@dataclass
class StartupSelection:
    """Result of parsing/activating the startup session."""

    session_id: str = DEFAULT_SESSION_ID
    root: str = ""
    workspace: str = ""
    worktree: str = ""
    should_exit: bool = False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Agent1 interactive assistant with isolated chat sessions.",
        add_help=True,
    )
    parser.add_argument(
        "--session",
        metavar="NAME",
        help="Resume the named chat session (default: AGENT_SESSION or 'default').",
    )
    parser.add_argument(
        "--new-session",
        metavar="NAME",
        nargs="?",
        const="",
        default=None,
        help="Create and enter a new session (a timestamped name is generated "
        "when NAME is omitted).",
    )
    parser.add_argument(
        "--base",
        metavar="REF",
        default="HEAD",
        help="With --new-session and the worktree layer: the ref the new "
        "session/<id> branch starts from (default: HEAD).",
    )
    parser.add_argument(
        "--no-worktree",
        action="store_true",
        help="Do not bind a new session to its own git worktree, even when "
        "the worktree layer is enabled.",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List available sessions and exit.",
    )
    parser.add_argument(
        "--session-root",
        metavar="PATH",
        default=None,
        help="Directory holding session folders (default: .sessions or "
        "AGENT_SESSION_ROOT).",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Start the web dashboard alongside the REPL.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run only the dashboard server.",
    )
    parser.add_argument("--port", type=int, default=None, help="Dashboard port.")
    return parser


def _generated_name(root: str) -> str:
    base = f"chat-{datetime.now():%Y%m%d-%H%M%S}"
    candidate = base
    n = 2
    while paths_for(candidate, root).chat_history.exists():
        candidate = f"{base}-{n}"
        n += 1
    return candidate


def format_session_table(root: str | None = None) -> str:
    """Render the session list for ``--list-sessions`` / REPL ``session list``."""
    rows = []
    for meta in list_sessions(root):
        owner = probe_owner(paths_for(meta.id, root))
        lock = f"locked pid={owner.get('pid')}" if owner else "free"
        model = meta.model or "-"
        rows.append(
            (
                meta.id,
                meta.updated_at,
                str(meta.turn_count),
                model,
                lock,
            )
        )
    if not rows:
        return "No sessions."
    header = ("ID", "UPDATED", "TURNS", "MODEL", "LOCK")
    widths = [
        max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))
    ]
    lines = ["  ".join(str(h).ljust(widths[i]) for i, h in enumerate(header))]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append(
            "  ".join(str(v).ljust(widths[i]) for i, v in enumerate(row))
        )
    return "\n".join(lines)


def resolve_startup(
    namespace: argparse.Namespace,
    *,
    workspace: str | None = None,
) -> StartupSelection:
    """Create/resume/lock the session selected on the command line.

    The workspace follows the session: a bound worktree wins over the
    caller's workspace when it exists (a pruned worktree degrades to the
    main checkout with a note).  ``should_exit`` is true for informational
    commands (``--list-sessions``) and for a refused session (locked,
    missing, or a worktree bind that failed); the caller should return.
    """
    if namespace.session_root:
        os.environ["AGENT_SESSION_ROOT"] = str(namespace.session_root)
    root = str(session_root())
    ws = os.path.abspath(workspace or str(repo_root()))
    use_worktree = worktrees.worktrees_enabled() and not namespace.no_worktree

    if namespace.list_sessions:
        print(format_session_table(root))
        return StartupSelection(root=root, workspace=ws, should_exit=True)

    worktree = ""
    main_repo = ws
    if namespace.new_session is not None:
        name = namespace.new_session.strip() or _generated_name(root)
        sid = slugify(name)
        if use_worktree:
            if not worktrees.is_git_repo(ws):
                print(
                    "[session] workspace is not a git repo — the session "
                    "shares the main checkout (no worktree)."
                )
            else:
                try:
                    worktree = str(
                        worktrees.bind(sid, repo=main_repo, base_ref=namespace.base)
                    )
                except worktrees.WorktreeError as exc:
                    print(f"[session] Worktree bind failed: {exc}")
                    return StartupSelection(root=root, workspace=ws, should_exit=True)
                ws = worktree
        try:
            meta = create_session(
                name,
                root=root,
                workspace=ws,
                worktree=worktree,
                repo=main_repo if worktree else "",
            )
        except (SessionError, SessionExists) as exc:
            print(f"[session] {exc}")
            if worktree:
                try:
                    worktrees.unbind(sid, repo=main_repo)
                except worktrees.WorktreeError as exc2:
                    print(f"[session] worktree cleanup also failed: {exc2}")
            return StartupSelection(root=root, workspace=ws, should_exit=True)
        sid = meta.id
    else:
        raw = namespace.session or os.environ.get("AGENT_SESSION") or DEFAULT_SESSION_ID
        try:
            sid = raw if raw == DEFAULT_SESSION_ID else slugify(raw)
        except SessionError as exc:
            print(f"[session] {exc}")
            return StartupSelection(root=root, workspace=ws, should_exit=True)

    # A session's own worktree wins over the caller's workspace.
    recorded = ""
    try:
        recorded = load_meta(sid, root).worktree
    except SessionError:
        pass
    if recorded and os.path.isdir(recorded):
        ws = os.path.abspath(recorded)
    elif recorded:
        print(
            f"[session] Worktree {recorded} is missing; continuing in "
            f"{ws} (run 'session prune' to clean up)."
        )

    try:
        activate_session(sid, root=root, workspace=ws)
    except SessionLocked as exc:
        print(f"[session] {exc}")
        print("[session] Choose another session with --session NAME.")
        return StartupSelection(root=root, workspace=ws, should_exit=True)
    except SessionNotFound:
        print(f"[session] Session '{sid}' does not exist.")
        print("[session] Create it with --new-session NAME or use --list-sessions.")
        return StartupSelection(root=root, workspace=ws, should_exit=True)
    except SessionError as exc:
        print(f"[session] {exc}")
        return StartupSelection(root=root, workspace=ws, should_exit=True)
    return StartupSelection(
        session_id=sid, root=root, workspace=ws, worktree=recorded or worktree
    )


def parse_and_activate(
    argv: Sequence[str] | None = None,
    *,
    workspace: str | None = None,
) -> StartupSelection:
    """Parse an argv list and activate the requested session."""
    parser = build_arg_parser()
    namespace, _unknown = parser.parse_known_args(argv)
    return resolve_startup(namespace, workspace=workspace)


__all__ = [
    "StartupSelection",
    "build_arg_parser",
    "format_session_table",
    "parse_and_activate",
    "resolve_startup",
]
