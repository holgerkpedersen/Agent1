"""Command-line session selection for the Agent1 REPL.

Sessions isolate conversation, memory, model and mode.  They deliberately
do NOT isolate the checkout: every session works in the caller's
workspace.  An earlier layer bound each named session to its own git
worktree on a ``session/<id>`` branch; it was removed because a worktree
is just another registration (and branch) of the SAME repository — a
session that checked out, committed or switched branches moved the
repository state every other session reads, so isolation was an illusion
that also fragmented shared state.  See ``docs/SESSIONS.md``.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from agent_core.session.context import activate_session
from agent_core.session.lock import SessionLocked, probe_owner
from agent_core.session.models import DEFAULT_SESSION_ID
from agent_core.session.store import (
    SessionError,
    SessionExists,
    SessionNotFound,
    create_session,
    list_sessions,
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

    The *workspace* is the caller's (or the repo root): sessions share one
    checkout by design.  ``should_exit`` is true for informational commands
    (``--list-sessions``) and for a refused session (locked or missing); the
    caller should return.
    """
    if namespace.session_root:
        os.environ["AGENT_SESSION_ROOT"] = str(namespace.session_root)
    root = str(session_root())
    ws = os.path.abspath(workspace or str(repo_root()))

    if namespace.list_sessions:
        print(format_session_table(root))
        return StartupSelection(root=root, workspace=ws, should_exit=True)

    if namespace.new_session is not None:
        name = namespace.new_session.strip() or _generated_name(root)
        sid = slugify(name)
        try:
            meta = create_session(
                name,
                root=root,
                workspace=ws,
            )
        except (SessionError, SessionExists) as exc:
            print(f"[session] {exc}")
            return StartupSelection(root=root, workspace=ws, should_exit=True)
        sid = meta.id
    else:
        raw = namespace.session or os.environ.get("AGENT_SESSION") or DEFAULT_SESSION_ID
        try:
            sid = raw if raw == DEFAULT_SESSION_ID else slugify(raw)
        except SessionError as exc:
            print(f"[session] {exc}")
            return StartupSelection(root=root, workspace=ws, should_exit=True)

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
    return StartupSelection(session_id=sid, root=root, workspace=ws)


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
