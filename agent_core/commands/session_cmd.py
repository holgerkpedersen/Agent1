"""Session command - list, create, switch, rename, delete, and prune sessions.

With the worktree layer enabled (``AGENT_SESSION_WORKTREES=1``) each named
session owns its own git worktree/branch; this command surfaces the
worktree state and the ``prune`` maintenance action.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from agent_core.session import worktrees
from agent_core.session import (
    SessionError,
    SessionLocked,
    create_session,
    delete_session,
    list_sessions,
    load_meta,
    paths_for,
    probe_owner,
    rename_session,
    slugify,
)

from .base import Command

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agent import Agent


_MUTATING = {
    "new",
    "use",
    "switch",
    "resume",
    "rename",
    "delete",
    "remove",
    "rm",
    "prune",
}


class SessionCommand(Command):
    """Manage isolated chat sessions."""

    @property
    def name(self) -> str:
        return "session"

    @property
    def help_text(self) -> str:
        return (
            "session [list|current|info <id>|peers|new <id> [--base REF]|"
            "use <id>|rename <old> <new>|delete <id> [--force]|prune] - "
            "Manage chat sessions (named sessions can own a git worktree)"
        )

    async def execute(self, args: list[str], agent: "Agent") -> bool:
        sub = args[0].strip().lower() if args else "list"
        rest = args[1:] if len(args) > 1 else []

        if sub in _MUTATING and agent.is_plan_mode():
            print(
                "  [plan] Session changes are blocked in plan mode. "
                "Switch with 'mode build' first."
            )
            return True

        if sub in ("list", "ls", "sessions"):
            self._list(agent)
            return True
        if sub in ("current", "show", "status"):
            self._current(agent)
            return True
        if sub == "info":
            self._info(agent, rest)
            return True
        if sub in ("peers", "who"):
            self._peers(agent)
            return True
        if sub == "new":
            self._new(agent, rest)
            return True
        if sub in ("use", "switch", "resume"):
            self._use(agent, rest)
            return True
        if sub == "rename":
            self._rename(agent, rest)
            return True
        if sub in ("delete", "remove", "rm"):
            self._delete(agent, rest)
            return True
        if sub == "prune":
            self._prune(agent)
            return True
        print(f"  Unknown session subcommand '{sub}'.")
        print(f"  {self.help_text}")
        return True

    def _list(self, agent: "Agent") -> None:
        active = agent.session_id
        rows = []
        for meta in list_sessions():
            owner = probe_owner(paths_for(meta.id))
            marks = []
            if meta.id == active:
                marks.append("active")
            if owner:
                marks.append(f"locked pid={owner.get('pid')}")
            rows.append(
                (
                    meta.id,
                    meta.updated_at,
                    str(meta.turn_count),
                    meta.model or "-",
                    meta.worktree or "-",
                    ",".join(marks) or "free",
                )
            )
        if not rows:
            print("  No sessions.")
            return
        header = ("ID", "UPDATED", "TURNS", "MODEL", "WORKTREE", "STATUS")
        widths = [
            max(len(str(row[i])) for row in rows + [header])
            for i in range(len(header))
        ]
        print("  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
        print("  " + "  ".join("-" * w for w in widths))
        for row in rows:
            print(
                "  "
                + "  ".join(str(v).ljust(widths[i]) for i, v in enumerate(row))
            )

    def _current(self, agent: "Agent") -> None:
        meta = load_meta(agent.session_id)
        paths = paths_for(meta.id)
        print(f"  Session:   {meta.id}")
        print(f"  Title:     {meta.title}")
        print(f"  Workspace: {meta.workspace or agent.workspace}")
        wt = meta.worktree
        if wt:
            missing = "" if os.path.isdir(wt) else "  [MISSING]"
            branch = worktrees.current_branch(wt) if os.path.isdir(wt) else ""
            print(f"  Worktree:  {wt}  [{branch}]{missing}")
        print(f"  Model:     {agent.llm.model_name}")
        print(f"  Mode:      {'plan' if agent.is_plan_mode() else 'build'}")
        print(f"  Updated:   {meta.updated_at}")
        print(f"  Chat file: {paths.chat_history_path}")
        print(f"  Memory:    {paths.agent_memory_path}")

    def _info(self, agent: "Agent", args: list[str]) -> None:
        if not args:
            print("  Usage: session info <id>")
            return
        try:
            meta = load_meta(args[0])
        except SessionError as exc:
            print(f"  {exc}")
            return
        paths = paths_for(meta.id)
        owner = probe_owner(paths)
        print(f"  Session:   {meta.id}")
        print(f"  Title:     {meta.title}")
        print(f"  Workspace: {meta.workspace or '-'}")
        wt = meta.worktree
        if wt:
            alive = os.path.isdir(wt)
            branch = worktrees.current_branch(wt) if alive else ""
            dirty = "  [MISSING]" if not alive else ""
            print(f"  Worktree:  {wt}  [{branch}]{dirty}")
            if alive:
                print(f"  Dirty:     {worktrees.dirty_count(wt)} file(s)")
        print(f"  Model:     {meta.model or '-'}")
        print(f"  Provider:  {meta.provider or '-'}")
        print(f"  Mode:      {meta.mode}")
        print(f"  Turns:     {meta.turn_count}")
        print(f"  Created:   {meta.created_at}")
        print(f"  Updated:   {meta.updated_at}")
        print(f"  Locked:    {'pid ' + str(owner.get('pid')) if owner else 'no'}")
        print(f"  Chat file: {paths.chat_history_path}")

    def _peers(self, agent: "Agent") -> None:
        from agent_core.session.context import peers

        found = peers()
        if not found:
            print("  No other live sessions on this workspace.")
            return
        for owner in found:
            print(
                f"  {owner.get('session')}: pid {owner.get('pid')} on "
                f"{owner.get('workspace')}"
            )

    def _new(self, agent: "Agent", args: list[str]) -> None:
        if not args:
            print("  Usage: session new <id> [--base REF]")
            return
        name = args[0]
        base = "HEAD"
        no_worktree = "--no-worktree" in args
        if "--base" in args:
            idx = args.index("--base")
            if idx + 1 < len(args):
                base = args[idx + 1]
        sid = slugify(name)
        ws = agent.workspace
        worktree = ""
        if worktrees.worktrees_enabled() and not no_worktree:
            if not worktrees.is_git_repo(ws):
                print(
                    "  [session] workspace is not a git repo — the session "
                    "shares the main checkout (no worktree)."
                )
            else:
                try:
                    worktree = str(
                        worktrees.bind(sid, base_ref=base)
                    )
                except worktrees.WorktreeError as exc:
                    print(f"  [session] Worktree bind failed: {exc}")
                    return
                ws = worktree
        try:
            meta = create_session(
                name,
                workspace=ws,
                worktree=worktree,
                model=agent.llm.model_name,
                provider=getattr(agent, "_session_provider_name", lambda: "")(),
            )
            agent.switch_session(meta.id)
        except (SessionError, SessionLocked) as exc:
            print(f"  {exc}")
            if worktree:
                try:
                    worktrees.unbind(sid)
                except worktrees.WorktreeError as exc2:
                    print(f"  [session] worktree cleanup also failed: {exc2}")
            return
        extra = (
            f" on branch {worktrees.branch_name(sid)} ({worktree})"
            if worktree
            else ""
        )
        print(f"  Created and switched to session '{meta.id}'{extra}.")

    def _use(self, agent: "Agent", args: list[str]) -> None:
        if not args:
            print("  Usage: session use <id>")
            return
        try:
            target = args[0] if args[0] == "default" else slugify(args[0])
            agent.switch_session(target)
        except (SessionError, SessionLocked) as exc:
            print(f"  {exc}")
            return
        print(f"  Switched to session '{agent.session_id}'.")

    def _rename(self, agent: "Agent", args: list[str]) -> None:
        if len(args) < 2:
            print("  Usage: session rename <old> <new>")
            return
        old = args[0] if args[0] == "default" else slugify(args[0])
        if old == agent.session_id:
            print("  Switch to another session before renaming the active one.")
            return
        owner = probe_owner(paths_for(old))
        if owner:
            print(f"  Session '{old}' is locked by pid {owner.get('pid')}.")
            return
        try:
            meta = rename_session(old, args[1])
        except SessionError as exc:
            print(f"  {exc}")
            return
        print(f"  Renamed session '{old}' -> '{meta.id}'.")

    def _delete(self, agent: "Agent", args: list[str]) -> None:
        if not args:
            print("  Usage: session delete <id> [--force]")
            return
        target = args[0] if args[0] == "default" else slugify(args[0])
        if target == agent.session_id:
            print("  Cannot delete the active session.")
            return
        owner = probe_owner(paths_for(target))
        if owner:
            print(f"  Session '{target}' is locked by pid {owner.get('pid')}.")
            return
        force = "--force" in args
        meta = None
        try:
            meta = load_meta(target)
        except SessionError:
            pass
        wt = meta.worktree if meta is not None else ""
        if not force:
            wt_note = f" (and its worktree {wt})" if wt else ""
            confirm = input(
                f"Delete session '{target}' and all its state{wt_note}? (y/n): "
            )
            if confirm.strip().lower() not in ("y", "yes"):
                print("  Cancelled.")
                return
        try:
            delete_session(target, remove_worktree=True, force=force)
        except SessionError as exc:
            print(f"  {exc}")
            return
        print(f"  Deleted session '{target}'.")

    def _prune(self, agent: "Agent") -> None:
        """Report and remove dangling worktree registrations."""
        try:
            report = worktrees.prune()
        except worktrees.WorktreeError as exc:
            print(f"  [session] {exc}")
            return
        if report:
            print(f"  Pruned worktrees:\n{report}")
        else:
            print("  No dangling worktrees to prune.")
