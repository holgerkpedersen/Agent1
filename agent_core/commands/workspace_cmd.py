"""Workspace command — show or switch the directory the agent works on.

``workspace`` shows the root in force, ``workspace <path>`` switches it and
``workspace reset`` returns to the resolved default.

The switch only sets ``agent._nlp_workspace``: every NLP tool already resolves
its working directory through :func:`agent_core.handlers.nlp_tools.
_effective_ws_dir`, which prefers that override, so no tool needs to know this
command exists.  After the switch the system prompt is rebuilt so the model
sees the new root on the very next turn instead of inferring it from tool
results — that inference is what produced edits and test runs aimed at the
wrong repository.
"""

from __future__ import annotations

import os

from .base import Command

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent import Agent


class WorkspaceCommand(Command):
    """Show or switch the agent's working root."""

    @property
    def name(self) -> str:
        return "workspace"

    @property
    def help_text(self) -> str:
        return (
            "workspace [path|reset] - Show or switch the working root\n"
            "  workspace        show the effective root and its git branch\n"
            "  workspace <path> work in <path> from the next tool call on\n"
            "  workspace reset  return to the default root\n"
            "  The model is told the root in its system prompt."
        )

    def _report(self, agent: "Agent", note: str = "") -> None:
        """Print the root now in force (plus *note* when given)."""
        from agent_core.handlers.nlp_tools import _effective_ws_dir
        from agent_core.prompts.context_blocks import _git_branch_label

        effective = _effective_ws_dir(agent)
        print(f"Workspace: {effective}")
        branch = _git_branch_label(effective)
        if branch:
            print(f"  git branch: {branch}")
        else:
            print("  not a git working tree")
        if note:
            print(f"  {note}")

    async def execute(self, args: list[str], agent: "Agent") -> bool:
        requested = " ".join(args).strip().strip('"').strip("'")
        if not requested or requested.lower() in ("show", "status"):
            self._report(agent, f"started from {os.getcwd()}")
            return True

        if requested.lower() == "reset":
            agent._nlp_workspace = None
            agent._refresh_system_message()
            getattr(agent, "_snapshot_session_meta", lambda: None)()
            self._report(agent, "reset to the default root.")
            return True

        path = os.path.abspath(os.path.expanduser(requested))
        if not os.path.isdir(path):
            self.error(f"not a directory: {path} (workspace unchanged)")
            return False

        agent._nlp_workspace = path
        # Rebuild now: the WORKSPACE block is read from the prompt history on
        # the next turn, so a stale block would keep advertising the old root.
        agent._refresh_system_message()
        getattr(agent, "_snapshot_session_meta", lambda: None)()
        self._report(agent, "switched.")
        return True
