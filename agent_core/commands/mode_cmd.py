"""Mode command — switch the session between build and plan modes.

Mirrors opencode's mode switching: ``mode`` (or ``mode show``) prints the
active mode, ``mode plan`` / ``mode build`` switch it.  Plan mode is
enforced in the NLP tool loop (schema filtering + executor rejection, see
:mod:`agent_core.modes`), so while it is active no file in the workspace
can be changed by the agent.
"""
from __future__ import annotations

from .base import Command

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent import Agent


class ModeCommand(Command):
    """Show or switch the agent's session mode."""

    @property
    def name(self) -> str:
        return "mode"

    @property
    def help_text(self) -> str:
        return (
            "mode [build|plan] - Show or set the session mode\n"
            "  plan  read-only research mode — mutating tools are blocked;\n"
            "        ask questions and end with a plan as text\n"
            "  build default mode — full toolset (writes allowed)\n"
            "  Mode is per-session state; a restart returns to build."
        )

    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        if not args or args[0].lower() in ("show", "status"):
            label = "plan" if agent.is_plan_mode() else "build"
            print(f"Current mode: {label}")
            if agent.is_plan_mode():
                print(
                    "  Read-only: write/edit/run/git/tests/fix/analyze are "
                    "blocked. Switch back with 'mode build'."
                )
            return True

        requested = args[0].strip().lower()
        if requested in ("plan", "build"):
            agent.set_mode(requested)
            if requested == "plan":
                print(
                    "Mode set to PLAN — read-only. Mutating tools "
                    "(write/edit/run/git/tests/fix/analyze) are blocked for "
                    "the NLP tool loop. Ask your question; end with a plan."
                )
            else:
                print("Mode set to BUILD — full toolset restored.")
        else:
            self.error("Usage: mode [build|plan]")
        return True
