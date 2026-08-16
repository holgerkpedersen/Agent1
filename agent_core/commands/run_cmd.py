"""`run <shell command>` — execute a shell command directly, LLM-free.

Purpose: deterministic, byte-exact operational access (e.g. dumping
harnessfix traces) without token cost, latency, or dependence on a working
LLM.  Reuses the guarded ``run`` tool path (blocked-command allowlist,
truncation), so it is exactly as safe as the LLM-driven tool — just without
the LLM.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Command

if TYPE_CHECKING:
    from agent import Agent


class RunCommand(Command):
    """Execute a shell command directly (LLM-free)."""

    @property
    def name(self) -> str:
        return "run"

    @property
    def help_text(self) -> str:
        return "run <command> - Execute a shell command directly (no LLM)"

    async def execute(self, args: list[str], agent: "Agent") -> bool:
        if not args:
            self.error("run requires a command.")
            return True
        command = " ".join(args)
        print(f"  $ {command}")
        output = await agent._execute_tool_call("run", {"command": command})
        print(output)
        return True
