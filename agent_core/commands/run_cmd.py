"""`run <shell command>` — execute a shell command directly, LLM-free.

Purpose: deterministic, byte-exact operational access (e.g. dumping
harnessfix traces) without token cost, latency, or dependence on a working
LLM.  Reuses the guarded ``run`` tool path (blocked-command allowlist,
process-tree-killing timeout, truncation), so it is exactly as safe as the
LLM-driven tool — just without the LLM.

Long-running jobs (repair loops, benchmarks) routinely exceed the tool's
120s default, so the command defaults to 600s and accepts an explicit
``--timeout <sec>`` flag in front of the command.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Command

if TYPE_CHECKING:
    from agent import Agent

#: The run tool's own default is 120s — far too short for repair loops.
_DEFAULT_TIMEOUT = 900


class RunCommand(Command):
    """Execute a shell command directly (LLM-free)."""

    @property
    def name(self) -> str:
        return "run"

    @property
    def help_text(self) -> str:
        return (
            "run <command> [--timeout <sec>] - Execute a shell command directly "
            f"(no LLM, default timeout {_DEFAULT_TIMEOUT}s)"
        )

    async def execute(self, args: list[str], agent: "Agent") -> bool:
        if not args:
            self.error("run requires a command.")
            return True
        timeout = _DEFAULT_TIMEOUT
        if args[0] == "--timeout":
            if len(args) < 2:
                self.error("--timeout expects seconds (a number).")
                return True
            try:
                timeout = max(1, int(args[1]))
            except ValueError:
                self.error("--timeout expects seconds (a number).")
                return True
            args = args[2:]
        if not args:
            self.error("run requires a command.")
            return True
        command = " ".join(args)
        print(f"  $ {command}   (timeout {timeout}s)")
        output = await agent._execute_tool_call(
            "run", {"command": command, "timeout": timeout}
        )
        print(output)
        return True
