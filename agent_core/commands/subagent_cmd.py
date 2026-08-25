"""Command for creating and managing subagents interactively."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent import Agent


class SubAgentCommand:
    """Create subagents to parallelize independent work.

    Usage::

        /subagent create <name> [--workspace <path>]   Create a named subagent
        /subagent run <name> <task>                    Run a task in an existing subagent
        /subagent list                                  List all active subagents
        /subagent summary <name>                       Show condensed context for a subagent
        /subagent reset <name>                         Clear a subagent's conversation history
    """

    NAME = "subagent"
    HELP = "Create and manage subagents for parallel work"

    # Storage for active subagents keyed by name.
    _registry: dict[str, Any] = {}  # populated lazily per-agent instance

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        if not hasattr(agent, "_subagents"):
            agent._subagents: dict[str, Any] = {}

    def execute(self, args: str) -> str:
        """Parse *args* and dispatch to the appropriate subcommand."""
        parts = args.strip().split(None, 1)
        if not parts:
            return self._help()

        cmd = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

        handlers = {
            "create": self._cmd_create,
            "run": self._cmd_run,
            "list": self._cmd_list,
            "summary": self._cmd_summary,
            "reset": self._cmd_reset,
            "help": self._help,
        }

        handler = handlers.get(cmd)
        if handler is None:
            return f"Unknown subagent command '{cmd}'. Try 'subagent help'."
        return handler(rest)

    def _cmd_create(self, args: str) -> str:
        """Create a new subagent."""
        parts = args.strip().split(None, 1)
        if not parts:
            return "Usage: /subagent create <name> [--workspace <path>]"

        name = parts[0]
        workspace: str | None = None

        # Parse optional --workspace flag.
        if len(parts) > 1 and parts[1].startswith("--workspace"):
            ws_parts = parts[1].split(None, 1)
            if len(ws_parts) > 1:
                workspace = ws_parts[1]

        sub = self.agent.spawn_subagent(name, workspace=workspace)
        self.agent._subagents[name] = sub
        ws_note = f" (workspace={workspace})" if workspace else ""
        return f"Created SubAgent '{name}'{ws_note}."

    def _cmd_run(self, args: str) -> str:
        """Run a task in an existing subagent."""
        parts = args.strip().split(None, 1)
        if len(parts) < 2:
            return "Usage: /subagent run <name> <task>"

        name, task = parts[0], parts[1]
        sub = self.agent._subagents.get(name)
        if sub is None:
            return f"SubAgent '{name}' not found. Create it with '/subagent create {name}'."

        result = sub.respond(task)
        return f"[{name}] {result}"

    def _cmd_list(self, args: str) -> str:
        """List all active subagents."""
        subs = self.agent._subagents
        if not subs:
            return "No subagents created yet. Use '/subagent create <name>'."

        lines = ["Active subagents:"]
        for name, sub in subs.items():
            conv_len = len(sub.get_conversation())
            summary = f"{conv_len // 2} turns" if conv_len else "empty"
            lines.append(f"  - {name}: {summary}")
        return "\n".join(lines)

    def _cmd_summary(self, args: str) -> str:
        """Show condensed context for a subagent."""
        name = args.strip()
        if not name:
            return "Usage: /subagent summary <name>"

        sub = self.agent._subagents.get(name)
        if sub is None:
            return f"SubAgent '{name}' not found."

        return sub.get_context_summary(max_messages=5)

    def _cmd_reset(self, args: str) -> str:
        """Reset a subagent's conversation history."""
        name = args.strip()
        if not name:
            return "Usage: /subagent reset <name>"

        sub = self.agent._subagents.get(name)
        if sub is None:
            return f"SubAgent '{name}' not found."

        sub.reset()
        return f"Reset SubAgent '{name}'."

    def _help(self, args: str = "") -> str:
        """Show help for the subagent command."""
        return """\
Usage: /subagent <command> [args]

Commands:
  create <name> [--workspace <path>]   Create a new named subagent
  run <name> <task>                    Run a task in an existing subagent
  list                                  List all active subagents
  summary <name>                       Show condensed context for a subagent
  reset <name>                         Clear a subagent's conversation history
  help                                 Show this help message

Examples:
  /subagent create researcher           Create a 'researcher' subagent
  /subagent run researcher Find all .py files
  /subagent summary researcher          See what the researcher has done
"""


__all__ = ["SubAgentCommand"]
