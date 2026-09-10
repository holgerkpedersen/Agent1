"""Subagent management command (REPL).

Registered in ``agent.py::_register_commands`` so ``subagent`` appears in
the banner listing and dispatches like every other command.  All handlers
are async where they touch the LLM — the previous version returned a bare
coroutine object from ``respond()`` (never awaited), which this rewrite
fixes.

Usage::

    subagent roles                          List available roles
    subagent create <name> [--role <role>] [--workspace <path>]
    subagent run <name> <task ...>          Run a task (awaits the LLM)
    subagent list                           List active subagents
    subagent summary <name>                 Condensed context of one subagent
    subagent reset <name>                   Clear a subagent's history

Roles come from :mod:`agent_core.subagent_roles`; a role gives the child a
persona, a tool whitelist and a turn cap (see that module).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Command
from agent_core.colors import cyan, green, magenta, blue, yellow

if TYPE_CHECKING:
    from agent import Agent


class SubAgentCommand(Command):
    """Create subagents to parallelize independent work."""

    # Storage for active subagents, keyed per agent instance.
    _registry_note = "per-agent: agent._subagents"

    def __init__(self) -> None:
        if not hasattr(self, "_agents_seen"):
            self._agents_seen: set[int] = set()

    @property
    def name(self) -> str:
        return "subagent"

    @property
    def help_text(self) -> str:
        return (
            "subagent roles|create|run|list|summary|reset - manage subagents\n"
            "  subagent roles                                 List available roles\n"
            "  subagent create <name> [--role <role>] [--workspace <path>]\n"
            "  subagent run <name> <task ...>                  Run a task in a subagent\n"
            "  subagent list / summary <name> / reset <name>\n"
            "  subagent route <issue text>                     Suggest a role for an issue\n"
            "  subagent create fix1 --role debugger            Create with an explicit role\n"
            "  subagent create w1 --role auto --workspace path Assign role automatically"
        )

    async def execute(self, args: list[str], agent: "Agent") -> bool:
        if not hasattr(agent, "_subagents"):
            agent._subagents: dict[str, object] = {}

        parts = [a for a in args if a.strip()]
        if not parts:
            print(self.help_text)
            return True

        cmd = parts[0].lower()
        rest = parts[1:]
        handlers = {
            "roles": self._cmd_roles,
            "create": self._cmd_create,
            "run": self._cmd_run,
            "list": self._cmd_list,
            "summary": self._cmd_summary,
            "reset": self._cmd_reset,
            "route": self._cmd_route,
        }
        handler = handlers.get(cmd)
        if handler is None:
            print(f"Unknown subagent command '{cyan(cmd)}'. Try 'subagent' alone for help.")
            return True
        await handler(rest, agent)
        return True

    # ------------------------------------------------------------------
    async def _cmd_roles(self, args: list[str], agent: "Agent") -> None:
        from agent_core.subagent_roles import ROLES, validate_roles

        problems = validate_roles()
        if problems:
            print("Role registry problems (bug):")
            for p in problems:
                print(f"  ! {p}")
            return
        print(cyan("Available subagent roles:"))
        for role in ROLES.values():
            ro = " [read-only]" if role.read_only else ""
            print(f"  {cyan(role.name):<12} {role.title}{ro} — tools: "
                  f"{', '.join(sorted(role.tools_allowed))}")

    async def _cmd_create(self, args: list[str], agent: "Agent") -> None:
        from agent_core.subagent_roles import get_role, role_names

        if not args:
            print(f"Usage: subagent create {cyan('<name>')} [--role <role>|--role auto] [--workspace <path>]")
            return

        name = args[0]
        role: str | None = None
        workspace: str | None = None

        i = 1
        while i < len(args):
            if args[i] == "--role" and i + 1 < len(args):
                role = args[i + 1]
                i += 2
            elif args[i] == "--workspace" and i + 1 < len(args):
                workspace = args[i + 1]
                i += 2
            else:
                print(f"Unrecognized argument: {args[i]}")
                return

        if role is None:
            print("Tip: pass --role (see 'subagent roles', or --role auto).")
        elif role.lower() == "auto":
            # Auto-routing needs the task; without one, triage is the safe default.
            role = "planner"
            print(f"No task given for auto-routing — using '{cyan('planner')}' (triage).")
        if role is not None and get_role(role) is None:
            print(f"Unknown role '{role}'. Available: {', '.join(role_names())}")
            return

        sub = agent.spawn_subagent(name, workspace=workspace, role=role)
        agent._subagents[name] = sub
        ws_note = f", workspace={workspace}" if workspace else ""
        role_note = f", role={sub.role_name} ({sub.mode} mode)" if sub.role_name else ""
        print(green(f"Created SubAgent '{cyan(name)}'{ws_note}{role_note}."))
        if sub.role_name:
            print(f"  Tools: {', '.join(sorted(sub._tools_allowed))}")

    async def _cmd_route(self, args: list[str], agent: "Agent") -> None:
        from agent_core.issue_router import route_with_score, routing_table

        issue = " ".join(args).strip()
        if not issue:
            print(routing_table())
            return
        role_name, score = route_with_score(issue)
        from agent_core.subagent_roles import get_role
        spec = get_role(role_name)
        title = spec.title if spec else role_name
        print(f"Suggested role: {cyan(role_name)} ({title}) — score {score}")
        if score == 0:
            print("  (no keyword hits — ambiguous issue; planner/triage is the safe default)")
        print(f"  next: subagent create w1 --role {cyan(role_name)}")

    async def _cmd_run(self, args: list[str], agent: "Agent") -> None:
        if len(args) < 2:
            print("Usage: subagent run <name> <task ...>")
            return
        name, task = args[0], " ".join(args[1:])
        sub = agent._subagents.get(name)
        if sub is None:
            print(f"SubAgent '{cyan(name)}' not found. Create it with 'subagent create {name}'.")
            return
        print(magenta(f"[{name}] working ..."))
        # THE FIX: actually await the coroutine (was returned un-awaited).
        result = await sub.respond(task)
        print(green(f"[{name}] {result}"))

    async def _cmd_list(self, args: list[str], agent: "Agent") -> None:
        subs = getattr(agent, "_subagents", {})
        if not subs:
            print("No subagents created yet. Use 'subagent create <name>'.")
            return
        print(cyan("Active subagents:"))
        for sname, sub in subs.items():
            conv_len = len(sub.get_conversation())
            summary = f"{conv_len // 2} turns" if conv_len else "empty"
            role_part = f", role={sub.role_name}" if sub.role_name else ""
            print(f"  - {cyan(sname)}{role_part}: {summary}")

    async def _cmd_summary(self, args: list[str], agent: "Agent") -> None:
        if not args:
            print("Usage: subagent summary <name>")
            return
        sub = agent._subagents.get(args[0])
        if sub is None:
            print(f"SubAgent '{args[0]}' not found.")
            return
        raw_lines = sub.get_context_summary(max_messages=5).splitlines()
        for i, line in enumerate(raw_lines):
            if i == 0:
                # Header line — color it cyan
                print(cyan(line))
            elif line.startswith("  > "):
                # User message prefix — green
                print(green(line))
            elif line.startswith("  < "):
                # Assistant response prefix — blue
                print(blue(line))
            else:
                # Fallback
                print(line)

    async def _cmd_reset(self, args: list[str], agent: "Agent") -> None:
        if not args:
            print("Usage: subagent reset <name>")
            return
        name = args[0]
        sub = agent._subagents.get(name)
        if sub is None:
            print(f"SubAgent '{cyan(name)}' not found.")
            return
        sub.reset()
        print(green(f"Reset SubAgent '{cyan(name)}'."))


__all__ = ["SubAgentCommand"]
