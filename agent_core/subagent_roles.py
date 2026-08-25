"""Role registry for subagents (plan: nine issue-class roles).

Each :class:`SubagentRole` bundles everything the parent needs to spawn a
fit-for-purpose child agent:

* ``mode``        — session mode tag (``build`` / ``plan``) the role runs under;
* ``tools_allowed``— the exact NLP tools the role may use (validated at import
  time against :data:`agent_core.tool_schemas.NLP_TOOL_NAMES`, and against
  ``PLAN_MODE_TOOLS`` for read-only roles, so a role can never name a tool
  that does not exist or contradict its own read-only claim);
* ``system_prompt`` — persona injected as the first system message;
* ``max_turns``   — conversation-turn cap feeding the stuck-synthesis guard.

This module stays import-free of ``agent`` (agent_core namespace rule).
"""
from __future__ import annotations

from dataclasses import dataclass

from agent_core.modes import MODE_BUILD, MODE_PLAN, PLAN_MODE_TOOLS
from agent_core.tool_schemas import NLP_TOOL_NAMES


@dataclass(frozen=True)
class SubagentRole:
    """Immutable specification of one subagent role."""

    #: Lookup key used by ``/subagent create <name> --role <role>``.
    name: str
    #: Human-readable title shown in listings.
    title: str
    #: Session mode the role operates under (``build`` or ``plan``).
    mode: str
    #: Exact set of NLP tools this role may call.
    tools_allowed: frozenset[str]
    #: Persona injected as the leading system message.
    system_prompt: str
    #: Hard cap on conversation turns before the parent must collect.
    max_turns: int = 12
    #: True when the role must never mutate the workspace.  Enforced by the
    #: validation below: a read-only role's tools must be a subset of
    #: ``PLAN_MODE_TOOLS`` (the verified read-only toolset).
    read_only: bool = False


_PLANNER_TOOLS = frozenset(
    {"search", "read", "list_files", "definitions", "references"}
)
_REVIEWER_TOOLS = _PLANNER_TOOLS | {"diff"}

ROLES: dict[str, SubagentRole] = {
    r.name: r
    for r in (
        SubagentRole(
            name="planner",
            title="Planner/Triage",
            mode=MODE_PLAN,
            tools_allowed=_PLANNER_TOOLS,
            read_only=True,
            system_prompt=(
                "You are the Planner/Triage subagent. Decompose the given issue "
                "into concrete, file-by-file tasks. You are READ-ONLY: locate "
                "code with search/definitions/references, read only what you "
                "need, and end with an ordered task list naming exact files and "
                "symbols. Never propose vague work."
            ),
        ),
        SubagentRole(
            name="implementer",
            title="Implementer",
            mode=MODE_BUILD,
            tools_allowed=frozenset({
                "search", "read", "list_files", "definitions", "references",
                "write", "edit", "run",
            }),
            system_prompt=(
                "You are the Implementer subagent. Execute exactly ONE assigned "
                "task: make the minimal correct change with write/edit, verify "
                "the file parses, and report files changed plus anything you "
                "could not finish. Do not refactor beyond the task."
            ),
        ),
        SubagentRole(
            name="tester",
            title="Tester",
            mode=MODE_BUILD,
            tools_allowed=frozenset({
                "search", "read", "list_files", "write", "edit", "run", "tests",
            }),
            system_prompt=(
                "You are the Tester subagent. Own the red→green cycle: write "
                "pytest regression tests under tests/, run them, and report "
                "exact pass/fail counts with failure output. A bug fix is not "
                "done until a test that fails without it passes with it."
            ),
        ),
        SubagentRole(
            name="debugger",
            title="Debugger/Fixer",
            mode=MODE_BUILD,
            tools_allowed=frozenset({
                "search", "read", "list_files", "definitions", "references",
                "diff", "edit", "write", "run", "tests",
            }),
            system_prompt=(
                "You are the Debugger/Fixer subagent. You receive a failing "
                "traceback or wrong behaviour. Find the root cause (not the "
                "first symptom), apply the MINIMAL patch, add the regression "
                "test that proves it, and re-run the failing case."
            ),
        ),
        SubagentRole(
            name="reviewer",
            title="Reviewer",
            mode=MODE_PLAN,
            tools_allowed=_REVIEWER_TOOLS,
            read_only=True,
            system_prompt=(
                "You are the Reviewer subagent. Inspect the given diff/files "
                "BEFORE commit: convention violations, duplicate definitions, "
                "logic errors, missing tests. You are READ-ONLY — report "
                "findings as a numbered list ordered by severity; never edit."
            ),
        ),
        SubagentRole(
            name="integrator",
            title="Integrator",
            mode=MODE_BUILD,
            tools_allowed=frozenset({
                "search", "read", "list_files", "diff", "git", "run",
            }),
            system_prompt=(
                "You are the Integrator subagent. Handle git discipline: stage "
                "and commit coherent change sets with conventional-commit "
                "messages, resolve merge conflicts, keep changelog entries in "
                "sync. Never force-push or rewrite published history."
            ),
        ),
        SubagentRole(
            name="researcher",
            title="Researcher",
            mode=MODE_PLAN,
            tools_allowed=frozenset({"web_search", "read"}),
            read_only=True,
            system_prompt=(
                "You are the Researcher subagent. Answer questions about EXTERNAL "
                "libraries, APIs and current events using web_search. Cite the "
                "source URL for every fact. Never fabricate; say 'unverified' "
                "when results are inconclusive."
            ),
        ),
        SubagentRole(
            name="security",
            title="Security Auditor",
            mode=MODE_PLAN,
            tools_allowed=_REVIEWER_TOOLS,
            read_only=True,
            system_prompt=(
                "You are the Security Auditor subagent. Review the touched files "
                "for hardcoded secrets, unsafe deserialization, path traversal, "
                "injection and untrusted-input handling. READ-ONLY: produce a "
                "severity-ranked findings list with file:line references."
            ),
        ),
        SubagentRole(
            name="documenter",
            title="Documenter",
            mode=MODE_BUILD,
            tools_allowed=frozenset({
                "search", "read", "list_files", "write", "edit",
            }),
            system_prompt=(
                "You are the Documenter subagent. After behaviour changes, repair "
                "documentation drift: README, USAGE, Architecture and CHANGES "
                "entries. Quote real flags/paths from the code — verify every "
                "claim you write by reading the source first."
            ),
        ),
    )
}


def get_role(name: str) -> SubagentRole | None:
    """Look up a role by *name* (case-insensitive); ``None`` when unknown."""
    return ROLES.get((name or "").strip().lower())


def role_names() -> list[str]:
    """Sorted role keys (drives ``/subagent roles``)."""
    return sorted(ROLES)


def validate_roles() -> list[str]:
    """Return every consistency problem in the registry (empty list == valid).

    Checked here once instead of trusting each definition site:
    * every tool exists in ``NLP_TOOL_NAMES`` (schema ↔ role parity);
    * ``read_only`` roles only use verified read-only tools (``PLAN_MODE_TOOLS``);
    * ``mode`` matches the read-only claim (read-only ⇒ plan mode);
    * prompts are non-empty and turn caps are positive.
    """
    problems: list[str] = []
    for role in ROLES.values():
        unknown = role.tools_allowed - NLP_TOOL_NAMES
        if unknown:
            problems.append(
                f"{role.name}: tools not in NLP_TOOL_NAMES: {sorted(unknown)}"
            )
        if role.read_only:
            leaking = role.tools_allowed - PLAN_MODE_TOOLS
            if leaking:
                problems.append(
                    f"{role.name}: read-only role uses mutating tools: {sorted(leaking)}"
                )
            if role.mode != MODE_PLAN:
                problems.append(
                    f"{role.name}: read-only role must run in '{MODE_PLAN}' mode"
                )
        elif role.mode != MODE_BUILD:
            problems.append(
                f"{role.name}: mutating role must run in '{MODE_BUILD}' mode"
            )
        if not role.system_prompt.strip():
            problems.append(f"{role.name}: empty system prompt")
        if role.max_turns <= 0:
            problems.append(f"{role.name}: max_turns must be positive")
    return problems


__all__ = [
    "ROLES",
    "SubagentRole",
    "get_role",
    "role_names",
    "validate_roles",
]
