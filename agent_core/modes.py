"""Session modes for the conversational agent (opencode-style).

Two built-in modes mirror opencode's design:

* ``build`` (default) — the full toolset; the agent may write, edit, run
  and fix code.
* ``plan`` — READ-ONLY research mode: the NLP tool loop only offers the
  read-only tool schemas, and the executor rejects every other tool call
  outright, so no file in the workspace can change during a plan-mode
  turn.  The model is steered to end the turn with a concrete plan as its
  text answer instead (mirroring opencode's plan-mode system prompt).

Enforcement is two-layered on purpose:

1. :func:`filter_tool_schemas` removes mutating tools from the schema list
   sent to the LLM, so a well-behaved model never attempts a write.
2. :func:`check_tool_allowed` is enforced in ``Agent._execute_tool_call``
   — the single choke point every tool-call consumer goes through
   (``chat_nlp``, ``multillm``, ...) — so even a hallucinated or
   schema-crafted write attempt is rejected at runtime.

The mode is session state on the ``Agent`` instance (``agent.mode``); it
does not persist across restarts, mirroring opencode's per-session modes.
This module stays import-free of ``agent`` (agent_core namespace rule).
"""
from __future__ import annotations

from typing import Any

MODE_BUILD = "build"
MODE_PLAN = "plan"

#: Tools permitted in plan mode — verified read-only at handler level:
#: ``search``/``read``/``list_files`` only inspect the filesystem, ``diff``
#: shells out to read-only ``git diff``, ``web_search`` queries DuckDuckGo,
#: and ``definitions``/``references`` are pure-AST readers (they never
#: write; ``definitions`` opens one file read-only, ``references`` walks the
#: workspace read-only).  Everything else can create or modify files —
#: ``write``, ``edit``, ``fix`` obviously, but also ``run`` (arbitrary
#: shell), ``git`` (add/commit/checkout …), ``tests`` (pytest writes
#: ``__pycache__``/``.pytest_cache``) and ``analyze`` (writes its report
#: into ``.docs/<ts>/``) — so plan mode excludes them wholesale.
PLAN_MODE_TOOLS: frozenset[str] = frozenset({
    "search", "read", "list_files", "diff", "web_search",
    "definitions", "references",
})

#: Runtime rejection returned by the executor for a blocked tool call.
_PLAN_REJECTION = (
    "[plan mode] '{name}' is blocked: Plan mode is read-only — no file in "
    "the workspace may be changed. Present your plan as a text answer "
    "instead; the user can switch back with 'mode build' to apply it."
)


def is_plan_mode(mode: str) -> bool:
    """True exactly for the plan mode tag (anything else behaves as build)."""
    return mode == MODE_PLAN


def filter_tool_schemas(
    schemas: list[dict[str, Any]], mode: str,
) -> list[dict[str, Any]]:
    """Return the tool schemas the LLM may see in *mode*.

    ``build`` (and any unknown tag, which fails safe to build behaviour at
    the command layer) gets the full list; ``plan`` gets only the
    read-only subset, so the model is never even offered a mutating tool.
    """
    if not is_plan_mode(mode):
        return list(schemas)
    return [
        s for s in schemas
        if s.get("function", {}).get("name") in PLAN_MODE_TOOLS
    ]


def check_tool_allowed(name: str, mode: str) -> str | None:
    """Rejection message when tool *name* must not run in *mode*, else None.

    Enforced in ``Agent._execute_tool_call`` — the choke point shared by
    every agentic loop — so plan mode holds even when a schema was not
    filtered (e.g. ``multillm`` advertising the full set).
    """
    lowered = name.lower()
    if is_plan_mode(mode) and lowered not in PLAN_MODE_TOOLS:
        return _PLAN_REJECTION.format(name=lowered)
    return None


def plan_mode_system_suffix() -> str:
    """Appended to the system prompt when a session STARTS in plan mode."""
    return (
        "\n\nSESSION MODE: PLAN (read-only).\n"
        "- Your toolset is limited to search, read, list_files, diff and "
        "web_search; any write/edit/run/git/tests/fix call is REJECTED.\n"
        "- Do not attempt or promise changes: research the workspace, then "
        "end the turn with a concrete, file-by-file implementation plan as "
        "your final text answer.\n"
        "- The user applies the plan later by switching back with "
        "'mode build'."
    )


def plan_mode_turn_note() -> str:
    """Prepended to each user turn while plan mode is active.

    A per-turn note (not a mid-conversation system message) because strict
    chat templates (qwen Jinja) reject system messages after the first
    entry — the same constraint that makes the continuation note a user
    message in ``chat_nlp``.
    """
    return (
        "[PLAN MODE] Read-only research turn: propose a plan, change "
        "nothing. Mutating tools are blocked; finish with the plan as "
        "text."
    )
