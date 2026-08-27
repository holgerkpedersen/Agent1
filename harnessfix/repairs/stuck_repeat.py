"""Repair: stuck-repeat prevention - concrete alternatives on the SECOND
consecutive identical tool call, while the model still has budget.

Trace-grounded: traces ending in the stuck guard show three identical calls
(the third strike stops the loop and forces synthesis).  The second strike
already tells the model the call was not re-executed, but gives no concrete
alternative - the very next response repeats again.  Appending a short,
tool-specific hint table at strike two gives the model something actionable
BEFORE the fatal third repeat.

Target: agent_core/llm/tool_loop.py (second-strike steering note + hint
table constants inserted above _PATH_MISS_PREFIXES).

Collision surface: the OLD second-strike suffix "Take a different action or
answer in text." - any test pinning it would break, so the guard scans for
it before applying.  The PINNED prefix "NOTE: This exact call has now been
executed" (tests/test_tool_loop_nlp.py) is preserved byte-for-byte.
"""
from __future__ import annotations

import py_compile
from pathlib import Path

STUCK_REPEAT_REPAIR_ID = "stuck-repeat-tool-hints"

#: Runtime string fragments this repair alters - the collision guard scans
#: the test suite for these BEFORE applying (assertions pin runtime output).
COLLISION_FRAGMENTS: tuple[str, ...] = (
    "Take a different action or answer in text.",
)

_TARGET = Path("agent_core/llm/tool_loop.py")

#: The single source file this repair transforms (used to scope the
#: autonomous driver's commit to the repair's own files only).
FILES: tuple[str, ...] = (str(_TARGET),)

#: Anchor line the hint-table block is inserted ABOVE (unique in the file).
_ANCHOR = '_PATH_MISS_PREFIXES = ("File not found:", "Error reading file:")'

#: A distinctive line from the hint block, used as an apply/revert state
#: sentinel so a half-applied tree fails loudly instead of silently.
_HINT_SENTINEL = "_REPEAT_HINTS: dict[str, str] = {"
_HINT_BLOCK = '''#: Steering hints appended to the SECOND-consecutive-identical-call note
#: (repair stuck-repeat-tool-hints): the model still has budget at strike two,
#: so a concrete alternative beats waiting for the third strike to stop the
#: loop.  Keys MUST match the LLM tool-calling names in
#: agent_core/tool_schemas.NLP_TOOL_NAMES so no real tool silently falls
#: through to the default.  The coverage test asserts every tool name has a
#: dedicated entry (or shares a capability-class fallback), so a future
#: schema rename cannot silently drop a tool's hint.
_REPEAT_HINTS: dict[str, str] = {
    # Read-like: you already have the contents — stop re-reading.
    "read": (
        "Alternatives: use definitions() on the file for a symbol map, search "
        "for what you actually need, or answer from what you already read."
    ),
    # Search-like: vary the query/path or act on what you have.
    "search": (
        "Alternatives: try a different symbol or path, list_files() to see what "
        "exists, or answer from the matches you already have."
    ),
    "references": (
        "Alternatives: read the file at a known location, search() for the "
        "symbol in context, or answer from the usages you already have."
    ),
    "definitions": (
        "Alternatives: read the file you just indexed, search() for a symbol "
        "inside it, or answer from the signatures you already have."
    ),
    "web_search": (
        "Alternatives: rephrase the query, open one of the results you already "
        "have, or answer from the snippets returned."
    ),
    # List-like: you already have the directory — proceed or narrow.
    "list_files": (
        "Alternatives: search() instead of listing again, or proceed with the "
        "entries you already have."
    ),
    # Mutating-file: a repeated write/edit usually means the prior change did
    # not land as expected — inspect rather than re-apply blindly.
    "write": (
        "Alternatives: read() the file to confirm its current state, edit() the "
        "existing content instead of overwriting, or report what you already wrote."
    ),
    "edit": (
        "Alternatives: read() the file to see why the edit did not apply, use a "
        "more distinctive old_text, or report the change you already made."
    ),
    # Command/shell-like: vary the command or inspect its result.
    "run": (
        "Alternatives: change the command's arguments, run the tests to verify "
        "state, or report the result you already have."
    ),
    "git": (
        "Alternatives: run a different subcommand (status/diff/log), inspect the "
        "output you already have, or report the current state."
    ),
    "tests": (
        "Alternatives: narrow the path/framework, read() a failing test, or "
        "report the failures you already have."
    ),
    "fix": (
        "Alternatives: target a specific file instead of the batch, read() the "
        "error yourself, or report the issue you already identified."
    ),
    "diff": (
        "Alternatives: read() the file directly, compare a different file, or "
        "answer from the diff you already have."
    ),
    # Analysis/delegation: a repeat means the answer did not satisfy — ask
    # differently or act on what came back.
    "analyze": (
        "Alternatives: ask a more specific question, read() the file yourself, "
        "or act on the summary you already have."
    ),
    "delegate": (
        "Alternatives: give the subagent a sharper task, delegate a different "
        "role, or act on the answer you already received."
    ),
    "delegate_batch": (
        "Alternatives: narrow the roles, sharpen the shared task, or act on the "
        "merged reports you already have."
    ),
    # MCP: the server/tool combo already failed or repeated — re-check args.
    "mcp_tools": (
        "Alternatives: pick a different server, read the discovered tool schema, "
        "or act on the catalog you already listed."
    ),
    "mcp_call": (
        "Alternatives: re-check the server/tool/arguments, call a different MCP "
        "tool, or act on the result you already have."
    ),
}
_REPEAT_HINT_DEFAULT = (
    "Alternatives: take a different action with what you already know, or "
    "give your final answer now."
)

'''

#: Second-strike suffix: original (reverted) vs hint-routed (applied).
_OLD_LINE = '"identical. Take a different action or answer in text."'
_NEW_LINE = '"identical. " + _REPEAT_HINTS.get(tool_name, _REPEAT_HINT_DEFAULT)'


class RepairApplyError(RuntimeError):
    """Raised when a repair cannot be applied or reverted cleanly."""


def _verify() -> None:
    try:
        py_compile.compile(str(_TARGET), doraise=True)
    except py_compile.PyCompileError as exc:
        raise RepairApplyError(f"repair broke {_TARGET}: {exc}") from exc


def apply() -> str:
    """Apply the repair; returns a short diff summary."""
    source = _TARGET.read_text(encoding="utf-8")
    if _NEW_LINE in source:
        return "already applied (no-op)"
    if _ANCHOR not in source:
        raise RepairApplyError(f"anchor not found in {_TARGET}: {_ANCHOR}")
    if _OLD_LINE not in source:
        raise RepairApplyError(f"pattern not found in {_TARGET}")
    if _HINT_SENTINEL in source:
        raise RepairApplyError(
            f"inconsistent state in {_TARGET}: hint table present but note "
            "not routed through it"
        )
    source = source.replace(_ANCHOR, _HINT_BLOCK + _ANCHOR, 1)
    source = source.replace(_OLD_LINE, _NEW_LINE, 1)
    _TARGET.write_text(source, encoding="utf-8")
    _verify()
    return (
        f"{_TARGET}: second-strike note now carries _REPEAT_HINTS[tool] "
        "(stuck-repeat prevention at strike two)"
    )


def revert() -> None:
    """Revert the repair (restore the original second-strike note)."""
    source = _TARGET.read_text(encoding="utf-8")
    if _OLD_LINE in source:
        if _HINT_SENTINEL not in source:
            return  # nothing to revert
        raise RepairApplyError(
            f"inconsistent state in {_TARGET}: hint table present although "
            "the note is reverted"
        )
    if _NEW_LINE not in source:
        raise RepairApplyError(f"repaired pattern not found in {_TARGET}")
    source = source.replace(_HINT_BLOCK + _ANCHOR, _ANCHOR, 1)
    source = source.replace(_NEW_LINE, _OLD_LINE, 1)
    _TARGET.write_text(source, encoding="utf-8")
    _verify()


def is_applied() -> bool:
    """Pure probe: True if the repair is already present in the tree.

    Does NOT apply the repair (unlike ``apply()``); the loop uses this to
    skip an already-applied candidate without mutating the tree.
    """
    source = _TARGET.read_text(encoding="utf-8")
    return _NEW_LINE in source and _HINT_SENTINEL in source
