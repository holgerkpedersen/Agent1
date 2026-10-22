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
import re
from pathlib import Path

STUCK_REPEAT_REPAIR_ID = "stuck-repeat-tool-hints"

#: Runtime string fragments this repair alters - the collision guard scans
#: the test suite for these BEFORE applying (assertions pin runtime output).
COLLISION_FRAGMENTS: tuple[str, ...] = ("Take a different action or answer in text.",)

_TARGET = Path("agent_core/llm/tool_loop.py")

#: The single source file this repair transforms (used to scope the
#: autonomous driver's commit to the repair's own files only).
FILES: tuple[str, ...] = (str(_TARGET),)

#: Anchor line the hint-table block is inserted ABOVE (unique in the file).
_ANCHOR = '_PATH_MISS_PREFIXES = ("File not found:", "Error reading file:")'

#: A distinctive line from the hint block, used as an apply/revert state
#: sentinel so a half-applied tree fails loudly instead of silently.
_HINT_SENTINEL = "_REPEAT_HINTS: dict[str, str] = {"
_HINT_BLOCK = """#: \
Steering hints appended to the SECOND-consecutive-identical-call note
#: (repair stuck-repeat-tool-hints): the model still has budget at strike two,
#: so a concrete alternative beats waiting for the third strike to stop the
#: loop.  Keys are tool names; anything unmapped falls back to the default.
_REPEAT_HINTS: dict[str, str] = {
    "read": (
        "Alternatives: use definitions() on the file for a symbol map, search "
        "for what you actually need, or answer from what you already read."
    ),
    "search": (
        "Alternatives: try a different symbol or path, list_files() to see what "
        "exists, or answer from the matches you already have."
    ),
    "list_files": (
        "Alternatives: search() instead of listing again, or proceed with the "
        "entries you already have."
    ),
    "run": (
        "Alternatives: run the tests to verify state, change the command's "
        "arguments, or report the result you already have."
    ),
}
_REPEAT_HINT_DEFAULT = (
    "Alternatives: take a different action with what you already know, or "
    "give your final answer now."
)

"""

#: Second-strike suffix: original (reverted) vs hint-routed (applied).
#:
#: The two forms are matched with REGEX, not with a literal ``in`` test: a
#: formatter (ruff) reflows the concatenation onto a second line, so the
#: byte-exact spelling of the applied form depends on the formatter's line
#: width, not on this module.  Matching the two halves separately keeps
#: apply/revert working on the single-line AND the wrapped layout.
_OLD_LINE = '"identical. Take a different action or answer in text."'
_NEW_LINE = '"identical. " + _REPEAT_HINTS.get(tool_name, _REPEAT_HINT_DEFAULT)'

#: Regex form of the reverted suffix (see above).
_OLD_RE = re.compile(r'"identical\. Take a different action or answer in text\."')

#: Regex form of the applied suffix: the literal and the hint lookup, with
#: any amount of whitespace between them - including the newline a formatter
#: inserts when the joined line does not fit the line length.
_NEW_RE = re.compile(
    r'"identical\. "[ \t\r\n]*\+[ \t\r\n]*'
    r"_REPEAT_HINTS\.get\([ \t\r\n]*tool_name[ \t\r\n]*,[ \t\r\n]*"
    r"_REPEAT_HINT_DEFAULT[ \t\r\n]*\)"
)


def _indent_of(source: str, pos: int) -> str:
    """Leading whitespace of the source line that contains *pos*."""
    line_start = source.rfind("\n", 0, pos) + 1
    return source[line_start:pos]


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
    if _NEW_RE.search(source):
        return "already applied (no-op)"
    if _ANCHOR not in source:
        raise RepairApplyError(f"anchor not found in {_TARGET}: {_ANCHOR}")
    match = _OLD_RE.search(source)
    if match is None:
        raise RepairApplyError(f"pattern not found in {_TARGET}")
    if _HINT_SENTINEL in source:
        raise RepairApplyError(
            f"inconsistent state in {_TARGET}: hint table present but note "
            "not routed through it"
        )
    indent = _indent_of(source, match.start())
    # The applied form keeps the lookup on its own line: that is how the
    # formatter lays it out, so the repaired tree stays formatter-clean.
    # The match starts AT the literal, so the line's own indentation is
    # already in ``source`` - only the NEW second line needs one.
    source = (
        source[: match.start()]
        + '"identical. "\n'
        + f"{indent}+ _REPEAT_HINTS.get(tool_name, _REPEAT_HINT_DEFAULT)"
        + source[match.end() :]
    )
    source = source.replace(_ANCHOR, _HINT_BLOCK + _ANCHOR, 1)
    _TARGET.write_text(source, encoding="utf-8")
    _verify()
    return (
        f"{_TARGET}: second-strike note now carries _REPEAT_HINTS[tool] "
        "(stuck-repeat prevention at strike two)"
    )


def revert() -> None:
    """Revert the repair (restore the original second-strike note)."""
    source = _TARGET.read_text(encoding="utf-8")
    if _OLD_RE.search(source):
        if _HINT_SENTINEL not in source:
            return  # nothing to revert
        raise RepairApplyError(
            f"inconsistent state in {_TARGET}: hint table present although "
            "the note is reverted"
        )
    match = _NEW_RE.search(source)
    if match is None:
        raise RepairApplyError(f"repaired pattern not found in {_TARGET}")
    # The match starts AT the literal, so the line's own indentation is
    # already in ``source`` - the replacement must not add another one.
    source = (
        source[: match.start()]
        + '"identical. Take a different action or answer in text."'
        + source[match.end() :]
    )
    source = source.replace(_HINT_BLOCK + _ANCHOR, _ANCHOR, 1)
    _TARGET.write_text(source, encoding="utf-8")
    _verify()


def is_applied() -> bool:
    """Pure probe: True if the repair is already present in the tree.

    Does NOT apply the repair (unlike ``apply()``); the loop uses this to
    skip an already-applied candidate without mutating the tree.
    """
    source = _TARGET.read_text(encoding="utf-8")
    return _NEW_RE.search(source) is not None and _HINT_SENTINEL in source
