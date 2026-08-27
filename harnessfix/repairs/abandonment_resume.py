"""Repair: abandonment-after-mutation resume protocol.

Trace-grounded: 8 of 263 traces in the 2026-08-25 corpus mutated files
(write/edit/fix, recorded in tool_result.affected_files) and then ended
*without* a loop_end event — i.e. the run was interrupted (crash / killed /
provider loss, decision #052) after the workspace was already changed.  The
model only sees its own text and has no memory of what it touched, so the
next turn starts cold and often repeats or contradicts the half-applied work.

The repair injects a RECONNECT note when the loop ends non-completed after at
least one mutation: it names the files touched this run (from the trace sink's
affected_files) and tells the model to resume the task rather than restart.
The note travels as a "user" message (strict chat templates reject mid-
conversation system roles) and is stripped from the persisted history like the
other steering notes, so a fresh session never sees stale steering.

Target: agent_core/llm/tool_loop.py (mutation tracker + reconnect note in the
synthesis block, guarded by the new GUARD_RESUME label).

Collision surface: the OLD no-progress force string "Give your final answer
now" is preserved byte-for-byte — only the reconnect branch adds a NEW note,
so assertions pinning the old string are unaffected.  The repair's own
runtime string is registered as a collision fragment so the guard can skip it
if any test pins it.
"""
from __future__ import annotations

import py_compile
from pathlib import Path

ABANDONMENT_RESUME_REPAIR_ID = "abandonment-resume-protocol"

#: Runtime string fragments this repair alters — the collision guard scans
#: the test suite for these BEFORE applying (assertions pin runtime output).
COLLISION_FRAGMENTS: tuple[str, ...] = (
    "RECONNECT: this run was interrupted after mutating",
)

_TARGET = Path("agent_core/llm/tool_loop.py")

#: The single source file this repair transforms (used to scope the
#: autonomous driver's commit to the repair's own files only).
FILES: tuple[str, ...] = (str(_TARGET),)

#: Anchor 1: the mutation-tool set, above which the reconnect note constant
#: and the mutation-tracking accumulator are inserted.
_ANCHOR_MUT = "MUTATING_TOOLS = frozenset({\"write\", \"edit\", \"fix\"})"
_HINT_SENTINEL = '_RESUME_NOTE = ('
_INSERT_BLOCK = '''#: Injected when a run ends non-completed AFTER mutating the workspace
#: (repair abandonment-resume-protocol): the model has no memory of what it
#: touched, so without this it starts the next turn cold and repeats or
#: contradicts the half-applied work.  Travels as a "user" message (strict
#: chat templates reject mid-conversation system roles) and is stripped from
#: the persisted history like the other steering notes.
_RESUME_NOTE = (
    "RECONNECT: this run was interrupted after mutating the following file(s): "
    "{files}. Resume the task from there — do NOT re-apply changes that are "
    "already written, and do NOT start over. Continue the remaining work, then "
    "give your final answer."
)
#: Trace guard label for the abandonment-resume note (decision #052). Defined
#: here because harnessfix.tracing does not expose a dedicated enum value.
GUARD_RESUME = "abandonment_resume"

'''

#: Anchor 2: the synthesis block's final "else: termination_reason = answer"
#: line.  The reconnect note is appended to the SAME current_messages the
#: forced-synthesis call uses, but only when files were mutated and the run
#: did not complete — so it reaches the model's final answer, not a fresh turn.
_ANCHOR_END = '            self.termination_reason = "answer"'
_OLD_END = '            self.termination_reason = "answer"'
_NEW_END = (
    '            self.termination_reason = "answer"\n'
    '        #: Abandonment-after-mutation resume protocol (repair\n'
    '        #: abandonment-resume-protocol): when the run was interrupted\n'
    '        #: after mutating the workspace, the model has no memory of what\n'
    '        #: it touched.  Name the mutated files so the next turn RESUMES\n'
    '        #: instead of restarting (decision #052; corpus: 8/263 traces).\n'
    '        if self._mutated_files and self.termination_reason != "answer":\n'
    '            note = _RESUME_NOTE.format(\n'
    '                files=", ".join(sorted(self._mutated_files)),\n'
    '            )\n'
    '            current_messages.append({"role": "user", "content": note})\n'
    '            injected_notes.append(note)\n'
    '            self._emit(\n'
    '                KIND_GUARD_TRIGGERED,\n'
    '                LAYER_LIFECYCLE,\n'
    '                guard=GUARD_RESUME,\n'
    '                iteration=iteration,\n'
    '                note=note,\n'
    '                affected_files=sorted(self._mutated_files),\n'
    '            )'
)


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
    if _NEW_END in source:
        return "already applied (no-op)"
    if _ANCHOR_MUT not in source:
        raise RepairApplyError(f"anchor not found in {_TARGET}: {_ANCHOR_MUT}")
    if _ANCHOR_END not in source:
        raise RepairApplyError(f"anchor not found in {_TARGET}: {_ANCHOR_END}")
    if _HINT_SENTINEL in source:
        raise RepairApplyError(
            f"inconsistent state in {_TARGET}: resume note present but "
            "synthesis block not routed through it"
        )
    source = source.replace(_ANCHOR_MUT, _ANCHOR_MUT + "\n" + _INSERT_BLOCK, 1)
    source = source.replace(_OLD_END, _NEW_END, 1)
    _TARGET.write_text(source, encoding="utf-8")
    _verify()
    return (
        f"{_TARGET}: reconnect note emitted after mutation+non-completion "
        "(abandonment-resume protocol)"
    )


def revert() -> None:
    """Revert the repair (restore the original synthesis block)."""
    source = _TARGET.read_text(encoding="utf-8")
    if _NEW_END not in source:
        # Not applied: the synthesis block is still the original form.
        if _HINT_SENTINEL in source:
            raise RepairApplyError(
                f"inconsistent state in {_TARGET}: resume note present "
                "although the synthesis block is not routed through it"
            )
        return  # nothing to revert
    if _HINT_SENTINEL not in source:
        raise RepairApplyError(
            f"inconsistent state in {_TARGET}: synthesis block routed but "
            "the resume note constant is missing"
        )
    source = source.replace(_NEW_END, _OLD_END, 1)
    source = source.replace(_ANCHOR_MUT + "\n" + _INSERT_BLOCK, _ANCHOR_MUT, 1)
    _TARGET.write_text(source, encoding="utf-8")
    _verify()


def is_applied() -> bool:
    """Pure probe: True if the repair is already present in the tree.

    Does NOT apply the repair (unlike ``apply()``); the loop uses this to
    skip an already-applied candidate without mutating the tree.
    """
    source = _TARGET.read_text(encoding="utf-8")
    return _NEW_END in source and _HINT_SENTINEL in source
