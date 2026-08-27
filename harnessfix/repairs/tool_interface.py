"""Repair: tool interface - richer tool error messages fed back to the model.

Trace-grounded: `tool_error` events record the exception type, but the model
only ever saw "Tool error: <exc>".  Including the exception class name in the
fed-back message gives the model diagnostic signal for its next action.

Target: agent_core/llm/tool_loop.py (the fed-back tool error string).
"""
from __future__ import annotations

import py_compile
from pathlib import Path

TOOL_INTERFACE_REPAIR_ID = "tool-interface-error-detail"

#: Runtime string fragments this repair alters — the collision guard scans
#: the test suite for these BEFORE applying (assertions pin runtime output).
COLLISION_FRAGMENTS: tuple[str, ...] = ("Tool error: ",)

_TARGET = Path("agent_core/llm/tool_loop.py")
_OLD = 'result_str = f"Tool error: {exc}"'
_NEW = 'result_str = f"Tool error ({type(exc).__name__}): {exc}"'

#: The single source file this repair transforms (used to scope the
#: autonomous driver's commit to the repair's own files only).
FILES: tuple[str, ...] = (str(_TARGET),)


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
    if _NEW in source:
        return "already applied (no-op)"
    if _OLD not in source:
        raise RepairApplyError(f"pattern not found in {_TARGET}")
    _TARGET.write_text(source.replace(_OLD, _NEW, 1), encoding="utf-8")
    _verify()
    return f"{_TARGET}: {_OLD} -> {_NEW}"


def revert() -> None:
    """Revert the repair (restore the original error message)."""
    source = _TARGET.read_text(encoding="utf-8")
    if _OLD in source:
        return  # nothing to revert
    if _NEW not in source:
        raise RepairApplyError(f"repaired pattern not found in {_TARGET}")
    _TARGET.write_text(source.replace(_NEW, _OLD, 1), encoding="utf-8")
    _verify()


def is_applied() -> bool:
    """Pure probe: True if the repair is already present in the tree.

    Does NOT apply the repair (unlike ``apply()``); the loop uses this to
    skip an already-applied candidate without mutating the tree.
    """
    return _NEW in _TARGET.read_text(encoding="utf-8")
