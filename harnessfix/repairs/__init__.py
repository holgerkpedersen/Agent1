"""Phase 3 - scoped repair catalog (spec section 3.4).

One module per harness layer; each repair is a small, revertible, code-level
change backed by trace evidence.  Repairs are applied via exact source
transforms and verified with py_compile; the loop reverts any repair that
fails the test/security gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .stuck_repeat import STUCK_REPEAT_REPAIR_ID
from .stuck_repeat import apply as _apply_stuck_repeat
from .stuck_repeat import revert as _revert_stuck_repeat
from .stuck_repeat import COLLISION_FRAGMENTS as _STUCK_REPEAT_COLLISIONS
from .tool_interface import apply as _apply_tool_interface
from .tool_interface import revert as _revert_tool_interface
from .tool_interface import TOOL_INTERFACE_REPAIR_ID
from .tool_interface import COLLISION_FRAGMENTS
from .abandonment_resume import apply as _apply_abandonment_resume
from .abandonment_resume import revert as _revert_abandonment_resume
from .abandonment_resume import ABANDONMENT_RESUME_REPAIR_ID
from .abandonment_resume import COLLISION_FRAGMENTS as _ABANDONMENT_RESUME_COLLISIONS

#: Harness layers targeted by catalog repairs (must be valid HarnessFix facets).
TOOL_INTERFACE_LAYER = "tool_interface"
LIFECYCLE_LAYER = "lifecycle"


@dataclass(frozen=True)
class Repair:
    """A single scoped harness repair: apply()/revert() code transforms.

    ``collision_fragments`` lists the RUNTIME string fragments the repair
    rewrites; the loop scans the test suite for them before applying and
    skips the repair on any hit (see repairs.collisions).
    """

    id: str
    layer: str
    description: str
    apply: Callable[[], str]
    revert: Callable[[], None]
    collision_fragments: tuple[str, ...] = ()

    def applied_summary(self) -> str:
        return self.apply()


#: Catalog of concrete repairs, keyed by repair id.
CATALOG: dict[str, Repair] = {
    TOOL_INTERFACE_REPAIR_ID: Repair(
        id=TOOL_INTERFACE_REPAIR_ID,
        layer=TOOL_INTERFACE_LAYER,
        description=(
            "tool interface: include the exception type in tool error messages "
            "fed back to the model, so a failing tool call carries diagnostic "
            "signal instead of a bare message."
        ),
        apply=_apply_tool_interface,
        revert=_revert_tool_interface,
        collision_fragments=COLLISION_FRAGMENTS,
    ),
    STUCK_REPEAT_REPAIR_ID: Repair(
        id=STUCK_REPEAT_REPAIR_ID,
        layer=LIFECYCLE_LAYER,
        description=(
            "stuck-repeat prevention: the second consecutive identical tool "
            "call gets concrete alternatives (per-tool hint table) while the "
            "model still has budget, instead of only being stopped at strike "
            "three."
        ),
        apply=_apply_stuck_repeat,
        revert=_revert_stuck_repeat,
        collision_fragments=_STUCK_REPEAT_COLLISIONS,
    ),
    ABANDONMENT_RESUME_REPAIR_ID: Repair(
        id=ABANDONMENT_RESUME_REPAIR_ID,
        layer=LIFECYCLE_LAYER,
        description=(
            "abandonment-after-mutation resume protocol: when a run ends "
            "non-completed after mutating files (crash/kill/provider loss, "
            "decision #052), the next turn is told which files were touched so "
            "it resumes the task instead of restarting from scratch."
        ),
        apply=_apply_abandonment_resume,
        revert=_revert_abandonment_resume,
        collision_fragments=_ABANDONMENT_RESUME_COLLISIONS,
    ),
}


def repairs_for_layer(layer: str) -> list[Repair]:
    """All catalog repairs targeting a harness layer (stable order)."""
    return [r for r in CATALOG.values() if r.layer == layer]
