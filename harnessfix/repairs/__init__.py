"""Phase 3 - scoped repair catalog (spec section 3.4).

One module per harness layer; each repair is a small, revertible, code-level
change backed by trace evidence.  Repairs are applied via exact source
transforms and verified with py_compile; the loop reverts any repair that
fails the test/security gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .tool_interface import apply as _apply_tool_interface
from .tool_interface import revert as _revert_tool_interface
from .tool_interface import TOOL_INTERFACE_REPAIR_ID
from .tool_interface import COLLISION_FRAGMENTS

#: Harness layer this repair targets (must be a valid HarnessFix facet).
TOOL_INTERFACE_LAYER = "tool_interface"


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
}


def repairs_for_layer(layer: str) -> list[Repair]:
    """All catalog repairs targeting a harness layer (stable order)."""
    return [r for r in CATALOG.values() if r.layer == layer]
