"""``propose`` — generate code, validate in memory, emit a reviewable bundle.

This is a thin front-end: it forwards to :class:`ImplementCommand` with the
``--propose`` flag set, so 100% of the generation, parsing and gating logic is
shared.  The agent never writes the working tree — it produces a proposal
bundle (markdown + ``git apply``-compatible ``.patch``) that a human or CI
merges.  This is strictly safer than the autonomous driver's ``git stash``
checkpoint because there is no dirty-tree window to corrupt.

See :mod:`agent_core.commands.proposal_core` for the bundle format and the
in-memory apply logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Command
from .implement_cmd import ImplementCommand

if TYPE_CHECKING:
    from agent import Agent


class ProposeCommand(Command):
    """Generate a proposal bundle without modifying the working tree."""

    @property
    def name(self) -> str:
        return "propose"

    @property
    def help_text(self) -> str:
        return (
            "propose <taskplan.md> [analysis.md] [plan.md] [entities.md] "
            "[--modify] [--validate] [--out <dir>] [--workspace <path>] "
            "- Generate a reviewed diff bundle; never writes the tree"
        )

    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        # Forward to implement --propose so all generation logic is reused.
        # argparse-style flags already understood by implement are passed
        # through verbatim; we just guarantee --propose is present.
        parts = list(args)
        if "--propose" not in parts:
            parts = ["--propose", *parts]
        return await ImplementCommand().execute(parts, agent)
