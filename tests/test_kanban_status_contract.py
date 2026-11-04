"""Cross-repo contract: Agent1's Kanban status tables must match Kanban's own.

Agent1 hard-codes the card↔issue status mapping (``kanban_bridge.FRAME_TO_STATUS``
and ``kanban_inspect.CARD_STATUS_TO_ISSUE`` / ``ISSUE_STATUS_TO_CARD``). The
Kanban app validates card status against its own frozenset
(``src/agent1/models.py:VALID_STATUSES``) and owns the column titles. A rename
on either side silently drops statuses from the bridge — so pin it with a test.

Skipped when the Kanban checkout is not next to this repo (same convention as
tests/test_kanban_roundtrip.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

AGENT1_ROOT = Path(__file__).resolve().parents[1]
KANBAN_ROOT = AGENT1_ROOT.parent / "Kanban"
sys.path.insert(0, str(AGENT1_ROOT))

if not (KANBAN_ROOT / "src" / "agent1" / "models.py").exists():
    pytest.skip(f"Kanban checkout not found at {KANBAN_ROOT}", allow_module_level=True)

sys.path.insert(0, str(KANBAN_ROOT))

from src.agent1.models import VALID_STATUSES  # noqa: E402

from harnessfix import kanban_bridge as kb  # noqa: E402
from harnessfix import kanban_inspect as ki  # noqa: E402


def test_card_status_targets_are_all_valid_kanban_statuses() -> None:
    """Every status Agent1 writes back must be one Kanban accepts."""
    assert set(ki.ISSUE_STATUS_TO_CARD.values()) <= set(VALID_STATUSES)


def test_card_to_issue_covers_every_kanban_status() -> None:
    """Every status Kanban can send has an Agent1 landing spot."""
    assert set(ki.CARD_STATUS_TO_ISSUE.keys()) == set(VALID_STATUSES)


def test_column_mapping_matches_bridge_frame_table() -> None:
    """The inspector must not invent its own column→status table."""
    assert ki.COLUMN_TO_ISSUE == kb.FRAME_TO_STATUS


def test_status_round_trip_is_stable() -> None:
    """issue → card → issue for the statuses both sides share."""
    for issue_status in ("open", "planned", "in-progress", "reviewing"):
        card_status = ki.ISSUE_STATUS_TO_CARD[issue_status]
        assert ki.CARD_STATUS_TO_ISSUE[card_status] == issue_status


def test_terminal_status_maps_to_done() -> None:
    """Agent1's terminal state lands the card in Kanban's 'done'."""
    assert ki.ISSUE_STATUS_TO_CARD["resolved"] == "done"
    assert ki.CARD_STATUS_TO_ISSUE["done"] == "resolved"
