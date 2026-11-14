"""Regression: throwaway test fixtures must never be published to the Kanban board.

Live board evidence for this bug (2026-11-14): card
``e1b68fa4d077402db4171102a53d310e`` sat in *Issues* titled ``x``, linked to
``iss-best-effort-except-C__Users_..._pytest_10378_test_resolve_is...``.  A
pytest run created that issue and the outbox published it as a real card.

``make_issue`` already suppresses the outbox for a non-canonical ``ledger_path``
(see ``tests/test_issues_ledger.py``),
but that single gate leaks in three ways:

1. ``ledger_path=None`` is treated as "canonical" by design, so every fixture
   call such as ``make_issue("best-effort-except", "x", ["agent.py:1"])``
   passes the gate;
2. ``resolve`` / ``update_issue`` / ``promote`` take **no** ``ledger_path`` at
   all, so their ``issue_resolve`` / ``issue_update`` ops are gated only by the
   env switch — a scratch-ledger issue re-publishes on every status change;
3. the placeholder titles (``x``, ``y``, ``z``, ``low``, ``high``, ``bench``)
   and pytest-temp-path titles that the fixtures use are exactly the titles
   ``harnessfix/kanban_purge.py`` classifies as junk — but nothing stopped them
   at the source, so purge could only clean up after the fact.

The guard belongs at the outbox: ``_enqueue_sync`` is the single funnel every
issue op and every child-card emission passes through.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from harnessfix import issues as issue_store
from harnessfix import kanban_junk


class _RecordingBridge:
    """Stand-in for ``kanban_bridge`` that records outbox payloads."""

    def __init__(self) -> None:
        self.ops: list[dict[str, Any]] = []

    def enqueue(self, payload: dict[str, Any]) -> None:  # noqa: D102
        self.ops.append(payload)

    def op_names(self) -> list[str]:
        return [p.get("op", "") for p in self.ops]


@pytest.fixture()
def outbox(monkeypatch: pytest.MonkeyPatch) -> _RecordingBridge:
    """Enable sync and capture every outbox payload the ledger would emit."""
    bridge = _RecordingBridge()
    monkeypatch.setattr(issue_store, "_should_sync", lambda: True)
    monkeypatch.setattr(issue_store, "_get_kanban_bridge", lambda: bridge)
    return bridge


# --------------------------------------------------------------------------
# 1. Placeholder titles are fixture noise — never publishable work.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("title", ["x", "y", "z", "low", "high", "bench"])
def test_placeholder_title_never_reaches_the_outbox(
    outbox: _RecordingBridge, title: str
) -> None:
    """``make_issue(..., "x", ...)`` with the default ledger must not publish."""
    issue = issue_store.make_issue("best-effort-except", title, ["agent.py:1"])

    assert issue["id"].startswith("iss-")
    assert outbox.ops == [], (
        f"fixture issue titled {title!r} was published to Kanban as {outbox.op_names()}"
    )


def test_placeholder_title_is_blocked_on_status_change_too(
    outbox: _RecordingBridge,
) -> None:
    """The resolve path takes no ``ledger_path``; it must still be gated."""
    issue = issue_store.make_issue("best-effort-except", "x", ["agent.py:1"])
    issues = [issue]
    outbox.ops.clear()

    assert issue_store.resolve(issues, issue["id"], "resolved") is True
    assert outbox.ops == [], f"resolve published a fixture issue: {outbox.op_names()}"


# --------------------------------------------------------------------------
# 2. Out-of-repo (pytest temp) locations are synthetic input.
# --------------------------------------------------------------------------

def test_pytest_temp_path_issue_never_reaches_the_outbox(
    outbox: _RecordingBridge, tmp_path: Path
) -> None:
    """A finding whose only location is a tmp file is not repo work."""
    location = f"{tmp_path.as_posix()}/sample.py:5"
    title = f"best-effort-except in {location}"

    issue_store.make_issue("best-effort-except", title, [location])

    assert outbox.ops == [], f"tmp-path finding published: {outbox.op_names()}"


# --------------------------------------------------------------------------
# 3. A non-canonical ledger is not repo work (the pre-existing gate).
# --------------------------------------------------------------------------

def test_scratch_ledger_issue_publishes_neither_parent_nor_child_card(
    outbox: _RecordingBridge, tmp_path: Path
) -> None:
    """The whole ring (create -> child card) stays silent for a scratch ledger."""
    ledger = tmp_path / ".issues.json"
    issue_store.make_issue(
        "best-effort-except",
        "a real looking finding",
        ["agent_core/llm/provider.py:42"],
        parent_card_id="card-1",
        emit_child_card=True,
        ledger_path=ledger,
    )

    assert outbox.ops == [], f"scratch ledger published: {outbox.op_names()}"


def test_external_absolute_location_without_temp_marker_is_not_published(
    outbox: _RecordingBridge,
) -> None:
    """Containment alone rejects an outside path (the temp marker is a shortcut)."""
    issue_store.make_issue(
        "best-effort-except",
        "best-effort-except in C:/Program Files/Other/sample.py:7",
        ["C:/Program Files/Other/sample.py:7"],
    )

    assert outbox.ops == [], f"out-of-repo finding published: {outbox.op_names()}"


# --------------------------------------------------------------------------
# 4. Over-blocking guard — real repo work MUST still sync.
# --------------------------------------------------------------------------

def test_real_repo_finding_is_still_published(outbox: _RecordingBridge) -> None:
    """The gate must not silence genuine findings (fail-open for real work)."""
    issue_store.make_issue(
        "best-effort-except",
        "best-effort-except in agent_core/llm/provider.py:42",
        ["agent_core/llm/provider.py:42"],
    )

    assert outbox.op_names() == ["issue_create"], (
        "real repo work was suppressed by the fixture gate"
    )


def test_child_card_is_still_emitted_for_real_work_on_a_card(
    outbox: _RecordingBridge,
) -> None:
    """Card-attributable real work keeps its child card (board visibility)."""
    issue_store.make_issue(
        "best-effort-except",
        "best-effort-except in agent_core/llm/provider.py:42",
        ["agent_core/llm/provider.py:42"],
        parent_card_id="card-1",
        emit_child_card=True,
    )

    assert outbox.op_names() == ["issue_create", "child_card_create"], outbox.op_names()


# --------------------------------------------------------------------------
# 5. Containment must not depend on the host filesystem flavour.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "location",
    [
        "C:/Program Files/Other/sample.py:7",
        "C:\\Program Files\\Other\\sample.py:7",
        "\\\\server\\share\\sample.py:7",
    ],
)
def test_drive_rooted_location_is_outside_on_every_host(location: str) -> None:
    """A Windows-absolute path is never repo-relative, even on a POSIX host.

    Regression: ``Path("C:/x/y.py")`` is *relative* on Linux, so joining it
    onto the repo root made it look like repo work.  The gate then published
    the fixture, and CI's Ubuntu job failed while Windows passed.
    """
    assert kanban_junk.in_repo(location) is False


def test_repo_relative_location_is_inside_on_every_host() -> None:
    """The fix must not invert the rule for genuine repo-relative findings."""
    assert kanban_junk.in_repo("agent_core/llm/provider.py:42") is True
