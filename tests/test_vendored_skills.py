"""Guard for the vendored runbook skills under ``skills/``.

The five vendored skills (TDD, systematic debugging, verification-before-
completion, code review, subagent orchestration) are loaded through
:mod:`agent_core.skills`, which validates frontmatter up front and silently
skips anything malformed.  Without a test, a future edit that adds an
unsupported frontmatter field would drop the skill from the index with only
a log warning — the runbook would vanish from every session.

These tests assert against the REAL discovery path (``discover_skills`` on
the actual workspace), not a copy of it.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from agent_core.skills import (
    SKILL_FILENAME,
    discover_skills,
    read_skill,
    skill_index_block,
)

WORKSPACE = Path(__file__).resolve().parent.parent

#: The vendored runbook set — each must stay loadable and indexed.
VENDORED_SKILLS = {
    "requesting-code-review",
    "subagent-orchestration",
    "systematic-debugging",
    "test-driven-development",
    "verification-before-completion",
}


def test_vendored_skills_are_discovered() -> None:
    found = {skill.name for skill in discover_skills(WORKSPACE)}
    missing = VENDORED_SKILLS - found
    assert not missing, f"vendored skills failed discovery (skipped?): {missing}"


def test_every_skill_dir_on_disk_passes_validation(caplog) -> None:
    """No ``skills/<name>/SKILL.md`` may be silently skipped.

    A malformed skill is logged as a warning and dropped — turn that
    fail-open behavior into a hard failure here so it cannot hide.
    """
    root = WORKSPACE / "skills"
    assert root.is_dir(), "skills/ directory missing from the workspace"
    with caplog.at_level(logging.WARNING, logger="agent_core.skills"):
        discover_skills(WORKSPACE)
    skipped = [rec.getMessage() for rec in caplog.records if "Skill skipped" in rec.getMessage()]
    assert not skipped, f"skills on disk failed validation: {skipped}"


def test_index_block_lists_all_vendored_skills() -> None:
    block = skill_index_block(discover_skills(WORKSPACE))
    for name in sorted(VENDORED_SKILLS):
        assert f"- {name} — " in block, f"{name} missing from the system-prompt index"


@pytest.mark.parametrize("skill_name", sorted(VENDORED_SKILLS))
def test_skill_body_pages_cleanly(skill_name: str) -> None:
    page = read_skill(WORKSPACE, skill_name, offset=1, limit=400)
    assert page.skill.name == skill_name
    assert page.total_lines > 0
    # Bodies are kept within one page so a single read_skill call gets the runbook.
    assert not page.body_truncated
    assert page.next_offset is None
