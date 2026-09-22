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


# ---------------------------------------------------------------------------
# The index must actually reach the chat system prompt
# ---------------------------------------------------------------------------
#
# Regression guard.  ``Agent._refresh_system_message`` rebuilds the dynamic
# system blocks FROM THE WORKSPACE on every turn, and the skill index is one of
# them (``Agent._skill_index_block`` -> ``load_skill_index``).  Vendoring these
# runbooks therefore changed the chat system prompt of any agent whose
# workspace is the repo -- which is what surfaced a previously passing test
# (``test_tool_loop_nlp.py`` trimmed history, asserted an exact "SYS" system
# message while running ``Agent(workspace=".")``).  The prompt change is
# CORRECT and the intent of the feature; these tests pin it through the real
# path so it can neither silently regress nor break unnoticed again.

def _system_prompt_for(workspace: Path, tmp_path: Path, monkeypatch) -> str:
    """Build an agent on *workspace* and return its refreshed system message.

    Both persisted-state files are redirected into ``tmp_path`` so the test
    never reads or writes the developer's real ``chat_history.json`` /
    ``agent_memory.json`` (a restored session's messages must not leak in).
    """
    import agent as agent_mod

    monkeypatch.setattr(
        agent_mod, "CHAT_HISTORY_JSON_PATH", str(tmp_path / "chat_history.json")
    )
    monkeypatch.setattr(
        agent_mod, "AGENT_MEMORY_JSON_PATH", str(tmp_path / "agent_memory.json")
    )
    bot = agent_mod.Agent(workspace=str(workspace))
    bot._refresh_system_message()
    return bot._chat_history[0]["content"]


def test_skill_index_reaches_chat_system_prompt(tmp_path, monkeypatch) -> None:
    """The vendored index is injected into the real chat system prompt."""
    from agent_core.skills import SKILL_INDEX_MARKER

    content = _system_prompt_for(WORKSPACE, tmp_path, monkeypatch)
    assert SKILL_INDEX_MARKER in content, "skill index missing from system prompt"
    for name in sorted(VENDORED_SKILLS):
        assert f"- {name} — " in content, f"{name} missing from the system prompt"


def test_skill_index_is_rebuilt_not_accumulated(tmp_path, monkeypatch) -> None:
    """Refreshing twice must not stack a second stale index block."""
    import agent as agent_mod
    from agent_core.skills import SKILL_INDEX_MARKER

    monkeypatch.setattr(
        agent_mod, "CHAT_HISTORY_JSON_PATH", str(tmp_path / "chat_history.json")
    )
    monkeypatch.setattr(
        agent_mod, "AGENT_MEMORY_JSON_PATH", str(tmp_path / "agent_memory.json")
    )
    bot = agent_mod.Agent(workspace=str(WORKSPACE))
    bot._refresh_system_message()
    first = bot._chat_history[0]["content"]
    bot._refresh_system_message()
    second = bot._chat_history[0]["content"]
    assert second == first, "dynamic blocks accumulated across refreshes"
    assert second.count(SKILL_INDEX_MARKER) == 1
    # The BASE prompt is what survives stripping -- nothing else leaks in.
    assert agent_mod._strip_dynamic_system_blocks(second) == agent_mod._SYSTEM_PROMPT


def test_workspace_without_skills_gets_no_index(tmp_path, monkeypatch) -> None:
    """The injection is workspace-derived -- the reason workspace="." broke.

    A workspace with no ``skills/`` directory must yield the untouched base
    prompt, byte for byte (empty string block, same rule as the decision
    constraints block).
    """
    from agent_core.skills import SKILL_INDEX_MARKER
    import agent as agent_mod

    content = _system_prompt_for(tmp_path, tmp_path, monkeypatch)
    assert SKILL_INDEX_MARKER not in content
    assert content == agent_mod._SYSTEM_PROMPT
