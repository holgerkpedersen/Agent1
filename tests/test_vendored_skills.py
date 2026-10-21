"""Guard for the vendored runbook skills under ``skills/``.

The vendored skills (TDD, systematic debugging, verification-before-
completion, code review, subagent orchestration, strategic jev use) are
loaded through
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
    SkillError,
    discover_skills,
    format_skill_page,
    read_skill,
    skill_index_block,
)

WORKSPACE = Path(__file__).resolve().parent.parent

#: The vendored runbook set — each must stay loadable and indexed.
VENDORED_SKILLS = {
    "asd-ste100",
    "jev-strategy",
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
    skipped = [
        rec.getMessage()
        for rec in caplog.records
        if "Skill skipped" in rec.getMessage()
    ]
    assert not skipped, f"skills on disk failed validation: {skipped}"


def test_index_block_lists_all_vendored_skills() -> None:
    block = skill_index_block(discover_skills(WORKSPACE))
    for name in sorted(VENDORED_SKILLS):
        assert f"- {name} — " in block, f"{name} missing from the system-prompt index"


def test_asd_ste100_skill_loads_with_metadata() -> None:
    """Regression: asd-ste100 was silently skipped ("unterminated frontmatter
    block") because its SKILL.md lacked a closing ``---`` and used nested keys
    that the flat frontmatter parser rejects.  Pin the loaded metadata."""
    by_name = {skill.name: skill for skill in discover_skills(WORKSPACE)}
    assert "asd-ste100" in by_name, "asd-ste100 skipped by discovery"
    skill = by_name["asd-ste100"]
    assert skill.description.strip()
    assert "ste100" in skill.tags


def test_asd_ste100_frontmatter_is_closed_and_flat() -> None:
    """The file must open and close its frontmatter block (flat key: value)."""
    text = (WORKSPACE / "skills" / "asd-ste100" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    lines = text.splitlines()
    assert lines[0].strip() == "---"
    closing = [i for i in range(1, len(lines)) if lines[i].strip() == "---"]
    assert closing, "frontmatter block is not closed with '---'"
    header = lines[1 : closing[0]]
    for line in header:
        assert not line.startswith((" ", "\t")), f"nested key not supported: {line!r}"


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
    # Isolate from the developer's .env: AGENT_DISPLAY_MODE=intent appends the
    # INTENT narration suffix, which must not leak into the byte-identical
    # baseline comparison.
    monkeypatch.delenv("AGENT_DISPLAY_MODE", raising=False)
    monkeypatch.chdir(tmp_path)
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


# ---------------------------------------------------------------------------
# Mangled-name recovery (the read_skill self-correction path)
# ---------------------------------------------------------------------------
#
# Regression (2026-10-14): a small model (qwen3.5-9b-FLM) appended a long
# deterministic digit/hex run -- containing a '+', so the slug check failed --
# to every read_skill name it generated, and the old rejection
# ("name '...' is not a slug") named no valid skill, so the model retried the
# same garbage six times and never loaded a runbook.  A mangled request must
# now either load the right skill (with a note teaching the exact name) or
# fail with the closest candidate and the full skill list.

#: The exact garbage suffix from that session's tool calls.
GARBAGE_SUFFIX = (
    "20087159643247196853019e+13dcdacddcdeedfbcdfcfbceecafabfdcafeffadfcbaefdcbeaeaffa"
)


@pytest.mark.parametrize("skill_name", sorted(VENDORED_SKILLS))
def test_skill_name_with_token_garbage_recovers(skill_name: str) -> None:
    """``<name>-<garbage>`` must load the skill and flag the correction."""
    mangled = f"{skill_name}-{GARBAGE_SUFFIX}"
    page = read_skill(WORKSPACE, mangled)
    assert page.skill.name == skill_name
    assert page.correction_note == (
        f"you requested {mangled!r} — loaded {skill_name!r} instead"
    )
    rendered = format_skill_page(page)
    assert rendered.splitlines()[0] == f"[note: {page.correction_note}]"


def test_skill_name_with_typo_and_garbage_recovers() -> None:
    """'jeve-strategy' (one extra letter) + garbage must load 'jev-strategy'."""
    page = read_skill(WORKSPACE, f"jeve-strategy-{GARBAGE_SUFFIX}")
    assert page.skill.name == "jev-strategy"
    assert page.correction_note is not None


def test_skill_name_with_typo_alone_recovers() -> None:
    """A one-letter typo without garbage is a fuzzy match, not an error."""
    page = read_skill(WORKSPACE, "test-drivrn-development")
    assert page.skill.name == "test-driven-development"
    assert page.correction_note is not None


def test_truncated_skill_name_recovers() -> None:
    """A truncated name must load the unique skill that extends it."""
    page = read_skill(WORKSPACE, "systematic-debug")
    assert page.skill.name == "systematic-debugging"
    assert page.correction_note is not None


def test_exact_skill_name_has_no_correction_note() -> None:
    page = read_skill(WORKSPACE, "jev-strategy")
    assert page.correction_note is None
    assert not format_skill_page(page).startswith("[note:")


def test_unknown_skill_error_lists_available_names() -> None:
    """An unrecoverable name must hand the model the full valid-name list."""
    with pytest.raises(SkillError) as excinfo:
        read_skill(WORKSPACE, "banana-split")
    message = str(excinfo.value)
    assert "available skills:" in message
    for name in sorted(VENDORED_SKILLS):
        assert name in message, f"{name} missing from the error message"
    assert "did you mean" not in message, "no near miss here, so no suggestion"


def test_unknown_skill_error_suggests_closest_name() -> None:
    """A near miss below the auto-correct cutoff gets a 'did you mean' hint."""
    with pytest.raises(SkillError) as excinfo:
        read_skill(WORKSPACE, "zzzzzzzzzz-development")
    message = str(excinfo.value)
    assert "did you mean 'test-driven-development'" in message
    assert "available skills:" in message
