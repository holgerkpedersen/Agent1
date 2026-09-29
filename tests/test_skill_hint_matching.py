"""Per-turn skill hint: matched skills are advertised in the user message.

Regression guard for :func:`agent_core.skills.match_skills_for_input` /
``skill_hint_block`` and their wiring into ``Agent._append_user_turn`` — a
hint must appear ONLY when a skill's name/tags actually match the input,
stay out of multimodal turns (image content arrays), and never kill a turn.
"""
from __future__ import annotations

import logging
from pathlib import Path

from agent_core.skills import (
    Skill,
    match_skills_for_input,
    skill_hint_block,
)

WORKSPACE = Path(__file__).resolve().parent.parent


def _skill(name: str, description: str, tags: tuple[str, ...] = ()) -> Skill:
    return Skill(
        name=name,
        description=description,
        when_to_use=None,
        tags=tags,
        path=WORKSPACE / "skills" / name / "SKILL.md",
    )


# ---------------------------------------------------------------------------
# Pure matching logic
# ---------------------------------------------------------------------------

def test_match_by_name_and_tags() -> None:
    skills = [
        _skill("repo-runbook", "Run the repo tests and build.", ("pytest",)),
        _skill("other-skill", "Unrelated skill."),
    ]
    matched = match_skills_for_input(skills, "run pytest on the suite")
    assert [s.name for s in matched] == ["repo-runbook"]


def test_match_is_case_insensitive_and_token_based() -> None:
    skills = [_skill("tdd", "", ("test-driven-development",))]
    # Tag appears as a full token inside the text.
    assert match_skills_for_input(skills, "Use Test-Driven-Development here")
    # Name substring also matches (deliberately conservative).
    assert match_skills_for_input(skills, "tdd this feature please")


def test_no_match_returns_empty() -> None:
    skills = [_skill("repo-runbook", "", ("pytest",))]
    assert match_skills_for_input(skills, "hello world") == []
    assert match_skills_for_input([], "hello") == []
    assert match_skills_for_input(skills, "") == []


def test_match_respects_max_matches() -> None:
    skills = [
        _skill(f"skill-{i}", "", (f"tag{i}",)) for i in range(5)  # noqa: E231
    ]
    text = " ".join(s.tags[0] for s in skills)
    assert len(match_skills_for_input(skills, text)) == 3
    assert len(match_skills_for_input(skills, text, max_matches=2)) == 2


# ---------------------------------------------------------------------------
# Hint formatting
# ---------------------------------------------------------------------------

def test_hint_block_empty_when_nothing_matched() -> None:
    assert skill_hint_block([]) == ""


def test_hint_block_lists_matched_names_and_description() -> None:
    block = skill_hint_block([_skill("repo-runbook", "Run the repo tests.")])
    assert "Skill hints" in block
    assert "- repo-runbook — Run the repo tests." in block


# ---------------------------------------------------------------------------
# Wiring into Agent._append_user_turn (real path, no network)
# ---------------------------------------------------------------------------

def _make_agent(tmp_path: Path, monkeypatch):
    import agent as agent_mod

    monkeypatch.setattr(
        agent_mod, "CHAT_HISTORY_JSON_PATH", str(tmp_path / "chat_history.json")
    )
    monkeypatch.setattr(
        agent_mod, "AGENT_MEMORY_JSON_PATH", str(tmp_path / "agent_memory.json")
    )
    return agent_mod.Agent(workspace=str(WORKSPACE))


def test_hint_injected_into_user_message_when_skill_matches(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    bot = _make_agent(tmp_path, monkeypatch)
    with caplog.at_level(logging.WARNING):
        bot._append_user_turn("please run pytest on the suite", None)
    content = bot._chat_history[-1]["content"]
    assert "Skill hints" in content
    # The original text is preserved after the hint.
    assert "please run pytest on the suite" in content


def test_no_hint_when_no_skill_matches(tmp_path: Path, monkeypatch) -> None:
    bot = _make_agent(tmp_path, monkeypatch)
    bot._append_user_turn("hello there", None)
    content = bot._chat_history[-1]["content"]
    assert "Skill hints" not in content
    assert content == "hello there"


def test_no_hint_on_multimodal_turns(tmp_path: Path, monkeypatch) -> None:
    """Image turns keep their OpenAI content-array shape untouched."""
    bot = _make_agent(tmp_path, monkeypatch)
    bot._append_user_turn("run pytest", ["data:image/png;base64,AAAA"])
    entry = bot._chat_history[-1]
    assert isinstance(entry["content"], list)
    text_blocks = [b for b in entry["content"] if b.get("type") == "text"]
    assert len(text_blocks) == 1
    assert "Skill hints" not in text_blocks[0]["text"]
