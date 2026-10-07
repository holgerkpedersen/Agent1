"""The session's original request must survive chat-history compaction.

Regression for the agent "forgetting the first prompt": the chat-history
projection keeps the system prompt and trims the OLDEST body messages
(``_MAX_CHAT_MESSAGES`` / ``_HISTORY_CHAR_BUDGET``), which is exactly where
the user's first prompt lives — and because the trim mutates the persisted
history, the request was gone permanently (also after a restart).

The fix pins the goal into the system prompt as a dynamic block
(``ORIGINAL TASK``) that is rebuilt every turn and stripped/re-injected like
the decision-constraints block, repeats an excerpt in the compaction note when
messages have to be dropped, and persists it in ``agent_memory.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import agent
from agent import (
    Agent,
    _HISTORY_CHAR_BUDGET,
    _MAX_CHAT_MESSAGES,
    _trim_chat_history,
)
from agent_core.memory import MAX_GOAL_CHARS, ORIGINAL_GOAL_MARKER
from agent_core.modes import MODE_PLAN

GOAL = "Byg en CLI der tæller ord i en fil og printer top-10."


@pytest.fixture()
def bot(tmp_path: Path, monkeypatch) -> Agent:
    hist = tmp_path / "chat_history.json"
    mem = tmp_path / "agent_memory.json"
    monkeypatch.setattr(agent, "CHAT_HISTORY_JSON_PATH", str(hist))
    monkeypatch.setattr(agent, "CHAT_HISTORY_TMP_PATH", str(hist) + ".tmp")
    monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem))
    monkeypatch.setattr(agent, "AGENT_MEMORY_TMP_PATH", str(mem) + ".tmp")
    return Agent(workspace=str(tmp_path))


def _fill_body(bot: Agent, count: int) -> None:
    """Append *count* plain user/assistant exchanges to the history body."""
    for i in range(count):
        bot._chat_history.append({"role": "user", "content": f"besked {i}"})
        bot._chat_history.append({"role": "assistant", "content": f"svar {i}"})


class TestOriginalGoalRetention:
    def test_first_prompt_becomes_the_goal(self, bot: Agent) -> None:
        bot._append_user_turn(GOAL, None)
        assert bot._original_goal.startswith("Byg en CLI")

    def test_later_prompts_do_not_replace_the_goal(self, bot: Agent) -> None:
        bot._append_user_turn(GOAL, None)
        bot._append_user_turn("Noget helt andet nu", None)
        assert bot._original_goal.startswith("Byg en CLI")

    def test_goal_is_the_raw_input_not_the_steering_note(self, bot: Agent) -> None:
        """Plan mode prepends a steering note to the stored message; the pinned
        goal must be what the user actually asked for."""
        bot.mode = MODE_PLAN
        bot._append_user_turn(GOAL, None)
        assert bot._original_goal == GOAL

    def test_goal_text_is_capped(self, bot: Agent) -> None:
        bot._append_user_turn("y" * (MAX_GOAL_CHARS + 500), None)
        assert len(bot._original_goal) == MAX_GOAL_CHARS


class TestOriginalGoalInPrompt:
    def test_goal_is_injected_into_the_system_prompt(self, bot: Agent) -> None:
        bot._append_user_turn(GOAL, None)
        bot._refresh_system_message()
        prompt = bot._chat_history[0]["content"]
        assert ORIGINAL_GOAL_MARKER in prompt
        assert "Byg en CLI" in prompt

    def test_goal_block_is_rebuilt_not_accumulated(self, bot: Agent) -> None:
        bot._append_user_turn(GOAL, None)
        for _ in range(3):
            bot._refresh_system_message()
        prompt = bot._chat_history[0]["content"]
        assert prompt.count(ORIGINAL_GOAL_MARKER) == 1

    def test_prompt_without_goal_has_no_block(self, bot: Agent) -> None:
        bot._refresh_system_message()
        assert ORIGINAL_GOAL_MARKER not in bot._chat_history[0]["content"]

    def test_strip_removes_the_goal_block(self, bot: Agent) -> None:
        bot._append_user_turn(GOAL, None)
        bot._refresh_system_message()
        stripped = agent._strip_dynamic_system_blocks(
            bot._chat_history[0]["content"],
        )
        assert ORIGINAL_GOAL_MARKER not in stripped
        # The BASE prompt itself survives the strip.
        assert stripped.strip()


class TestOriginalGoalSurvivesCompaction:
    def test_goal_survives_the_message_cap(self, bot: Agent) -> None:
        bot._append_user_turn(GOAL, None)
        bot._refresh_system_message()
        _fill_body(bot, _MAX_CHAT_MESSAGES + 20)
        trimmed = _trim_chat_history(list(bot._chat_history), bot._original_goal)
        # The system prompt is always kept and still carries the pinned goal.
        assert trimmed[0]["role"] == "system"
        assert ORIGINAL_GOAL_MARKER in trimmed[0]["content"]
        assert "Byg en CLI" in trimmed[0]["content"]

    def test_goal_survives_the_char_budget(self, bot: Agent) -> None:
        bot._append_user_turn(GOAL, None)
        bot._refresh_system_message()
        # One huge body message blows the char budget on its own, so the trim
        # MUST drop it and leave the compaction note behind.
        bot._chat_history.append({
            "role": "user",
            "content": "x" * (_HISTORY_CHAR_BUDGET + 500),
        })
        bot._chat_history.append({"role": "assistant", "content": "kort svar"})
        trimmed = _trim_chat_history(list(bot._chat_history), bot._original_goal)
        notes = [
            str(m.get("content") or "")
            for m in trimmed
            if "context compaction" in str(m.get("content") or "")
        ]
        assert notes, "no compaction note was emitted"
        assert "Original task" in notes[0]
        assert "Byg en CLI" in notes[0]

    def test_trim_note_without_goal_is_unchanged(self, bot: Agent) -> None:
        bot._append_user_turn(GOAL, None)
        bot._refresh_system_message()
        bot._chat_history.append({
            "role": "user",
            "content": "x" * (_HISTORY_CHAR_BUDGET + 500),
        })
        bot._chat_history.append({"role": "assistant", "content": "kort svar"})
        trimmed = _trim_chat_history(list(bot._chat_history))
        notes = [
            str(m.get("content") or "")
            for m in trimmed
            if "context compaction" in str(m.get("content") or "")
        ]
        assert notes
        assert "Original task" not in notes[0]


class TestOriginalGoalPersistence:
    def test_goal_is_persisted_and_restored(self, bot: Agent) -> None:
        bot._append_user_turn(GOAL, None)
        bot._save_memory()
        revived = Agent(workspace=bot.workspace)
        assert revived._original_goal.startswith("Byg en CLI")
        revived._refresh_system_message()
        assert ORIGINAL_GOAL_MARKER in revived._chat_history[0]["content"]

    def test_goal_is_written_to_agent_memory_json(self, bot: Agent) -> None:
        bot._append_user_turn(GOAL, None)
        bot._save_memory()
        data = json.loads(
            Path(agent.AGENT_MEMORY_JSON_PATH).read_text(encoding="utf-8"),
        )
        assert data["original_goal"].startswith("Byg en CLI")

    def test_goal_derived_from_restored_history(self, bot: Agent) -> None:
        """A session restored from a history that predates the goal block must
        still recover the request from its first user message."""
        bot._chat_history = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": GOAL},
            {"role": "assistant", "content": "ok"},
        ]
        bot._original_goal = ""
        bot._ensure_original_goal()
        assert bot._original_goal == GOAL

    def test_goal_ignored_for_multimodal_without_text(self, bot: Agent) -> None:
        bot._chat_history = [
            {"role": "system", "content": "s"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,"},
                    },
                ],
            },
            {"role": "user", "content": GOAL},
        ]
        bot._original_goal = ""
        bot._ensure_original_goal()
        assert bot._original_goal == GOAL

    def test_clear_resets_the_goal(self, bot: Agent) -> None:
        bot._append_user_turn(GOAL, None)
        bot.clear_history()
        assert bot._original_goal == ""


class TestCompactionNoteAlwaysCarriesTheGoal:
    """Every path that DROPS history must leave a goal reminder behind.

    Regression: the count cap (``_MAX_CHAT_MESSAGES``) dropped the oldest
    messages — exactly where the first prompt lives — without emitting any
    compaction note at all (only the char-budget path did), and the persist
    projection (``_project_chat_history``) trimmed WITHOUT the goal, so the
    note it left behind never carried the ``Original task`` reminder.
    """

    def test_count_cap_trim_emits_note_with_goal(self, bot: Agent) -> None:
        bot._append_user_turn(GOAL, None)
        _fill_body(bot, _MAX_CHAT_MESSAGES + 20)
        trimmed = _trim_chat_history(list(bot._chat_history), bot._original_goal)
        notes = [
            str(m.get("content") or "")
            for m in trimmed
            if "context compaction" in str(m.get("content") or "")
        ]
        assert notes, "count-cap trim dropped messages without a compaction note"
        assert "Original task" in notes[0]
        assert "Byg en CLI" in notes[0]

    def test_count_cap_trim_still_fits_the_cap(self, bot: Agent) -> None:
        bot._append_user_turn(GOAL, None)
        _fill_body(bot, _MAX_CHAT_MESSAGES + 20)
        trimmed = _trim_chat_history(list(bot._chat_history), bot._original_goal)
        assert len(trimmed) <= _MAX_CHAT_MESSAGES

    def test_saved_note_carries_the_goal(self, bot: Agent) -> None:
        """The save-time projection trims too — its note must carry the goal."""
        bot._append_user_turn(GOAL, None)
        bot._refresh_system_message()
        _fill_body(bot, _MAX_CHAT_MESSAGES + 20)
        bot._save_chat_history()
        data = json.loads(
            Path(agent.CHAT_HISTORY_JSON_PATH).read_text(encoding="utf-8"),
        )
        notes = [
            str(m.get("content") or "")
            for m in data
            if "context compaction" in str(m.get("content") or "")
        ]
        assert notes, "save-time trim dropped messages without a compaction note"
        assert "Original task" in notes[0]
        assert "Byg en CLI" in notes[0]


class TestGoalRecoveryFromRestoredHistory:
    """Goal recovery must never pin a WRONG prompt as "the original request"."""

    def test_goal_recovered_from_system_block(self, bot: Agent) -> None:
        """agent_memory.json lost + history trimmed: the ORIGINAL TASK block in
        the persisted system prompt is the authoritative source.  Regression:
        recovery used to pin the first *surviving* user message ('msg 61') as
        the session's original request."""
        bot._append_user_turn(GOAL, None)
        bot._refresh_system_message()
        _fill_body(bot, _MAX_CHAT_MESSAGES + 20)
        bot._save_chat_history()
        bot._save_memory()
        Path(agent.AGENT_MEMORY_JSON_PATH).unlink()
        revived = Agent(workspace=bot.workspace)
        assert revived._original_goal.startswith("Byg en CLI")

    def test_goal_recovered_from_note_excerpt(self, bot: Agent) -> None:
        """No system block (old history): the compaction note's excerpt is the
        next best source — and the note text itself must never become the goal."""
        bot._chat_history = [
            {"role": "system", "content": "s"},
            {
                "role": "user",
                "content": agent._HISTORY_TRIM_NOTE.format(dropped=3)
                + "\nOriginal task (do not lose sight of it): " + GOAL,
            },
            {"role": "user", "content": "senere besked"},
        ]
        bot._original_goal = ""
        bot._ensure_original_goal()
        assert bot._original_goal.startswith("Byg en CLI")
        assert "context compaction" not in bot._original_goal

    def test_no_goal_pinned_when_original_is_unrecoverable(self, bot: Agent) -> None:
        """A later message must NOT be promoted to "the original request" —
        the block explicitly steers every answer towards it, so a wrong pin is
        worse than none."""
        bot._chat_history = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": agent._HISTORY_TRIM_NOTE.format(dropped=3)},
            {"role": "user", "content": "senere besked"},
        ]
        bot._original_goal = ""
        bot._ensure_original_goal("endnu senere besked")
        assert bot._original_goal == ""

    def test_goal_strips_plan_mode_wrapper(self, bot: Agent) -> None:
        from agent_core.modes import plan_mode_turn_note

        wrapped = f"{plan_mode_turn_note()}\n\n{GOAL}"
        bot._chat_history = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": wrapped},
        ]
        bot._original_goal = ""
        bot._ensure_original_goal()
        assert bot._original_goal == GOAL

    def test_goal_strips_skill_hint_wrapper(self, bot: Agent) -> None:
        wrapped = (
            "Skill hints (load with read_skill when the task matches):\n"
            "- fixer\n\n" + GOAL
        )
        bot._chat_history = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": wrapped},
        ]
        bot._original_goal = ""
        bot._ensure_original_goal()
        assert bot._original_goal == GOAL


class TestGoalBlockExtraction:
    def test_round_trip(self) -> None:
        from agent_core.memory import original_goal_block, original_goal_from_block

        prompt = "BASE PROMPT" + original_goal_block(GOAL) + "\nother block"
        assert original_goal_from_block(prompt) == GOAL

    def test_empty_when_absent(self) -> None:
        from agent_core.memory import original_goal_from_block

        assert original_goal_from_block("BASE PROMPT") == ""


class TestSystemHeadNeverSwallowsTheFirstPrompt:
    def test_non_system_first_message_is_not_swallowed(self, bot: Agent) -> None:
        """A history whose [0] is a USER message must not have it rewritten
        into the system prompt — the user's message stays in the body."""
        bot._chat_history = [{"role": "user", "content": GOAL}]
        bot._refresh_system_message()
        assert bot._chat_history[0]["role"] == "system"
        assert any(
            m.get("role") == "user" and m.get("content") == GOAL
            for m in bot._chat_history
        )
