"""Regression tests for the mode-colored interactive chat prompt."""

from __future__ import annotations

import datetime


from agent import _build_chat_prompt, CHAT_INPUT_PROMPT
from agent_core.colors import blue, green, _RESET, _ENABLED


class _FakeAgent:
    """Minimal agent stub exposing a controllable ``is_plan_mode``."""

    def __init__(self, plan: bool) -> None:
        self._plan = plan

    def is_plan_mode(self) -> bool:
        return self._plan


def _strip_ansi(text: str) -> str:
    import re

    return re.sub(r"\033\[[0-9;]*m", "", text)


def test_build_prompt_build_mode_is_green_and_tagged():
    agent = _FakeAgent(plan=False)
    now = datetime.datetime(2025, 1, 2, 3, 4)
    prompt = _build_chat_prompt(agent, "main", now)
    # Header is one line; the caret prompt is separate (CHAT_INPUT_PROMPT).
    assert _strip_ansi(prompt) == "\n[2025-01-02 03:04] main [build]"
    assert CHAT_INPUT_PROMPT == "> "
    assert "\n" not in CHAT_INPUT_PROMPT and "\033" not in CHAT_INPUT_PROMPT
    if _ENABLED:
        assert green("[build]") in prompt
        assert _RESET in prompt


def test_build_prompt_plan_mode_is_blue_and_tagged():
    agent = _FakeAgent(plan=True)
    now = datetime.datetime(2025, 1, 2, 3, 4)
    prompt = _build_chat_prompt(agent, "dev", now)
    assert _strip_ansi(prompt) == "\n[2025-01-02 03:04] dev [plan]"
    if _ENABLED:
        assert blue("[plan]") in prompt


def test_build_prompt_plan_vs_build_differ():
    now = datetime.datetime(2025, 1, 2, 3, 4)
    build_p = _build_chat_prompt(_FakeAgent(plan=False), "b", now)
    plan_p = _build_chat_prompt(_FakeAgent(plan=True), "b", now)
    assert build_p != plan_p
    assert "[build]" in _strip_ansi(build_p)
    assert "[plan]" in _strip_ansi(plan_p)
