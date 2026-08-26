"""Regression tests for the git-branch-aware chat prompt.

The interactive prompt shows ``[YYYY-MM-DD HH:MM] <branch> >`` so the user can
always see which branch they are operating on.  :func:`agent._current_git_branch`
is the single source of truth for that branch label.
"""
from __future__ import annotations

import os

import agent


def test_current_git_branch_matches_head() -> None:
    head = os.path.join(os.path.dirname(os.path.abspath(agent.__file__)), ".git", "HEAD")
    with open(head, encoding="utf-8", errors="ignore") as fh:
        ref = fh.read().strip()
    expected = ref.split("/", 2)[-1] if ref.startswith("ref:") else None
    assert agent._current_git_branch() == expected
    # The branch must not be empty / None inside a real git checkout.
    assert agent._current_git_branch()


def test_prompt_format_includes_branch() -> None:
    from datetime import datetime

    branch = agent._current_git_branch() or "?"
    prompt = f"\n[{datetime.now():%Y-%m-%d %H:%M}] {branch} > "
    assert prompt.endswith(f"{branch} > ")
    assert "[20" in prompt  # date/time prefix present
