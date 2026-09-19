"""Regression: the git tool must TEACH correct staging instead of guessing.

A live run (2026-09-19) saw the model call ``git(subcommand="add", args="-")``
right after a bare "push". A bare ``-`` is not a pathspec, so git answered with
"fatal: pathspec '-' did not match any files" and the call was wasted, even
though there was nothing to stage.

The fix is instruction-first — the git tool description and the system prompt
now say to run ``status`` first and to stage with ``args="-A"`` — plus a
handler backstop that answers a bare/empty ``add`` with the correct form.
"""

from __future__ import annotations

import asyncio

from agent import Agent
from agent_core.tool_schemas import NLP_TOOL_SCHEMAS


def _git_schema() -> dict:
    return next(
        s for s in NLP_TOOL_SCHEMAS if s["function"]["name"] == "git"
    )


class TestGitToolIsSelfDescribing:
    def test_description_teaches_add_all_and_status_first(self) -> None:
        desc = _git_schema()["function"]["description"]
        assert "-A" in desc
        assert "status" in desc.lower()
        # The exact mistake must be called out.
        assert "-" in desc and "pathspec" in desc.lower()

    def test_args_description_gives_concrete_examples(self) -> None:
        args = _git_schema()["function"]["parameters"]["properties"]["args"]
        text = args["description"]
        assert "-A" in text
        assert "-m" in text  # commit example

    def test_args_required_for_the_common_subcommands(self) -> None:
        props = _git_schema()["function"]["parameters"]["properties"]
        sub = props["subcommand"]["description"]
        for name in ("add", "commit", "push", "status"):
            assert name in sub


class TestSystemPromptInstructsGit:
    def test_prompt_names_status_first_and_add_all(self) -> None:
        import agent as agent_mod

        text = agent_mod._SYSTEM_PROMPT
        assert "subcommand='add'" in text
        assert "args='-A'" in text
        assert "status" in text


class TestGitAddBackstop:
    """The handler never forwards a bare '-' / empty add to git."""

    @staticmethod
    def _agent() -> Agent:
        return Agent.__new__(Agent)

    def test_bare_dash_add_returns_guidance(self) -> None:
        out = asyncio.run(
            self._agent()._nlp_git({"subcommand": "add", "args": "-"})
        )
        assert "pathspec" in out
        assert "-A" in out

    def test_empty_add_returns_guidance(self) -> None:
        out = asyncio.run(self._agent()._nlp_git({"subcommand": "add"}))
        assert "needs paths" in out
        assert "-A" in out
