"""Regression: the git tool must TEACH correct staging instead of guessing.

A live run (2026-09-19) saw the model call ``git(subcommand="add", args="-")``
right after a bare "push". A bare ``-`` is not a pathspec, so git answered with
"fatal: pathspec '-' did not match any files" and the call was wasted, even
though there was nothing to stage.

The fix is instruction-first — the git tool description and the system prompt
now say to run ``status`` first and to stage with ``args="-A"`` — plus a
handler backstop that answers a bare/empty ``add`` with the correct form.

A second live run (2026-09-28) asked "commit changes" and the model called
``git(subcommand="push")`` alone: the staged changes were never committed and
the push sent only the previous commit. The backstops below stop that class of
mistake — a message-less ``commit`` (editor hang) and a ``push`` with staged
work still uncommitted — and the prompt/schema say the full sequence.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

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

    def test_prompt_says_commit_changes_is_the_full_sequence(self) -> None:
        import agent as agent_mod

        text = agent_mod._SYSTEM_PROMPT.lower()
        assert "commit changes" in text
        assert "never push alone" in text


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


class TestGitPushAndCommitBackstops:
    """A message-less commit / a push over staged work must not waste a call.

    Regression (2026-09-28): "commit changes" ran ``git push`` alone while
    files were staged, so the push sent the previous commit and silently left
    the staged work behind. These backstops answer with guidance instead.
    """

    @staticmethod
    def _agent() -> Agent:
        return Agent.__new__(Agent)

    @staticmethod
    def _repo_with_staged_change(tmp_path: Path) -> Agent:
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=False)
        subprocess.run(
            ["git", "-C", str(tmp_path), "config", "user.email", "t@t"],
            check=False,
        )
        subprocess.run(
            ["git", "-C", str(tmp_path), "config", "user.name", "t"],
            check=False,
        )
        (tmp_path / "base.txt").write_text("base\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(tmp_path), "add", "base.txt"], check=False,
        )
        subprocess.run(
            ["git", "-C", str(tmp_path), "commit", "-qm", "init"], check=False,
        )
        (tmp_path / "staged.txt").write_text("staged\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(tmp_path), "add", "staged.txt"], check=False,
        )
        return Agent(workspace=str(tmp_path))

    def test_push_with_staged_work_returns_commit_guidance(
        self, tmp_path: Path
    ) -> None:
        out = asyncio.run(
            self._repo_with_staged_change(tmp_path)._nlp_git(
                {"subcommand": "push"}
            )
        )
        assert "staged" in out.lower()
        assert "staged.txt" in out
        assert "commit" in out.lower()
        # The push must NOT have run: guidance, not a git error/result.
        assert "guidance" in out.lower()

    def test_clean_tree_push_is_not_blocked(self, tmp_path: Path) -> None:
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=False)
        subprocess.run(
            ["git", "-C", str(tmp_path), "config", "user.email", "t@t"],
            check=False,
        )
        subprocess.run(
            ["git", "-C", str(tmp_path), "config", "user.name", "t"],
            check=False,
        )
        (tmp_path / "base.txt").write_text("base\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(tmp_path), "add", "base.txt"], check=False,
        )
        subprocess.run(
            ["git", "-C", str(tmp_path), "commit", "-qm", "init"], check=False,
        )
        out = asyncio.run(
            Agent(workspace=str(tmp_path))._nlp_git({"subcommand": "push"})
        )
        assert "guidance" not in out.lower()

    def test_commit_without_message_returns_guidance(self) -> None:
        out = asyncio.run(self._agent()._nlp_git({"subcommand": "commit"}))
        assert "message" in out.lower()
        assert "-m" in out

    def test_bare_amend_commit_is_blocked(self) -> None:
        out = asyncio.run(
            self._agent()._nlp_git({"subcommand": "commit", "args": "--amend"})
        )
        assert "guidance" in out.lower()
        assert "message" in out.lower()

    def test_amend_no_edit_commit_is_not_blocked(self, tmp_path: Path) -> None:
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=False)
        subprocess.run(
            ["git", "-C", str(tmp_path), "config", "user.email", "t@t"],
            check=False,
        )
        subprocess.run(
            ["git", "-C", str(tmp_path), "config", "user.name", "t"],
            check=False,
        )
        (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(tmp_path), "add", "a.txt"], check=False)
        subprocess.run(
            ["git", "-C", str(tmp_path), "commit", "-qm", "init"], check=False,
        )
        agent = Agent(workspace=str(tmp_path))
        out = asyncio.run(
            agent._nlp_git({"subcommand": "commit", "args": "--amend --no-edit"})
        )
        assert "guidance" not in out.lower()

    def test_commit_with_message_is_not_blocked(self, tmp_path: Path) -> None:
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=False)
        subprocess.run(
            ["git", "-C", str(tmp_path), "config", "user.email", "t@t"],
            check=False,
        )
        subprocess.run(
            ["git", "-C", str(tmp_path), "config", "user.name", "t"],
            check=False,
        )
        (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(tmp_path), "add", "a.txt"], check=False)
        out = asyncio.run(
            Agent(workspace=str(tmp_path))._nlp_git(
                {"subcommand": "commit", "args": '-m "feat: test"'}
            )
        )
        assert "guidance" not in out.lower()
