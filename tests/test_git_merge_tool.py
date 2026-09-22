"""Regression tests for the consistent ``merge`` NLP tool (git_merge module).

Live sessions made ``git merge`` inconsistent in three concrete ways, each
reproduced against a real repo while building this tool:

* ``git merge --continue`` **hangs forever** when git opens an editor to
  confirm the merge commit message (no stdin, no timeout).
* ``--continue`` **rejects any argument** — ``git merge --continue --no-edit``
  exits 129 with ``fatal: --continue expects no arguments`` — so the obvious
  "make it non-interactive" fix is itself a dead end.
* starting a second merge while one is in progress wastes a call on
  ``fatal: Merging is not possible because you have unmerged files``, and
  ``--continue``/``--abort`` with no merge in progress return a bare
  ``fatal: There is no merge in progress (MERGE_HEAD missing)``.

The tool therefore owns the merge state machine: it detects an in-progress
merge, always runs non-interactively (``GIT_EDITOR`` no-op +
``GIT_MERGE_AUTOEDIT=no``), never passes arguments to ``--continue``, and
answers no-op requests with guidance instead of a wasted git call.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from agent_core.tools import git_merge

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git not installed"
)


# ---------------------------------------------------------------------------
# Real-repo helpers
# ---------------------------------------------------------------------------

def _git(cwd: Path, *args: str, env: dict[str, str] | None = None):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=e, timeout=60,
        stdin=subprocess.DEVNULL,
    )


def _init_repo(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    _git(d, "init", "-q", "-b", "main")
    _git(d, "config", "user.email", "tester@example.com")
    _git(d, "config", "user.name", "Tester")
    _git(d, "config", "commit.gpgsign", "false")
    return d


def _commit(d: Path, name: str, content: str) -> None:
    (d / name).write_text(content, encoding="utf-8")
    _git(d, "add", "-A")
    _git(d, "commit", "-q", "-m", content.strip())


@pytest.fixture()
def clean_repo(tmp_path: Path) -> Path:
    """Repo with one commit, no merge in progress."""
    d = _init_repo(tmp_path / "repo")
    _commit(d, "f.txt", "base\n")
    return d


@pytest.fixture()
def conflict_repo(tmp_path: Path) -> Path:
    """Repo whose ``feature`` branch conflicts with ``main`` on f.txt."""
    d = _init_repo(tmp_path / "repo")
    _commit(d, "f.txt", "base\n")
    _git(d, "checkout", "-q", "-b", "feature")
    _commit(d, "f.txt", "theirs\n")
    _git(d, "checkout", "-q", "main")
    _commit(d, "f.txt", "ours\n")
    return d


def _merge_head_present(d: Path) -> bool:
    return _git(d, "rev-parse", "-q", "--verify", "MERGE_HEAD").returncode == 0


# ---------------------------------------------------------------------------
# argv construction — the consistency contract
# ---------------------------------------------------------------------------

class TestBuildMergeArgv:
    def test_start_always_passes_no_edit(self) -> None:
        """No editor may ever open: every start is non-interactive."""
        assert git_merge.build_merge_argv("start", branch="feature") == [
            "git", "merge", "--no-edit", "feature",
        ]

    def test_continue_takes_no_arguments(self) -> None:
        """``--continue`` must be the LAST token (git exits 129 otherwise)."""
        argv = git_merge.build_merge_argv("continue")
        assert argv == ["git", "merge", "--continue"]
        assert len(argv) == 3

    def test_abort_and_quit_argv(self) -> None:
        assert git_merge.build_merge_argv("abort") == ["git", "merge", "--abort"]
        assert git_merge.build_merge_argv("quit") == ["git", "merge", "--quit"]

    def test_strategy_and_strategy_option_flags(self) -> None:
        argv = git_merge.build_merge_argv(
            "start", branch="feature", strategy="ort", strategy_option="theirs",
        )
        assert argv == [
            "git", "merge", "--no-edit", "-s", "ort", "-X", "theirs", "feature",
        ]

    def test_ff_flags(self) -> None:
        assert "--ff-only" in git_merge.build_merge_argv(
            "start", branch="b", ff="only",
        )
        assert "--no-ff" in git_merge.build_merge_argv(
            "start", branch="b", ff="no",
        )
        # "auto" (the default) adds nothing.
        assert "--ff-only" not in git_merge.build_merge_argv(
            "start", branch="b", ff="auto",
        )

    def test_message_and_squash_and_no_commit(self) -> None:
        argv = git_merge.build_merge_argv(
            "start", branch="b", message="merge b", squash=True,
            no_commit=True, allow_unrelated=True,
        )
        assert "-m" in argv and "merge b" in argv
        assert "--squash" in argv
        assert "--no-commit" in argv
        assert "--allow-unrelated-histories" in argv
        # branch stays last so it reads as the merge source.
        assert argv[-1] == "b"

    def test_status_does_not_run_git_merge(self) -> None:
        with pytest.raises(ValueError):
            git_merge.build_merge_argv("status")

    def test_unknown_action_raises(self) -> None:
        with pytest.raises(ValueError):
            git_merge.build_merge_argv("teleport")

    def test_flag_shaped_branch_is_refused(self) -> None:
        """A branch like ``--abort`` must never be read as a merge flag."""
        with pytest.raises(ValueError):
            git_merge.build_merge_argv("start", branch="--abort")

    def test_flag_shaped_strategy_option_is_refused(self) -> None:
        with pytest.raises(ValueError):
            git_merge.build_merge_argv(
                "start", branch="b", strategy_option="--squash",
            )


# ---------------------------------------------------------------------------
# Non-interactive environment
# ---------------------------------------------------------------------------

class TestMergeEnvironment:
    def test_environment_disables_every_editor(self) -> None:
        env = git_merge.merge_environment()
        # GIT_EDITOR must be a no-op command, never an interactive editor.
        assert env["GIT_EDITOR"] not in ("", "vi", "vim", "nano", "notepad")
        assert env["GIT_EDITOR"] == "true"
        assert env["GIT_MERGE_AUTOEDIT"] == "no"

    def test_environment_preserves_caller_variables(self) -> None:
        env = git_merge.merge_environment({"PATH": "/usr/bin"})
        assert env["PATH"] == "/usr/bin"
        assert env["GIT_EDITOR"] == "true"

    def test_environment_does_not_mutate_input(self) -> None:
        base = {"PATH": "/usr/bin"}
        git_merge.merge_environment(base)
        assert base == {"PATH": "/usr/bin"}


# ---------------------------------------------------------------------------
# Action normalisation
# ---------------------------------------------------------------------------

class TestNormalizeAction:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("", "status"),
            (None, "status"),
            ("status", "status"),
            ("state", "status"),
            ("start", "start"),
            ("merge", "start"),
            ("continue", "continue"),
            ("resume", "continue"),
            ("CONTINUE", "continue"),
            ("abort", "abort"),
            ("quit", "quit"),
        ],
    )
    def test_aliases(self, raw: object, expected: str) -> None:
        assert git_merge.normalize_action(raw) == expected

    def test_unknown_action_returns_none(self) -> None:
        assert git_merge.normalize_action("teleport") is None


# ---------------------------------------------------------------------------
# Conflict parsing
# ---------------------------------------------------------------------------

class TestConflictParsing:
    def test_parses_names_and_ignores_blank_lines(self) -> None:
        out = "src/a.py\n\n src/b.py \n"
        assert git_merge.parse_conflicted_files(out) == ["src/a.py", "src/b.py"]

    def test_empty_output_is_no_conflicts(self) -> None:
        assert git_merge.parse_conflicted_files("") == []
        assert git_merge.parse_conflicted_files("  \n") == []


# ---------------------------------------------------------------------------
# Real-repo behaviour: status
# ---------------------------------------------------------------------------

class TestStatusAction:
    def test_reports_no_merge_in_progress(self, clean_repo: Path) -> None:
        out = git_merge.run_merge(clean_repo, {"action": "status"})
        assert "Merge in progress: no" in out

    def test_reports_conflicts_while_in_progress(self, conflict_repo: Path) -> None:
        git_merge.run_merge(conflict_repo, {"action": "start", "branch": "feature"})
        out = git_merge.run_merge(conflict_repo, {"action": "status"})
        assert "Merge in progress: yes" in out
        assert "f.txt" in out

    def test_default_action_is_status(self, clean_repo: Path) -> None:
        out = git_merge.run_merge(clean_repo, {})
        assert "Merge in progress: no" in out


# ---------------------------------------------------------------------------
# Real-repo behaviour: start
# ---------------------------------------------------------------------------

class TestStartAction:
    def test_clean_merge_completes(self, tmp_path: Path) -> None:
        d = _init_repo(tmp_path / "r")
        _commit(d, "f.txt", "base\n")
        _git(d, "checkout", "-q", "-b", "feature")
        _commit(d, "g.txt", "g\n")
        _git(d, "checkout", "-q", "main")
        out = git_merge.run_merge(d, {"action": "start", "branch": "feature"})
        assert "error" not in out.lower()
        assert _merge_head_present(d) is False
        assert (d / "g.txt").exists()

    def test_conflict_reports_files_and_next_step(self, conflict_repo: Path) -> None:
        out = git_merge.run_merge(
            conflict_repo, {"action": "start", "branch": "feature"},
        )
        assert "f.txt" in out
        assert "conflict" in out.lower()
        # The model is told exactly how to proceed.
        assert "continue" in out.lower()
        assert _merge_head_present(conflict_repo) is True

    def test_start_without_branch_is_guidance_not_git_error(
        self, clean_repo: Path,
    ) -> None:
        out = git_merge.run_merge(clean_repo, {"action": "start"})
        assert "branch" in out.lower()
        assert "not something we can merge" not in out

    def test_start_while_in_progress_does_not_call_git(
        self, conflict_repo: Path,
    ) -> None:
        git_merge.run_merge(conflict_repo, {"action": "start", "branch": "feature"})
        out = git_merge.run_merge(
            conflict_repo, {"action": "start", "branch": "feature"},
        )
        assert "already in progress" in out.lower()
        assert "unmerged files" not in out
        # Still recoverable: the state was not disturbed.
        assert _merge_head_present(conflict_repo) is True

    def test_unknown_branch_reports_git_message(self, clean_repo: Path) -> None:
        out = git_merge.run_merge(
            clean_repo, {"action": "start", "branch": "nope"},
        )
        assert "not something we can merge" in out

    def test_typo_strategy_is_caught_before_git(self, clean_repo: Path) -> None:
        out = git_merge.run_merge(
            clean_repo, {"action": "start", "branch": "main", "strategy": "nope"},
        )
        assert "nope" in out
        # Helpful: lists what git actually accepts.
        assert "ort" in out
        assert "Could not find merge strategy" not in out

    def test_strategy_option_theirs_resolves_content(self, conflict_repo: Path) -> None:
        out = git_merge.run_merge(conflict_repo, {
            "action": "start", "branch": "feature", "strategy_option": "theirs",
        })
        assert _merge_head_present(conflict_repo) is False
        assert (conflict_repo / "f.txt").read_text(encoding="utf-8") == "theirs\n"
        assert "conflict" not in out.lower()

    def test_ff_only_refuses_diverged_history(self, conflict_repo: Path) -> None:
        out = git_merge.run_merge(
            conflict_repo, {"action": "start", "branch": "feature", "ff": "only"},
        )
        assert "fatal" in out.lower() or "not possible" in out.lower()
        assert _merge_head_present(conflict_repo) is False

    def test_squash_stages_without_committing(self, tmp_path: Path) -> None:
        d = _init_repo(tmp_path / "r")
        _commit(d, "f.txt", "base\n")
        _git(d, "checkout", "-q", "-b", "b3")
        _commit(d, "g.txt", "g\n")
        _git(d, "checkout", "-q", "main")
        out = git_merge.run_merge(
            d, {"action": "start", "branch": "b3", "squash": True},
        )
        assert _merge_head_present(d) is False
        status = _git(d, "status", "--porcelain").stdout
        assert "g.txt" in status
        assert "squash" in out.lower() or "staged" in out.lower()

    def test_allow_unrelated_histories(self, tmp_path: Path) -> None:
        d = _init_repo(tmp_path / "a")
        _commit(d, "f.txt", "base\n")
        other = _init_repo(tmp_path / "b")
        _commit(other, "z.txt", "z\n")
        _git(d, "remote", "add", "o", str(other))
        _git(d, "fetch", "-q", "o")
        refused = git_merge.run_merge(
            d, {"action": "start", "branch": "o/main"},
        )
        assert "unrelated histories" in refused.lower()
        out = git_merge.run_merge(d, {
            "action": "start", "branch": "o/main", "allow_unrelated": True,
        })
        # The flag did its job: the refusal is gone and z.txt was merged in.
        assert "unrelated histories" not in out.lower()
        assert "fatal" not in out.lower()
        assert (d / "z.txt").exists()
        assert _merge_head_present(d) is False


# ---------------------------------------------------------------------------
# Real-repo behaviour: continue / abort / quit  (the hang regressions)
# ---------------------------------------------------------------------------

class TestContinueAction:
    def test_continue_after_resolution_completes_without_hanging(
        self, conflict_repo: Path,
    ) -> None:
        """The headline bug: bare ``git merge --continue`` blocks on the editor."""
        git_merge.run_merge(conflict_repo, {"action": "start", "branch": "feature"})
        (conflict_repo / "f.txt").write_text("resolved\n", encoding="utf-8")
        _git(conflict_repo, "add", "-A")
        out = git_merge.run_merge(
            conflict_repo, {"action": "continue"}, timeout=30,
        )
        assert "timed out" not in out.lower()
        assert _merge_head_present(conflict_repo) is False
        subject = _git(conflict_repo, "log", "-1", "--pretty=%s").stdout
        assert "Merge" in subject
        assert (conflict_repo / "f.txt").read_text(encoding="utf-8") == "resolved\n"

    def test_continue_with_unresolved_conflicts_is_refused(
        self, conflict_repo: Path,
    ) -> None:
        git_merge.run_merge(conflict_repo, {"action": "start", "branch": "feature"})
        out = git_merge.run_merge(conflict_repo, {"action": "continue"})
        assert "f.txt" in out
        assert "unresolved" in out.lower() or "conflict" in out.lower()
        # The merge is still pending, nothing was committed behind our back.
        assert _merge_head_present(conflict_repo) is True

    def test_continue_without_merge_in_progress_is_friendly(
        self, clean_repo: Path,
    ) -> None:
        out = git_merge.run_merge(clean_repo, {"action": "continue"})
        assert "no merge in progress" in out.lower()
        assert "fatal" not in out.lower()
        assert "expects no arguments" not in out

    def test_continue_ignores_stray_arguments(self, conflict_repo: Path) -> None:
        """A model-supplied args/branch must not reach ``--continue``."""
        git_merge.run_merge(conflict_repo, {"action": "start", "branch": "feature"})
        (conflict_repo / "f.txt").write_text("resolved\n", encoding="utf-8")
        _git(conflict_repo, "add", "-A")
        out = git_merge.run_merge(conflict_repo, {
            "action": "continue", "branch": "feature", "args": "--no-edit",
        }, timeout=30)
        assert "expects no arguments" not in out
        assert _merge_head_present(conflict_repo) is False


class TestAbortAndQuit:
    def test_abort_restores_clean_tree(self, conflict_repo: Path) -> None:
        git_merge.run_merge(conflict_repo, {"action": "start", "branch": "feature"})
        out = git_merge.run_merge(conflict_repo, {"action": "abort"})
        assert "fatal" not in out.lower()
        assert _merge_head_present(conflict_repo) is False
        assert _git(conflict_repo, "status", "--porcelain").stdout.strip() == ""
        assert (conflict_repo / "f.txt").read_text(encoding="utf-8") == "ours\n"

    def test_abort_without_merge_in_progress_is_friendly(
        self, clean_repo: Path,
    ) -> None:
        out = git_merge.run_merge(clean_repo, {"action": "abort"})
        assert "no merge in progress" in out.lower()
        assert "fatal" not in out.lower()

    def test_quit_clears_state_without_restoring(
        self, conflict_repo: Path,
    ) -> None:
        git_merge.run_merge(conflict_repo, {"action": "start", "branch": "feature"})
        out = git_merge.run_merge(conflict_repo, {"action": "quit"})
        assert _merge_head_present(conflict_repo) is False
        assert "fatal" not in out.lower()


# ---------------------------------------------------------------------------
# Output hygiene + validation
# ---------------------------------------------------------------------------

class TestOutputAndValidation:
    def test_unknown_action_lists_valid_actions(self, clean_repo: Path) -> None:
        out = git_merge.run_merge(clean_repo, {"action": "teleport"})
        assert "teleport" in out
        for name in ("status", "start", "continue", "abort", "quit"):
            assert name in out

    def test_output_is_bounded(self, clean_repo: Path) -> None:
        out = git_merge.run_merge(
            clean_repo, {"action": "start", "branch": "x" * 4000},
        )
        assert len(out) <= 6000

    def test_invocation_is_echoed_for_audit(self, conflict_repo: Path) -> None:
        out = git_merge.run_merge(
            conflict_repo, {"action": "start", "branch": "feature"},
        )
        assert "git merge" in out


# ---------------------------------------------------------------------------
# Wiring: schema, dispatch, plan mode
# ---------------------------------------------------------------------------

class TestMergeToolWiring:
    def _schema(self) -> dict:
        from agent_core.tool_schemas import NLP_TOOL_SCHEMAS

        return next(
            s for s in NLP_TOOL_SCHEMAS if s["function"]["name"] == "merge"
        )

    def test_merge_is_advertised(self) -> None:
        from agent_core.tool_schemas import NLP_TOOL_NAMES

        assert "merge" in NLP_TOOL_NAMES

    def test_schema_teaches_continue_without_arguments(self) -> None:
        schema = self._schema()
        blob = (
            schema["function"]["description"]
            + str(schema["function"]["parameters"])
        ).lower()
        assert "continue" in blob
        assert "status" in blob
        # The two mistakes that made merges inconsistent are called out.
        assert "no arguments" in blob or "without arguments" in blob
        assert "branch" in blob

    def test_dispatch_table_has_merge(self) -> None:
        from agent import Agent

        handlers = Agent.__new__(Agent)._nlp_tool_handlers()
        assert "merge" in handlers

    def test_plan_mode_blocks_merge(self) -> None:
        from agent_core.modes import MODE_PLAN, check_tool_allowed

        assert check_tool_allowed("merge", MODE_PLAN) is not None

    def test_handler_routes_to_real_repo(self, clean_repo: Path) -> None:
        import asyncio

        from agent import Agent

        agent = Agent(workspace=str(clean_repo))
        out = asyncio.run(agent._nlp_merge({"action": "status"}))
        assert "Merge in progress: no" in out

    def test_handler_reports_conflicts_end_to_end(self, conflict_repo: Path) -> None:
        import asyncio

        from agent import Agent

        agent = Agent(workspace=str(conflict_repo))
        out = asyncio.run(
            agent._nlp_merge({"action": "start", "branch": "feature"})
        )
        assert "f.txt" in out

    def test_handler_is_reachable_via_execute_tool_call(
        self, conflict_repo: Path,
    ) -> None:
        """The dispatcher, not just the method, must route 'merge'."""
        import asyncio

        from agent import Agent

        agent = Agent(workspace=str(conflict_repo))
        out = asyncio.run(agent._execute_tool_call(
            "merge", {"action": "start", "branch": "feature"},
        ))
        assert "f.txt" in out
        assert _merge_head_present(conflict_repo) is True
