"""Regression: vacuous commit subjects must not reach master.

Live run (2026-09-28): HEAD was ``9db6ca2 "commit changes"`` -- a subject with
zero information, produced because the model was told "commit changes" and
handed that string straight to ``git commit -m``.  ``CONTRIBUTING.md`` has
documented the real format (type-prefixed, imperative) for months and nothing
enforced it, so the convention silently rotted: 150 of 612 subjects in history
carry no type prefix at all.

``agent_core/commit_policy.py`` is the single source of truth for "is this
subject worth putting in history"; ``.githooks/commit-msg`` is the enforcement
point.  Vacuous subjects are errors (they BLOCK a commit); format deviations
are warnings (advisory, they never lock the author out of committing).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent_core.commit_policy import (
    ALLOWED_TYPES,
    check_message,
    is_placeholder_subject,
    validate_subject,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestPlaceholderDetection:
    """The exact live subject, and its neighbours, must be vacuous."""

    @pytest.mark.parametrize(
        "subject",
        [
            "commit changes",
            "commit",
            "changes",
            "wip",
            "WIP",
            "misc",
            "stuff",
            "temp",
            "tmp",
            "untitled",
            "asdf",
            "ok",
            "done",
            "final",
            "final version",
            "minor changes",
            "various fixes",
            "improvements",
            "bug fix",
            "update",
            "fix",
            "agent.py",
            "agent_core/jev_engine.py",
            "  commit changes  ",
            "commit changes.",
            "COMMIT CHANGES",
        ],
    )
    def test_vacuous_subject_is_a_placeholder(self, subject: str) -> None:
        assert is_placeholder_subject(subject) is True, subject

    @pytest.mark.parametrize(
        "subject",
        [
            "fix(agent): reject unverified repo-state claims in the NLP loop",
            "feat(prompt): guard gpt-oss/harmony and system prompt",
            "docs: document the analyze directory mode",
            "Update agent.py after REPL reload",
            "fix(jev): correct the module docstring punctuation",
            "revert: undo the routing change",
        ],
    )
    def test_meaningful_subject_is_not_a_placeholder(self, subject: str) -> None:
        assert is_placeholder_subject(subject) is False, subject


class TestValidateSubject:
    def test_empty_subject_is_an_error(self) -> None:
        result = validate_subject("")
        assert result.errors
        assert "empty" in result.errors[0].lower()

    def test_whitespace_only_subject_is_an_error(self) -> None:
        assert validate_subject("   \t  ").errors

    def test_placeholder_is_an_error(self) -> None:
        errors = validate_subject("commit changes").errors
        assert errors
        assert any("vacuous" in e.lower() for e in errors)

    def test_absurdly_long_subject_is_an_error(self) -> None:
        result = validate_subject("fix: " + "x" * 200)
        assert any("long" in e.lower() for e in result.errors)

    def test_missing_type_prefix_is_a_warning_not_an_error(self) -> None:
        result = validate_subject("Update agent.py after REPL reload")
        assert result.errors == []
        assert any("type" in w.lower() for w in result.warnings)

    def test_unknown_type_is_a_warning(self) -> None:
        result = validate_subject("wibble: do a thing")
        assert result.errors == []
        assert any("wibble" in w for w in result.warnings)

    def test_trailing_period_is_a_warning(self) -> None:
        result = validate_subject("fix: correct the docstring punctuation.")
        assert result.errors == []
        assert any("period" in w.lower() for w in result.warnings)

    def test_subject_over_72_is_a_warning(self) -> None:
        result = validate_subject("fix: " + "y" * 80)
        assert result.errors == []
        assert any("72" in w for w in result.warnings)

    def test_lowercase_type_is_enforced_by_convention(self) -> None:
        result = validate_subject("Fix: capitalised type prefix")
        assert result.errors == []
        assert any("lowercase" in w.lower() for w in result.warnings)

    def test_allowlist_matches_contributing(self) -> None:
        # CONTRIBUTING.md: feat, fix, refactor, docs, test, chore.
        assert ALLOWED_TYPES == frozenset(
            {"feat", "fix", "refactor", "docs", "test", "chore"}
        )


class TestCheckMessage:
    def test_clean_conventional_message_passes_silently(self) -> None:
        report = check_message(
            "fix(commit): block vacuous commit subjects\n"
            "\n"
            "The hook now refuses 'commit changes' so the convention cannot rot.\n"
            "\n"
            "Files: agent_core/commit_policy.py, .githooks/commit-msg.\n"
        )
        assert report.errors == []
        assert report.warnings == []

    def test_multiline_body_does_not_trigger_a_warning(self) -> None:
        report = check_message(
            "chore: bump the ruff pin\n"
            "\n"
            "Body line one.\n"
            "Body line two.\n"
        )
        assert report.warnings == []

    def test_leading_blank_line_uses_the_real_subject(self) -> None:
        # `git commit` with a cleanup strip can leave a leading blank line;
        # the first NON-blank line is the subject git will show.
        report = check_message("\n\nfix: real subject here\n\nbody\n")
        assert report.errors == []
        assert report.warnings == []

    def test_crlf_message_file_does_not_leak_carriage_returns(self) -> None:
        report = check_message("fix: windows line endings\r\n\r\nbody\r\n")
        assert report.errors == []
        assert "\r" not in " ".join(report.warnings)

    def test_report_renders_a_readable_block(self) -> None:
        report = check_message("commit changes")
        text = report.render()
        assert "commit changes" in text
        assert "commit-msg" in text

    def test_clean_report_renders_nothing(self) -> None:
        assert check_message("fix: a good subject\n\nbody\n").render() == ""


class TestCommitMsgHookEndToEnd:
    """The hook itself, run as git runs it, on a real throwaway repo."""

    @staticmethod
    def _repo(tmp_path: Path) -> Path:
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        subprocess.run(
            ["git", "-C", str(tmp_path), "config", "user.email", "t@t"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True
        )
        # git looks for hooks in .git/hooks/ -- NOT the worktree root.  (An
        # earlier version of this test installed the hook at the repo root,
        # so the hook never ran and "commit changes" sailed straight through.)
        hook = tmp_path / ".git" / "hooks" / "commit-msg"
        hook.write_text(
            (REPO_ROOT / ".githooks" / "commit-msg").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        hook.chmod(0o755)
        # The hook resolves the policy relative to itself; from .git/hooks/
        # that means climbing back out to the worktree root.
        (tmp_path / "agent_core").mkdir(exist_ok=True)
        (tmp_path / "agent_core" / "commit_policy.py").write_text(
            (REPO_ROOT / "agent_core" / "commit_policy.py").read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )
        return tmp_path

    @staticmethod
    def _commit(tmp_path: Path, message: str) -> subprocess.CompletedProcess[str]:
        # git refuses to commit with nothing staged, so stage first -- the
        # hook only runs once there is a real commit to make.
        subprocess.run(
            ["git", "-C", str(tmp_path), "add", "a.txt"],
            capture_output=True, timeout=60, check=True,
        )
        return subprocess.run(
            ["git", "-C", str(tmp_path), "commit", "-m", message],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )

    def test_hook_rejects_commit_changes(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        (repo / "a.txt").write_text("a\n", encoding="utf-8")
        proc = self._commit(repo, "commit changes")
        assert proc.returncode != 0, proc.stdout
        assert "vacuous" in (proc.stdout + proc.stderr).lower()

    def test_hook_allows_a_conventional_message(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        (repo / "a.txt").write_text("a\n", encoding="utf-8")
        proc = self._commit(repo, "fix(commit): block vacuous subjects")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        subject = subprocess.run(
            ["git", "-C", str(repo), "log", "-1", "--format=%s"],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip()
        assert subject == "fix(commit): block vacuous subjects"

    def test_hook_keeps_warnings_advisory(self, tmp_path: Path) -> None:
        """A format deviation warns but must NOT block the commit."""
        repo = self._repo(tmp_path)
        (repo / "a.txt").write_text("a\n", encoding="utf-8")
        proc = self._commit(repo, "Update the analysis module")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        out = proc.stdout + proc.stderr
        assert "type" in out.lower()

    def test_hook_does_not_import_the_agent_stack(self, tmp_path: Path) -> None:
        """The hook must stay runnable in a bare shell checkout (no deps)."""
        import ast

        hook = REPO_ROOT / ".githooks" / "commit-msg"
        text = hook.read_text(encoding="utf-8")
        assert "import agent" not in text
        assert "import pytest" not in text
        # Parses standalone, and imports only the stdlib it needs to run.
        tree = ast.parse(text)
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in (
                node.names
                if isinstance(node, ast.Import)
                else [ast.alias(name=node.module or "")]
            )
        }
        assert imported <= {
            "__future__", "importlib", "os", "pathlib", "sys",
        }, imported
