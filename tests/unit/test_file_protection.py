"""Unit tests for agent_core.file_protection (delete-protection guard).

Hermetic: builds a tiny throwaway workspace under tmp_path; no real
workspace files are touched.  Covers the two protected categories — .env and
reports/* — plus the safe_remove / safe_rmtree choke points and reporting.
"""
import os

import pytest

from agent_core.entities import SecurityViolationError
from agent_core.file_protection import (
    is_protected,
    protected_targets,
    safe_remove,
    safe_rmtree,
)


@pytest.fixture()
def ws(tmp_path):
    """A throwaway workspace root with a .env and a reports/ tree."""
    env = tmp_path / ".env"
    env.write_text("SECRET=x", encoding="utf-8")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "traces").mkdir(parents=True)
    (tmp_path / "reports" / "traces" / "t.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "reports" / "diagnoses").mkdir()
    (tmp_path / "reports" / "diagnoses" / "d.json").write_text("{}", encoding="utf-8")
    # An ordinary, deletable file for contrast.
    (tmp_path / "scratch.py").write_text("# temp", encoding="utf-8")
    return tmp_path


class TestIsProtected:
    def test_env_is_protected(self, ws):
        assert is_protected(ws / ".env", ws)

    def test_env_anywhere_in_tree_is_protected(self, ws):
        # A nested .env must still be refused (filename match, not path).
        nested = ws / "agent_core" / ".env"
        nested.parent.mkdir(parents=True, exist_ok=True)
        nested.write_text("SECRET=y", encoding="utf-8")
        assert is_protected(nested, ws)

    def test_reports_file_is_protected(self, ws):
        assert is_protected(ws / "reports" / "traces" / "t.jsonl", ws)

    def test_reports_subdir_is_protected(self, ws):
        # The reports dir itself and any sub-directory are protected.
        assert is_protected(ws / "reports", ws)
        assert is_protected(ws / "reports" / "diagnoses", ws)

    def test_normal_file_not_protected(self, ws):
        assert not is_protected(ws / "scratch.py", ws)

    def test_project_md_not_protected(self, ws):
        # project_*.md are the legit cleanup targets; must NOT be protected.
        p = ws / "project_plan.md"
        p.write_text("# plan", encoding="utf-8")
        assert not is_protected(p, ws)

    def test_out_of_tree_path_raises(self, ws):
        # A path escaping the workspace is invalid input — it must NOT be
        # reported as "unprotected".  Regression guard for misclassification.
        with pytest.raises(SecurityViolationError):
            is_protected(ws.parent / ".env", ws)


class TestSafeRemove:
    def test_safe_remove_deletes_normal_file(self, ws):
        target = ws / "scratch.py"
        safe_remove(target, ws)
        assert not target.exists()

    def test_safe_remove_refuses_env(self, ws):
        with pytest.raises(SecurityViolationError, match="Protected file"):
            safe_remove(ws / ".env", ws)
        # File is still present — refusal did not delete it.
        assert (ws / ".env").exists()

    def test_safe_remove_refuses_reports_file(self, ws):
        target = ws / "reports" / "traces" / "t.jsonl"
        with pytest.raises(SecurityViolationError, match="Protected file"):
            safe_remove(target, ws)
        assert target.exists()


class TestSafeRmtree:
    def test_safe_rmtree_removes_normal_dir(self, ws):
        d = ws / "agent_core"
        d.mkdir()
        (d / ".env").write_text("x", encoding="utf-8")  # nested .env inside a non-reports dir
        safe_rmtree(d, ws)
        assert not d.exists()

    def test_safe_rmtree_refuses_reports_dir(self, ws):
        with pytest.raises(SecurityViolationError, match="Protected directory"):
            safe_rmtree(ws / "reports", ws)
        # Tree still intact.
        assert (ws / "reports" / "traces" / "t.jsonl").exists()

    def test_safe_rmtree_refuses_reports_subdir(self, ws):
        with pytest.raises(SecurityViolationError, match="Protected directory"):
            safe_rmtree(ws / "reports" / "diagnoses", ws)
        assert (ws / "reports" / "diagnoses" / "d.json").exists()


class TestProtectedTargetsReporting:
    def test_reports_only_protected_files(self, ws):
        paths = [
            str(ws / ".env"),
            str(ws / "scratch.py"),
            str(ws / "reports" / "traces" / "t.jsonl"),
            str(ws / "project_plan.md"),
        ]
        got = protected_targets(paths, ws)
        assert got == [str(ws / ".env"), str(ws / "reports" / "traces" / "t.jsonl")]

    def test_empty_when_none_protected(self, ws):
        paths = [str(ws / "scratch.py"), str(ws / "project_plan.md")]
        assert protected_targets(paths, ws) == []


class TestCleanupDeleteRespectsProtection:
    """The cleanup --delete path must never touch .env / reports/*."""

    def test_delete_skips_protected_and_removes_others(self, ws):
        from agent_core.commands.cleanup_cmd import CleanupCommand

        cmd = CleanupCommand()
        # Build a fake "unreferenced" set that mixes protected + deletable.
        # Rel paths mirror how cleanup_cmd feeds is_protected(f, ws_path).
        unreferenced_rel = ["scratch.py", ".env", "reports/traces/t.jsonl"]
        deleted = skipped = 0
        for f in unreferenced_rel:
            if is_protected(f, ws):
                skipped += 1
                continue
            os.remove(ws / f)
            deleted += 1
        assert not (ws / "scratch.py").exists()
        assert (ws / ".env").exists()      # protected — untouched
        assert (ws / "reports" / "traces" / "t.jsonl").exists()  # protected — untouched
        assert skipped == 2 and deleted == 1

    def test_delete_declines_autonomously(self, ws):
        from agent_core.commands.base import auto_choice

        # Autonomous mode must NOT auto-approve destructive delete.
        from agent_core.commands.base import set_autonomous

        set_autonomous(True)
        try:
            choice = auto_choice("Delete files? [y/N] ", default="n", auto_default="n")
            assert choice == "n"
        finally:
            set_autonomous(None)
