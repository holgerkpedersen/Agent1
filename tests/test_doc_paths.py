"""Tests for .docs/<timestamp>/ run folders (workflow doc location).

Regression coverage: workflow docs used to be written to the workspace root
(project_*.md), polluting the repo.  They now live in git-ignored
.docs/<YYYY-MM-DD_HH-MM-SS>/ folders — one per run — and readers must find
the newest run first, with the root kept as a legacy fallback.
"""
from datetime import datetime
from pathlib import Path

from agent_core.commands.doc_paths import (
    DOCS_DIR_NAME,
    docs_root,
    find_doc,
    find_input,
    latest_run_dir,
    new_run_dir,
    resolve_output,
    run_stamp,
)


class TestRunFolders:
    def test_new_run_dir_is_created_under_docs(self, tmp_path):
        run = new_run_dir(tmp_path)
        assert run.is_dir()
        assert run.parent.name == DOCS_DIR_NAME
        assert run.parent.parent == tmp_path

    def test_run_stamp_format_is_sortable(self):
        stamp = run_stamp(datetime(2026, 8, 15, 11, 17, 11))
        assert stamp == "2026-08-15_11-17-11"

    def test_two_runs_same_second_get_distinct_folders(self, tmp_path):
        now = datetime(2026, 8, 15, 11, 17, 11)
        first = new_run_dir(tmp_path, now=now)
        second = new_run_dir(tmp_path, now=now)
        assert first != second
        assert second.is_dir()

    def test_latest_run_dir_orders_chronologically(self, tmp_path):
        new_run_dir(tmp_path, now=datetime(2026, 8, 15, 9, 0, 0))
        latest = new_run_dir(tmp_path, now=datetime(2026, 8, 15, 11, 17, 11))
        assert latest_run_dir(tmp_path) == latest

    def test_latest_run_dir_none_without_docs(self, tmp_path):
        assert latest_run_dir(tmp_path) is None

    def test_docs_root_created_if_missing(self, tmp_path):
        root = docs_root(tmp_path)
        assert root.is_dir()
        assert root == tmp_path / DOCS_DIR_NAME


class TestFindDoc:
    def test_newest_run_wins_over_root(self, tmp_path):
        root_doc = tmp_path / "project_tasks.md"
        root_doc.write_text("root", encoding="utf-8")
        run = new_run_dir(tmp_path, now=datetime(2026, 8, 15, 11, 0, 0))
        run_doc = run / "project_tasks.md"
        run_doc.write_text("run", encoding="utf-8")

        found = find_doc(tmp_path, "project_tasks.md")
        assert found == str(run_doc)

    def test_root_fallback_for_legacy_files(self, tmp_path):
        legacy = tmp_path / "project_tasks.md"
        legacy.write_text("legacy", encoding="utf-8")
        # An older run folder without the doc must not shadow the root copy.
        run = new_run_dir(tmp_path, now=datetime(2026, 1, 1, 0, 0, 0))
        (run / "project_plan.md").write_text("plan", encoding="utf-8")

        found = find_doc(tmp_path, "project_tasks.md")
        assert found == str(legacy)

    def test_missing_doc_returns_none(self, tmp_path):
        new_run_dir(tmp_path)
        assert find_doc(tmp_path, "project_tasks.md") is None


class TestFindInput:
    def test_existing_path_unchanged(self, tmp_path):
        p = tmp_path / "custom.md"
        p.write_text("x", encoding="utf-8")
        assert find_input(tmp_path, str(p)) == str(p)

    def test_missing_name_resolves_to_newest_run(self, tmp_path):
        run = new_run_dir(tmp_path, now=datetime(2026, 8, 15, 11, 0, 0))
        (run / "project_analysis.md").write_text("a", encoding="utf-8")
        assert find_input(tmp_path, "project_analysis.md") == str(
            run / "project_analysis.md"
        )

    def test_missing_name_returns_resolved_absolute(self, tmp_path):
        # Contract: find_input always returns an absolute path — a missing
        # file resolves against the workspace so the caller's own
        # "file not found" error still fires with a usable path.
        assert find_input(tmp_path, "nope.md") == str(tmp_path / "nope.md")
        assert Path(find_input(tmp_path, "nope.md")).is_absolute()


class TestResolveOutput:
    def test_bare_name_goes_to_fresh_run_folder(self, tmp_path):
        out = Path(resolve_output(tmp_path, "entities.md"))
        assert out.name == "entities.md"
        assert out.parent.parent.name == DOCS_DIR_NAME

    def test_bare_name_joins_sibling_run_folder(self, tmp_path):
        run = new_run_dir(tmp_path, now=datetime(2026, 8, 15, 11, 0, 0))
        sibling = run / "project_analysis.md"
        sibling.write_text("a", encoding="utf-8")
        out = resolve_output(tmp_path, "entities.md", sibling_of=str(sibling))
        assert out == str(run / "entities.md")

    def test_explicit_relative_path_respected(self, tmp_path):
        out = resolve_output(tmp_path, "sub/entities.md")
        assert out == "sub/entities.md"

    def test_explicit_absolute_path_respected(self, tmp_path):
        target = tmp_path / "elsewhere.md"
        assert resolve_output(tmp_path, str(target)) == str(target)
