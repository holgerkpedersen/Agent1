"""Tests for canonical workspace-relative path handling.

Relative paths are interpreted against the WORKSPACE (never the process
CWD); the absolute form is always derivable via join(workspace, rel).
"""
from pathlib import Path
from agent_core.commands.doc_paths import find_input
from agent_core.decisions import (
    _canonical_rel,
    add_decision,
    find_decisions,
    find_overlaps,
    load_decisions,
    normalize_affected_files,
)


class TestCanonicalRel:
    def test_relative_resolves_against_workspace(self, tmp_path):
        assert _canonical_rel(str(tmp_path), "agent_core/security/allowlist.py") == (
            "agent_core/security/allowlist.py"
        )

    def test_absolute_under_workspace_relativized(self, tmp_path):
        abs_path = str(tmp_path / "agent_core" / "tool_schemas.py")
        assert _canonical_rel(str(tmp_path), abs_path) == "agent_core/tool_schemas.py"

    def test_escape_outside_workspace_rejected(self, tmp_path):
        assert _canonical_rel(str(tmp_path), "../ReactAgent/agent.py") is None

    def test_backslashes_normalized(self, tmp_path):
        assert _canonical_rel(str(tmp_path), r"agent_core\security\allowlist.py") == (
            "agent_core/security/allowlist.py"
        )


class TestNormalizeAffectedFiles:
    def test_keeps_existing_and_drops_missing(self, tmp_path):
        (tmp_path / "agent.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "agent_core").mkdir()
        (tmp_path / "agent_core" / "tool_schemas.py").write_text("X = 1\n", encoding="utf-8")
        normalized = normalize_affected_files(str(tmp_path), [
            "agent.py",
            "agent_core/tool_schemas.py",
            "missing_module.py",
            "../ReactAgent",
        ])
        assert normalized == ["agent.py", "agent_core/tool_schemas.py"]

    def test_doc_basename_falls_back_to_newest_run(self, tmp_path):
        run = tmp_path / ".docs" / "2026-08-15_12-06-06"
        run.mkdir(parents=True)
        (run / "project_analysis.md").write_text("analysis", encoding="utf-8")
        normalized = normalize_affected_files(str(tmp_path), ["project_analysis.md"])
        assert normalized == [".docs/2026-08-15_12-06-06/project_analysis.md"]

    def test_deduplicates_and_normalizes_absolute(self, tmp_path):
        (tmp_path / "agent.py").write_text("x = 1\n", encoding="utf-8")
        normalized = normalize_affected_files(str(tmp_path), [
            "agent.py",
            str(tmp_path / "agent.py"),
            "agent.py",
        ])
        assert normalized == ["agent.py"]


class TestFindInputContract:
    def test_relative_existing_resolved_against_workspace(self, tmp_path, monkeypatch):
        """A relative path must resolve against the WORKSPACE, not the CWD."""
        (tmp_path / "project_tasks.md").write_text("tasks", encoding="utf-8")
        monkeypatch.chdir(tmp_path.parent)  # CWD differs from the workspace
        result = find_input(str(tmp_path), "project_tasks.md")
        assert result == str(tmp_path / "project_tasks.md")
        assert Path(result).is_absolute()

    def test_missing_relative_falls_back_to_newest_run(self, tmp_path):
        run = tmp_path / ".docs" / "2026-08-15_11-00-00"
        run.mkdir(parents=True)
        (run / "project_tasks.md").write_text("tasks", encoding="utf-8")
        result = find_input(str(tmp_path), "project_tasks.md")
        assert result == str(run / "project_tasks.md")

    def test_missing_everywhere_returns_resolved_absolute(self, tmp_path):
        result = find_input(str(tmp_path), "nope.md")
        assert result == str(tmp_path / "nope.md")
        assert Path(result).is_absolute()


class TestDecisionMatching:
    def test_add_and_find_match_on_canonical_form(self, tmp_path):
        (tmp_path / "agent.py").write_text("x = 1\n", encoding="utf-8")
        add_decision(
            str(tmp_path),
            "Test decision",
            decision="Use X",
            affected_files=["agent.py"],
        )
        decisions = load_decisions(str(tmp_path))
        assert decisions[0]["affected_files"] == ["agent.py"]

        # Workspace-relative caller matches...
        assert find_decisions(str(tmp_path), files=["agent.py"])
        # ...and so does an absolute caller path.
        assert find_decisions(str(tmp_path), files=[str(tmp_path / "agent.py")])
        # Unrelated files do not match.
        assert not find_decisions(str(tmp_path), files=["agent_core/other.py"])

    def test_find_overlaps_normalizes_new_side(self, tmp_path):
        (tmp_path / "agent.py").write_text("x = 1\n", encoding="utf-8")
        add_decision(
            str(tmp_path), "Overlap decision", decision="A",
            affected_files=["agent.py"],
        )
        existing = load_decisions(str(tmp_path))
        overlaps = find_overlaps(
            {"tags": [], "affected_files": [str(tmp_path / "agent.py")]},
            existing,
            str(tmp_path),
        )
        assert len(overlaps) == 1
