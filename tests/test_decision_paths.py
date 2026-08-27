"""Tests for canonical workspace-relative path handling.

Relative paths are interpreted against the WORKSPACE (never the process
CWD); the absolute form is always derivable via join(workspace, rel).
"""
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

from agent_core.commands.doc_paths import find_input
from agent_core.decisions import (
    _canonical_rel,
    add_decision,
    annotate_candidates,
    decisions_as_system_prompt,
    extract_from_analysis,
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

    def test_load_decisions_normalizes_null_tags(self, tmp_path):
        """Stored records with tags=null must load as [] (regression for the
        TypeError at decide_cmd.py:143, ``None[:5]``)."""
        store = tmp_path / ".decisions.json"
        store.write_text(
            json.dumps([
                {"id": "001", "title": "legacy", "tags": None, "affected_files": []},
                {"id": "002", "title": "legacy-obj", "tags": "not-a-list", "affected_files": []},
            ]),
            encoding="utf-8",
        )
        decisions = load_decisions(str(tmp_path))
        assert decisions[0]["tags"] == []
        assert decisions[1]["tags"] == []
        # And the exact consuming expression from decide_cmd list must not crash.
        for d in decisions:
            assert ", ".join((d.get("tags") or [])[:5]) == ""

    def test_load_decisions_unifies_date_from_created_at(self, tmp_path):
        """Records that store the timestamp under ``created_at`` (instead of
        ``date``) must still load with a usable ``date`` — regression for the
        KeyError at decide_cmd.py:145, ``d['date'][:10]``."""
        store = tmp_path / ".decisions.json"
        store.write_text(
            json.dumps([
                {"id": "078", "title": "renamed-date",
                 "created_at": "2026-08-24T10:43:00.306892+00:00",
                 "tags": [], "affected_files": []},
                {"id": "079", "title": "no-date-at-all",
                 "tags": [], "affected_files": []},
            ]),
            encoding="utf-8",
        )
        decisions = load_decisions(str(tmp_path))
        by_id = {d["id"]: d for d in decisions}
        assert by_id["078"]["date"] == "2026-08-24T10:43:00.306892+00:00"
        assert by_id["079"]["date"] == ""
        # The exact consuming expression from decide_cmd list must not crash.
        for d in decisions:
            assert (d.get("date") or "")[:10] is not None


class TestAnnotateCandidates:
    """Mechanical fact-check of extracted candidates before recording."""

    def test_negative_coverage_claim_contradicted(self, tmp_path):
        (tmp_path / "benchmark.py").write_text("X = 1\n", encoding="utf-8")
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test_benchmark_scoring.py").write_text("import benchmark\n", encoding="utf-8")
        candidate = [{
            "title": "Add tests for benchmark.py",
            "context": "No test coverage exists for benchmark.py.",
            "decision": "Write tests.",
            "affected_files": [],
        }]
        annotate_candidates(candidate, str(tmp_path))
        warnings = candidate[0].get("warnings", [])
        assert any("benchmark.py" in w and "test_benchmark_scoring.py" in w for w in warnings)

    def test_negative_claim_without_file_ref_warns(self, tmp_path):
        candidate = [{
            "title": "Cover the untested core",
            "context": "The core module is completely untested.",
            "decision": "Add tests.",
            "affected_files": [],
        }]
        annotate_candidates(candidate, str(tmp_path))
        assert any("without a file reference" in w for w in candidate[0]["warnings"])

    def test_nonexistent_affected_file_warns(self, tmp_path):
        candidate = [{
            "title": "Fix phantoms",
            "context": "Refactor.",
            "decision": "Do it.",
            "affected_files": ["missing_module.py"],
        }]
        annotate_candidates(candidate, str(tmp_path))
        assert any("missing_module.py" in w for w in candidate[0]["warnings"])

    def test_clean_candidate_gets_no_warnings_key(self, tmp_path):
        (tmp_path / "real.py").write_text("X = 1\n", encoding="utf-8")
        candidate = [{
            "title": "Refactor real.py",
            "context": "Improve clarity.",
            "decision": "Refactor.",
            "affected_files": ["real.py"],
        }]
        annotate_candidates(candidate, str(tmp_path))
        assert "warnings" not in candidate[0]

    def test_repetition_of_unverified_claim_token(self, tmp_path):
        report = "- [UNVERIFIED] `phantom_symbol` — symbol not found anywhere in workspace"
        candidate = [{
            "title": "Track phantom_symbol usage",
            "context": "phantom_symbol is mutated without locks.",
            "decision": "Add locking.",
            "affected_files": [],
        }]
        annotate_candidates(candidate, str(tmp_path), verification_report=report)
        assert any("phantom_symbol" in w for w in candidate[0]["warnings"])

    def test_warnings_deduplicated(self, tmp_path):
        report = "- [UNVERIFIED] `dup_token` — not found"
        candidate = [{
            "title": "dup_token issue",
            "context": "dup_token repeated #1",
            "decision": "Decision mentions dup_token again.",
            "affected_files": [],
        }]
        annotate_candidates(candidate, str(tmp_path), verification_report=report)
        assert len(candidate[0]["warnings"]) == 1


class TestMetaWarnings:
    """Warned candidates persist their unverified basis on the record."""

    def test_add_decision_stores_meta_warnings(self, tmp_path):
        record = add_decision(
            str(tmp_path),
            "Hypothetical decision",
            decision="Do X",
            warnings=["Contradicted by workspace: tests found for benchmark.py"],
        )
        assert record["meta_warnings"] == [
            "Contradicted by workspace: tests found for benchmark.py"
        ]
        stored = load_decisions(str(tmp_path))
        assert stored[0]["meta_warnings"] == record["meta_warnings"]

    def test_decisions_as_system_prompt_marks_warned(self, tmp_path):
        add_decision(str(tmp_path), "Warned decision", decision="X",
                     warnings=["unverifiable"])
        add_decision(str(tmp_path), "Clean decision", decision="Y")
        block = decisions_as_system_prompt(str(tmp_path), files=[])
        assert "⚠ RECORDED WITH UNVERIFIED CLAIMS" in block
        assert "Warned decision" in block
        assert "Clean decision" in block


class TestExtractFromAnalysisPrompt:
    """Ground truth + distrust list are fed into the extraction prompt."""

    def test_inventory_and_report_injected(self, tmp_path):
        captured = {}

        async def fake_chat(messages, disable_thinking=True):
            captured["sys"] = messages[0]["content"]
            captured["user"] = messages[1]["content"]
            return "[]"

        agent = AsyncMock()
        agent.llm.chat = fake_chat
        asyncio.run(extract_from_analysis(
            agent,
            "Some analysis.",
            inventory="real.py — does a thing",
            verification_report="- [UNVERIFIED] `ghost` — not found",
        ))
        assert "Workspace listing (ground truth" in captured["user"]
        assert "real.py" in captured["user"]
        assert "Verification report" in captured["user"]
        assert "ghost" in captured["user"]
        assert "NEVER state that something does not exist" in captured["sys"]
