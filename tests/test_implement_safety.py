"""Tests for implement_cmd safety helpers: filename guards and auto-repair."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

from agent_core.commands.implement_cmd import (
    _is_dangerous_filename,
    _find_safe_subpackage,
    _unwired_closure,
    _prune_empty_dirs,
    _check_planned_duplicates,
    _filter_duplicate_planned,
    _suggest_consumers,
)


class TestFilterDuplicatePlanned:
    def test_duplicates_dropped_rest_kept(self) -> None:
        remaining, blocked = _filter_duplicate_planned(
            ["a.py", "b.py", "c.py"],
            ["b.py — duplicates existing module(s): allowlist.py", "c.py — duplicates existing module(s): sanitizer.py"],
        )
        assert remaining == ["a.py"]
        assert blocked == ["b.py", "c.py"]

    def test_no_duplicates_unchanged(self) -> None:
        remaining, blocked = _filter_duplicate_planned(["a.py", "b.py"], [])
        assert remaining == ["a.py", "b.py"]
        assert blocked == []


class TestSuggestConsumers:
    def test_token_based_and_concept_map(self, tmp_path) -> None:
        (tmp_path / "agent.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "agent_core").mkdir()
        (tmp_path / "agent_core" / "tool_schemas.py").write_text("X = 1\n", encoding="utf-8")

        suggestions = _suggest_consumers(
            ["agent_core/nlp/tool_schema.py", "agent_core/nlp/chain_limiter.py"],
            str(tmp_path),
        )
        assert any("tool_schemas.py" in s for s in suggestions["agent_core/nlp/tool_schema.py"])
        assert "agent.py" in suggestions["agent_core/nlp/chain_limiter.py"]

    def test_fallback_is_agent_py(self, tmp_path) -> None:
        (tmp_path / "agent.py").write_text("x = 1\n", encoding="utf-8")
        suggestions = _suggest_consumers(["agent_core/nlp/oddball.py"], str(tmp_path))
        assert suggestions["agent_core/nlp/oddball.py"] == ["agent.py"]


class TestPlannedDuplicates:
    """Planned NEW modules that duplicate existing ones must be flagged before
    any generation happens."""

    def test_flags_near_duplicate_planned_modules(self, tmp_path):
        (tmp_path / "agent_core").mkdir(parents=True)
        (tmp_path / "agent_core" / "security").mkdir()
        (tmp_path / "agent_core" / "security" / "allowlist.py").write_text(
            '"""Command allow-list."""\nSAFE = set()\n', encoding="utf-8",
        )
        (tmp_path / "agent_core" / "security" / "sanitizer.py").write_text(
            '"""Input sanitizer."""\nX = 1\n', encoding="utf-8",
        )

        reasons = _check_planned_duplicates(
            [
                "agent_core/security/shell_allowlist.py",
                "agent_core/security/sanitizer_fix.py",
                "agent_core/nlp/output_sanitizer.py",
                "agent_core/nlp/chain_limiter.py",
            ],
            str(tmp_path),
        )

        text = "\n".join(reasons)
        assert "shell_allowlist.py" in text and "allowlist.py" in text
        assert "sanitizer_fix.py" in text and "sanitizer.py" in text
        assert "output_sanitizer.py" in text and "sanitizer.py" in text
        assert "chain_limiter.py" not in text  # no existing counterpart

    def test_distinct_planned_modules_pass(self, tmp_path):
        (tmp_path / "agent_core" / "security").mkdir(parents=True)
        (tmp_path / "agent_core" / "security" / "allowlist.py").write_text(
            '"""Command allow-list."""\nSAFE = set()\n', encoding="utf-8",
        )
        reasons = _check_planned_duplicates(
            ["agent_core/commands/reporting_cmd.py"], str(tmp_path),
        )
        assert reasons == []

    def test_semantic_layer_fires_through_gate(self, tmp_path):
        """The geometric layer must flag a planned module whose DESCRIPTION
        matches an existing module semantically — even when no name token is
        shared."""
        (tmp_path / "agent_core" / "security").mkdir(parents=True)
        (tmp_path / "agent_core" / "security" / "sanitizer.py").write_text(
            '"""Input sanitizer stripping shell-injection payloads before re-injection."""\nX = 1\n',
            encoding="utf-8",
        )
        taskplan = (
            "1. `agent_core/nlp/safe_text.py` — Sanitize text with the input sanitizer "
            "before re-injection into the model\n"
        )
        reasons = _check_planned_duplicates(
            ["agent_core/nlp/safe_text.py"],
            str(tmp_path),
            taskplan,
        )
        assert any("TF-IDF" in r for r in reasons), reasons


class TestUnwiredClosure:
    """'y' on the review delete prompt must remove the whole orphaned
    component, not just the directly-flagged files."""

    def test_closure_cascades_to_sibling_orphans(self, tmp_path):
        # a -> b -> c : only 'a' is unwired initially (b/c are referenced),
        # but deleting 'a' orphans b, which orphans c — all three must go.
        (tmp_path / "a.py").write_text("from b import x\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("from c import y\n", encoding="utf-8")
        (tmp_path / "c.py").write_text("y = 1\n", encoding="utf-8")
        py_new = ["a.py", "b.py", "c.py"]

        delete_set = _unwired_closure(py_new, str(tmp_path), {"a.py"})
        assert delete_set == {"a.py", "b.py", "c.py"}

    def test_closure_never_deletes_files_imported_by_real_code(self, tmp_path):
        # main.py (existing code) imports a; only b is flagged. Deleting b
        # orphans c, but a must survive because real code references it.
        (tmp_path / "main.py").write_text("from a import z\n", encoding="utf-8")
        (tmp_path / "a.py").write_text("from b import x\nz = 1\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("from c import y\n", encoding="utf-8")
        (tmp_path / "c.py").write_text("y = 1\n", encoding="utf-8")
        py_new = ["a.py", "b.py", "c.py"]

        delete_set = _unwired_closure(py_new, str(tmp_path), {"b.py"})
        assert delete_set == {"b.py", "c.py"}
        assert "a.py" not in delete_set


class TestPruneEmptyDirs:
    def test_removes_package_left_empty_by_deletion(self, tmp_path):
        pkg = tmp_path / "agent_core" / "nlp"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").touch()
        mod = pkg / "chain_limiter.py"
        mod.write_text("x = 1\n", encoding="utf-8")
        mod.unlink()  # the delete loop removed the module

        _prune_empty_dirs(str(tmp_path), {"agent_core/nlp/chain_limiter.py"})

        assert not mod.exists()
        assert not (pkg / "__init__.py").exists()
        assert not pkg.exists()

    def test_keeps_package_with_other_content(self, tmp_path):
        pkg = tmp_path / "agent_core" / "nlp"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").touch()
        (pkg / "output_sanitizer.py").write_text("x = 1\n", encoding="utf-8")
        deleted = pkg / "chain_limiter.py"
        deleted.write_text("y = 1\n", encoding="utf-8")
        deleted.unlink()  # the delete loop removed the module

        _prune_empty_dirs(str(tmp_path), {"agent_core/nlp/chain_limiter.py"})

        assert not deleted.exists()
        assert (pkg / "__init__.py").exists()
        assert (pkg / "output_sanitizer.py").exists()


class TestIsDangerousFilename:
    def test_bare_filename_at_root_is_dangerous(self):
        ws = Path(os.getcwd())
        dangerous, reason = _is_dangerous_filename("types.py", ws)
        assert dangerous
        assert "bare workspace-root" in reason

    def test_bare_filename_evaluator_is_dangerous(self):
        ws = Path(os.getcwd())
        dangerous, reason = _is_dangerous_filename("evaluator.py", ws)
        assert dangerous
        assert "bare workspace-root" in reason

    def test_sub_package_path_is_not_dangerous(self):
        ws = Path(os.getcwd())
        dangerous, _ = _is_dangerous_filename("agent1/types.py", ws)
        assert not dangerous

    def test_src_sub_package_is_not_dangerous(self):
        ws = Path(os.getcwd())
        dangerous, _ = _is_dangerous_filename("src/agent1/memory.py", ws)
        assert not dangerous

    def test_agent_core_sub_path_is_not_dangerous(self):
        ws = Path(os.getcwd())
        dangerous, _ = _is_dangerous_filename("agent_core/commands/new_cmd.py", ws)
        assert not dangerous

    def test_init_at_workspace_root_is_dangerous_even_in_subpackage_check(self):
        # __init__.py is caught by either the explicit init check or bare filename check
        ws = Path(os.getcwd())
        dangerous, reason = _is_dangerous_filename("__init__.py", ws)
        assert dangerous
        assert "workspace root" in reason.lower()  # Caught by either check

    def test_init_in_sub_package_is_not_dangerous(self):
        ws = Path(os.getcwd())
        dangerous, _ = _is_dangerous_filename("agent1/__init__.py", ws)
        assert not dangerous

    def test_empty_name_is_dangerous(self):
        ws = Path(os.getcwd())
        dangerous, _ = _is_dangerous_filename("", ws)
        assert dangerous

    def test_invalid_name_is_dangerous(self):
        ws = Path(os.getcwd())
        dangerous, _ = _is_dangerous_filename("/", ws)
        assert dangerous

    def test_all_common_bare_names_are_dangerous(self):
        ws = Path(os.getcwd())
        for name in ["config.py", "logger.py", "utils.py", "memory.py",
                     "agent.py", "model.py", "cache.py", "tools.py"]:
            dangerous, reason = _is_dangerous_filename(name, ws)
            assert dangerous, f"{name} should be dangerous but wasn't: {reason}"

    def test_different_workspace_root_detects_correctly(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            dangerous, _ = _is_dangerous_filename("types.py", ws)
            assert dangerous


class TestFindSafeSubpackage:
    def setup_method(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self._tmp.name)

    def teardown_method(self):
        self._tmp.cleanup()

    def test_creates_agent1_if_none_exist(self):
        result = _find_safe_subpackage(self.ws)
        assert result == "agent1"
        assert (self.ws / "agent1").is_dir()
        assert (self.ws / "agent1" / "__init__.py").exists()

    def test_returns_existing_agent1(self):
        (self.ws / "agent1").mkdir(parents=True)
        (self.ws / "agent1" / "__init__.py").touch()
        result = _find_safe_subpackage(self.ws)
        assert result == "agent1"

    def test_returns_src_agent1_if_agent1_missing(self):
        (self.ws / "src").mkdir()
        (self.ws / "src" / "agent1").mkdir()
        result = _find_safe_subpackage(self.ws)
        assert result == "src/agent1"
