"""Tests for implement_cmd safety helpers: filename guards and auto-repair."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

from agent_core.commands.implement_cmd import (
    _is_dangerous_filename,
    _find_safe_subpackage,
    _shadowing_stdlib_dir,
    _unwired_closure,
    _prune_empty_dirs,
    _check_planned_duplicates,
    _filter_duplicate_planned,
    _suggest_consumers,
    _ensure_package_inits,
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

    def test_planned_test_file_never_flagged(self, tmp_path):
        """A planned test file mirrors its target by design — the duplicate
        gate must not flag it, even when it names the covered module
        (observed: test_agent_chat_nlp.py vs agent.py)."""
        (tmp_path / "agent.py").write_text(
            '"""Agent — chat loop."""\ndef chat_nlp():\n    return 1\n',
            encoding="utf-8",
        )
        reasons = _check_planned_duplicates(
            ["tests/unit/test_agent_chat_nlp.py"],
            str(tmp_path),
            "1. `tests/unit/test_agent_chat_nlp.py` — cover chat_nlp loop\n",
        )
        assert reasons == []

    def test_mixed_planned_only_production_flagged(self, tmp_path):
        (tmp_path / "agent_core" / "security").mkdir(parents=True)
        (tmp_path / "agent_core" / "security" / "allowlist.py").write_text(
            '"""Command allow-list."""\nSAFE = set()\n', encoding="utf-8",
        )
        reasons = _check_planned_duplicates(
            [
                "tests/unit/test_allowlist.py",
                "agent_core/security/shell_allowlist.py",
            ],
            str(tmp_path),
        )
        assert any("shell_allowlist.py" in r for r in reasons)
        assert not any("test_allowlist.py" in r for r in reasons)


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


class TestShadowingStdlibDir:
    """A path segment only 'shadows' when the directory does NOT already
    exist in the workspace — an existing package like ``agent_core/utils/``
    is a deliberate project choice and must not be renamed."""

    def test_existing_dir_is_not_a_shadow(self, tmp_path):
        (tmp_path / "agent_core" / "utils").mkdir(parents=True)
        assert _shadowing_stdlib_dir("agent_core/utils/lint_guard.py", tmp_path) == ""
        assert _shadowing_stdlib_dir("agent_core/utils/module_similarity.py", tmp_path) == ""

    def test_new_dir_is_a_shadow(self, tmp_path):
        assert _shadowing_stdlib_dir("logging/log.py", tmp_path) == "logging"

    def test_bare_name_is_not_a_shadow(self, tmp_path):
        assert _shadowing_stdlib_dir("constants.py", tmp_path) == ""

    def test_dangerous_filename_ignores_existing_project_package(self, tmp_path):
        (tmp_path / "agent_core" / "utils").mkdir(parents=True)
        dangerous, _ = _is_dangerous_filename("agent_core/utils/lint_guard.py", tmp_path)
        assert not dangerous

    def test_dangerous_filename_flags_new_stdlib_dir(self, tmp_path):
        dangerous, reason = _is_dangerous_filename("logging/log.py", tmp_path)
        assert dangerous
        assert "shadows stdlib" in reason


class TestModifyMode:
    """--modify merges generated content into existing compile-OK modules as
    a reviewed diff instead of skipping them (default) or overwriting them
    wholesale (--force)."""

    @staticmethod
    def _run(tmp_path, stub_content, extra_args=(), auto="y", modify=True):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import patch
        from agent_core.commands.implement_cmd import ImplementCommand

        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        target = pkg / "mod_me.py"
        target.write_text("def foo():\n    return 1\n", encoding="utf-8")
        (tmp_path / "tasks.md").write_text(
            "1. `pkg/mod_me.py` — extend foo\n", encoding="utf-8"
        )

        async def chat(messages, **kwargs):
            return f"[FILE: pkg/mod_me.py]\n```python\n{stub_content}```\n"

        agent = SimpleNamespace(workspace=str(tmp_path), llm=SimpleNamespace(chat=chat))
        with patch("agent_core.commands.implement_cmd.auto_choice", return_value=auto):
            args = [str(tmp_path / "tasks.md"), *extra_args]
            if modify:
                args.insert(1, "--modify")
            ok = asyncio.run(
                ImplementCommand().execute(args, agent)
            )
        return ok, target

    def test_applies_diff_after_approval(self, tmp_path):
        ok, target = self._run(tmp_path, "def foo():\n    return 2\n")
        assert ok is True
        assert target.read_text(encoding="utf-8") == "def foo():\n    return 2\n"

    def test_declines_without_approval(self, tmp_path):
        ok, target = self._run(tmp_path, "def foo():\n    return 2\n", auto="n")
        assert ok is True
        assert target.read_text(encoding="utf-8") == "def foo():\n    return 1\n"

    def test_unchanged_content_skipped(self, tmp_path):
        ok, target = self._run(tmp_path, "def foo():\n    return 1\n")
        assert ok is True
        assert target.read_text(encoding="utf-8") == "def foo():\n    return 1\n"

    def test_wholesale_rewrite_rejected_without_allow_rewrite(self, tmp_path):
        ok, target = self._run(
            tmp_path,
            "def totally_different_interface():\n    return [i for i in range(50)]\n",
        )
        assert ok is True
        assert target.read_text(encoding="utf-8") == "def foo():\n    return 1\n"

    def test_wholesale_rewrite_applies_with_allow_rewrite(self, tmp_path):
        ok, target = self._run(
            tmp_path,
            "def totally_different_interface():\n    return [i for i in range(50)]\n",
            extra_args=["--allow-rewrite"],
        )
        assert ok is True
        assert target.read_text(encoding="utf-8").startswith("def totally_different_interface")

    def test_keep_mode_still_applies_modify_diff(self, tmp_path):
        """--keep + --modify must NOT short-circuit: existing compile-OK
        modules are modify targets, not skips (observed: keep-mode run
        reported "already exists, compile OK" and did nothing)."""
        ok, target = self._run(tmp_path, "def foo():\n    return 2\n", extra_args=["--keep"])
        assert ok is True
        assert target.read_text(encoding="utf-8") == "def foo():\n    return 2\n"

    def test_keep_without_modify_still_skips(self, tmp_path):
        ok, target = self._run(
            tmp_path, "def foo():\n    return 2\n", extra_args=["--keep"], modify=False,
        )
        assert ok is True
        assert target.read_text(encoding="utf-8") == "def foo():\n    return 1\n"


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


class TestEnsurePackageInits:
    """Package-init touch loop must NOT packageify tests/ or src/ trees
    (observed: the run created tests/__init__.py + tests/unit/__init__.py,
    which break pytest implicit-sibling imports, and src/__init__.py +
    src/agent1/__init__.py noise)."""

    def test_creates_inits_for_normal_packages(self, tmp_path):
        (tmp_path / "pkg" / "sub").mkdir(parents=True)
        _ensure_package_inits(tmp_path / "pkg" / "sub", tmp_path)
        assert (tmp_path / "pkg" / "sub" / "__init__.py").exists()
        assert (tmp_path / "pkg" / "__init__.py").exists()

    def test_skips_tests_tree(self, tmp_path):
        (tmp_path / "tests" / "unit").mkdir(parents=True)
        _ensure_package_inits(tmp_path / "tests" / "unit", tmp_path)
        assert not (tmp_path / "tests" / "__init__.py").exists()
        assert not (tmp_path / "tests" / "unit" / "__init__.py").exists()

    def test_skips_src_tree(self, tmp_path):
        (tmp_path / "src" / "agent1" / "core").mkdir(parents=True)
        _ensure_package_inits(tmp_path / "src" / "agent1" / "core", tmp_path)
        assert not (tmp_path / "src" / "__init__.py").exists()
        assert not (tmp_path / "src" / "agent1" / "__init__.py").exists()
        assert not (tmp_path / "src" / "agent1" / "core" / "__init__.py").exists()

    def test_mixed_tree_only_normal_part_touched(self, tmp_path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "app").mkdir()
        _ensure_package_inits(tmp_path / "app", tmp_path)
        _ensure_package_inits(tmp_path / "tests", tmp_path)
        assert (tmp_path / "app" / "__init__.py").exists()
        assert not (tmp_path / "tests" / "__init__.py").exists()

    def test_preserves_nonempty_init(self, tmp_path):
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "__init__.py").write_text("X = 1\n", encoding="utf-8")
        _ensure_package_inits(tmp_path / "pkg", tmp_path)
        assert (tmp_path / "pkg" / "__init__.py").read_text(encoding="utf-8") == "X = 1\n"


class TestDependencyCascadeSafety:
    """The post-loop dependency cleanup must NEVER delete a pre-existing file
    this run did not write (observed: an untouched agent.py was unlinked
    because it imported a wholesale-rewrite-rejected tool_loop.py)."""

    @staticmethod
    def _setup_workspace(tmp_path):
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "sibling.py").write_text(
            "def sibling_impl():\n    return 1\n", encoding="utf-8"
        )
        (pkg / "mod_me.py").write_text(
            "from pkg.sibling import sibling_impl\n\n"
            "def foo():\n    return sibling_impl()\n",
            encoding="utf-8",
        )
        (tmp_path / "tasks.md").write_text(
            "1. `pkg/mod_me.py` — extend foo\n"
            "2. `pkg/sibling.py` — extend sibling_impl\n",
            encoding="utf-8",
        )

    def test_rejected_modify_targets_never_deleted(self, tmp_path):
        """Regression: mod_me.py imports sibling.py; both generated contents
        are wholesale rewrites and get rejected.  Neither original may be
        touched, and a backup of both must exist under backups/."""
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import patch
        from agent_core.commands.implement_cmd import ImplementCommand

        self._setup_workspace(tmp_path)
        original_mod = (tmp_path / "pkg" / "mod_me.py").read_text(encoding="utf-8")
        original_sib = (tmp_path / "pkg" / "sibling.py").read_text(encoding="utf-8")

        async def chat(messages, **kwargs):
            last_user = messages[-1]["content"] if messages else ""
            if "pkg/sibling.py" in last_user:
                return "[FILE: pkg/sibling.py]\n```python\ndef sibling_v2():\n    return 2\n```\n"
            return "[FILE: pkg/mod_me.py]\n```python\ndef mod_v2():\n    return 3\n```\n"

        agent = SimpleNamespace(workspace=str(tmp_path), llm=SimpleNamespace(chat=chat))
        with patch("agent_core.commands.implement_cmd.auto_choice", return_value="n"):
            ok = asyncio.run(
                ImplementCommand().execute(
                    [str(tmp_path / "tasks.md"), "--modify"], agent
                )
            )

        assert ok is True
        # The two untouched originals must survive the dependency cascade.
        assert (tmp_path / "pkg" / "mod_me.py").read_text(encoding="utf-8") == original_mod
        assert (tmp_path / "pkg" / "sibling.py").read_text(encoding="utf-8") == original_sib
        # Pre-run backups of the existing targets exist.
        backups = list((tmp_path / "backups").glob("*.py"))
        assert len(backups) >= 2
        assert any("sibling_impl" in b.read_text(encoding="utf-8") for b in backups)

    def test_keep_mode_existing_file_importing_rejected_kept(self, tmp_path, capsys):
        """Without --modify, an existing compile-OK file is skipped; a NEW
        planned file that gets rejected must not cause the existing one to be
        deleted just because it imports the new module."""
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import patch
        from agent_core.commands.implement_cmd import ImplementCommand

        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "existing.py").write_text(
            "from pkg.new_helper import helper\n\n"
            "def foo():\n    return helper()\n",
            encoding="utf-8",
        )
        (tmp_path / "tasks.md").write_text(
            "1. `pkg/existing.py` — keep existing\n"
            "2. `pkg/new_helper.py` — add helper module\n",
            encoding="utf-8",
        )

        async def chat(messages, **kwargs):
            # Oversized content → rejected by the 50KB guard.
            blob = "x = 1\n" * 12000
            return f"[FILE: pkg/new_helper.py]\n```python\n{blob}```\n"

        agent = SimpleNamespace(workspace=str(tmp_path), llm=SimpleNamespace(chat=chat))
        with patch("agent_core.commands.implement_cmd.auto_choice", return_value="n"):
            ok = asyncio.run(
                ImplementCommand().execute(
                    [str(tmp_path / "tasks.md"), "--keep"], agent
                )
            )

        assert ok is True
        # new_helper was rejected (>50KB); the existing importer survives the
        # dependency cascade because it was never written this run.
        assert (tmp_path / "pkg" / "existing.py").exists()
        assert not (tmp_path / "pkg" / "new_helper.py").exists()
        assert "KEPT: pkg/existing.py imports rejected module pkg/new_helper.py" in capsys.readouterr().out

    def test_batch_filename_mismatch_ignored(self, tmp_path, capsys):
        """A [FILE:] name outside the planned batch must be ignored, not
        accepted (observed: the secrets.py batch returned a sanitizer.py
        block, silently dropping the planned file)."""
        import asyncio
        from types import SimpleNamespace
        from agent_core.commands.implement_cmd import ImplementCommand

        (tmp_path / "pkg").mkdir()
        (tmp_path / "tasks.md").write_text(
            "1. `pkg/newmod.py` — add module\n",
            encoding="utf-8",
        )

        async def chat(messages, **kwargs):
            return "[FILE: pkg/wrong_name.py]\n```python\ndef w():\n    return 1\n```\n"

        agent = SimpleNamespace(workspace=str(tmp_path), llm=SimpleNamespace(chat=chat))
        ok = asyncio.run(
            ImplementCommand().execute([str(tmp_path / "tasks.md")], agent)
        )

        assert ok is True
        assert not (tmp_path / "pkg" / "newmod.py").exists()
        assert not (tmp_path / "pkg" / "wrong_name.py").exists()
        out = capsys.readouterr().out
        assert "WARNING: [FILE: pkg/wrong_name.py] is not in the planned batch" in out
