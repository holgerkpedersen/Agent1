"""Tests for autonomous mode primitives and the tailored next command."""
import json
from unittest.mock import patch

import pytest

from agent_core.commands.base import auto_choice, is_autonomous, set_autonomous
from agent_core.commands.workflow_cmd import (
    _planned_files_from_taskplan,
    _tailored_implement_parts,
)


@pytest.fixture(autouse=True)
def _reset_autonomy(monkeypatch):
    monkeypatch.delenv("AGENT_AUTONOMOUS", raising=False)
    set_autonomous(None)
    yield
    set_autonomous(None)


class TestAutonomyPrimitives:
    def test_interactive_mode_reads_input(self):
        with patch("builtins.input", return_value="y"):
            assert auto_choice("Spørgsmål? (y/N): ", default="n") == "y"

    def test_interactive_eof_returns_empty(self):
        with patch("builtins.input", side_effect=EOFError):
            assert auto_choice("Spørgsmål? (y/N): ", default="n") == ""

    def test_autonomous_returns_auto_default_without_prompt(self, capsys):
        set_autonomous(True)
        with patch("builtins.input", side_effect=AssertionError("must not prompt")):
            assert auto_choice("Slet? (y/N): ", default="n", auto_default="n") == "n"
        assert "auto: n" in capsys.readouterr().out

    def test_autonomous_falls_back_to_default(self):
        set_autonomous(True)
        assert auto_choice("Valg [a/b]: ", default="b") == "b"

    def test_env_var_enables_autonomy(self, monkeypatch):
        monkeypatch.setenv("AGENT_AUTONOMOUS", "1")
        assert is_autonomous() is True

    def test_flag_overrides_env(self, monkeypatch):
        monkeypatch.setenv("AGENT_AUTONOMOUS", "1")
        set_autonomous(False)
        assert is_autonomous() is False
        set_autonomous(None)
        assert is_autonomous() is True


class TestTailoredImplementParts:
    def _taskplan(self, tmp_path, lines):
        p = tmp_path / "project_tasks.md"
        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    def test_command_shape_with_analysis(self, tmp_path):
        tasks = self._taskplan(tmp_path, ["1. `agent_core/nlp/scheduler.py` — Schedule prompts"])
        analysis = tmp_path / "project_analysis.md"
        analysis.write_text("analysis", encoding="utf-8")
        parts, hint = _tailored_implement_parts(
            str(tasks), str(tmp_path / "project_plan.md"),
            str(tmp_path / "project_entities.md"), str(analysis),
            str(tmp_path), tasks.read_text(encoding="utf-8"),
        )
        assert parts[0] == "implement"
        assert str(tasks) in parts
        assert str(analysis) in parts  # analysis sits in the [1] slot
        assert "--workspace" in parts
        assert "--force" not in parts
        assert "No known conflicts" in hint

    def test_duplicate_warning_and_no_force(self, tmp_path):
        (tmp_path / "agent_core" / "security").mkdir(parents=True)
        (tmp_path / "agent_core" / "security" / "allowlist.py").write_text(
            '"""Safe shell command allow-list."""\nX = 1\n', encoding="utf-8",
        )
        tasks = self._taskplan(
            tmp_path,
            ["1. `agent_core/security/shell_allowlist.py` — Shell command allow-list + hardened blocklist"],
        )
        parts, hint = _tailored_implement_parts(
            str(tasks), str(tmp_path / "project_plan.md"),
            str(tmp_path / "project_entities.md"), None,
            str(tmp_path), tasks.read_text(encoding="utf-8"),
        )
        assert "duplicate existing modules" in hint
        assert "--force" not in parts

    def test_keep_only_with_matching_cache(self, tmp_path):
        tasks = self._taskplan(tmp_path, ["1. `agent_core/nlp/scheduler.py` — Schedule prompts"])
        content = tasks.read_text(encoding="utf-8")
        import hashlib
        cache = {
            "taskplan": str(tasks),
            "files": ["agent_core/nlp/scheduler.py"],
            "taskplan_hash": hashlib.md5(content.encode()).hexdigest()[:8],
        }
        (tmp_path / ".implement_cache.json").write_text(json.dumps(cache), encoding="utf-8")

        parts, _ = _tailored_implement_parts(
            str(tasks), str(tmp_path / "project_plan.md"),
            str(tmp_path / "project_entities.md"), None,
            str(tmp_path), content,
        )
        assert "--keep" in parts

    def test_no_keep_without_cache(self, tmp_path):
        tasks = self._taskplan(tmp_path, ["1. `agent_core/nlp/scheduler.py` — Schedule prompts"])
        parts, _ = _tailored_implement_parts(
            str(tasks), str(tmp_path / "project_plan.md"),
            str(tmp_path / "project_entities.md"), None,
            str(tmp_path), tasks.read_text(encoding="utf-8"),
        )
        assert "--keep" not in parts


class TestPlannedFiles:
    def test_extracts_backticked_py(self):
        files = _planned_files_from_taskplan(
            "1. `agent_core/nlp/a.py` — one\n2. `agent_core/nlp/b.py` — two\n"
            "3. `agent_core/nlp/a.py` — dup"
        )
        assert files == ["agent_core/nlp/a.py", "agent_core/nlp/b.py"]
