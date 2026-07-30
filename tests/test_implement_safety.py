"""Tests for implement_cmd safety helpers: filename guards and auto-repair."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

from agent_core.commands.implement_cmd import (
    _is_dangerous_filename,
    _find_safe_subpackage,
)


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
