"""Tests for analyze_cmd helpers: import parsing and file reference extraction."""
import pytest
from agent_core.commands.analyze_cmd import _parse_imports, _parse_file_refs


class TestParseImports:
    def test_parses_standard_imports(self):
        src = """
from agent_core.commands.fix_cmd import FixCommand, extract_signatures
from agent_core.commands.registry import CommandRegistry
from agent_core.constants import KNOWN_MODELS
import os
import json
"""
        result = _parse_imports(src)
        assert "agent_core/commands/fix_cmd.py" in result
        assert "agent_core/commands/registry.py" in result
        assert "agent_core/constants.py" in result

    def test_excludes_stdlib_imports(self):
        src = "import os\nimport json\nfrom typing import Optional\nfrom pathlib import Path"
        result = _parse_imports(src)
        assert result == []

    def test_excludes_relative_imports(self):
        src = "from .base import Command\nfrom .registry import CommandRegistry"
        result = _parse_imports(src)
        assert result == []

    def test_top_level_package_import_resolves_to_init(self):
        src = "from agent_core import to_windows_path"
        result = _parse_imports(src)
        assert "agent_core/__init__.py" in result

    def test_includes_src_agent1_imports(self):
        src = "from src.agent1.core.embedding_service import EmbeddingService"
        result = _parse_imports(src)
        assert "src/agent1/core/embedding_service.py" in result

    def test_includes_tests_imports(self):
        src = "from tests.test_tool_router import TestToolRouter"
        result = _parse_imports(src)
        assert "tests/test_tool_router.py" in result

    def test_resolves_deeply_nested_imports(self):
        src = "from agent_core.llm.lmstudio import LMStudioProvider"
        result = _parse_imports(src)
        assert "agent_core/llm/lmstudio.py" in result

    def test_deduplicates_duplicate_imports(self):
        src = "from agent_core.commands.fix_cmd import FixCommand\nfrom agent_core.commands.fix_cmd import extract_signatures"
        result = _parse_imports(src)
        assert result.count("agent_core/commands/fix_cmd.py") == 1

    def test_excludes_non_project_third_party(self):
        src = "from httpx import get\nfrom pydantic import BaseModel\nfrom numpy import array"
        result = _parse_imports(src)
        assert result == []


class TestParseFileRefs:
    def test_extracts_explicit_paths(self):
        text = """
The fix command is in agent_core/commands/fix_cmd.py.
It extends Command from agent_core/commands/base.py.
It also references agent_core/exceptions.py for error types.
"""
        result = _parse_file_refs(text)
        assert "agent_core/commands/fix_cmd.py" in result
        assert "agent_core/commands/base.py" in result
        assert "agent_core/exceptions.py" in result

    def test_deduplicates(self):
        text = "agent_core/commands/fix_cmd.py appears twice. agent_core/commands/fix_cmd.py again."
        result = _parse_file_refs(text)
        assert result.count("agent_core/commands/fix_cmd.py") == 1

    def test_extracts_src_paths(self):
        text = "The memory module is in src/agent1/core/memory_store.py."
        result = _parse_file_refs(text)
        assert "src/agent1/core/memory_store.py" in result

    def test_extracts_tests_paths(self):
        text = "See tests/test_tool_router.py for test examples."
        result = _parse_file_refs(text)
        assert "tests/test_tool_router.py" in result

    def test_ignores_non_project_paths(self):
        text = "stdlib has functools.py and pathlib.py, and pip installs site-packages/numpy/core.py"
        result = _parse_file_refs(text)
        assert "functools.py" not in result
        assert "pathlib.py" not in result

    def test_empty_string_returns_empty(self):
        assert _parse_file_refs("") == []

    def test_extracts_agent1_paths(self):
        text = "Generated code goes into agent1/logger.py or agent1/memory.py."
        result = _parse_file_refs(text)
        assert "agent1/logger.py" in result
        assert "agent1/memory.py" in result
