from __future__ import annotations

import ast
from pathlib import Path
from typing import Final, List

import pytest

from agent_core.config import AgentSettings
from agent_core.exceptions import FileOperationError
from agent_core.handlers.analyze_handler import AnalyzeCommand


@pytest.fixture
def temp_workspace(tmp_path: Path) -> tuple[AgentSettings, Path]:
    settings = AgentSettings(workspace_root=tmp_path)
    return settings, tmp_path


@pytest.fixture
def handler() -> AnalyzeCommand:
    return AnalyzeCommand()


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_analyze_valid_python_file(handler: AnalyzeCommand, temp_workspace: tuple[AgentSettings, Path]) -> None:
    settings, workspace = temp_workspace
    target_file = workspace / "sample.py"
    _write_file(target_file, "def greet(name):\n    return f'Hello {name}'\n")

    exit_code = handler.handle([str(target_file)])  # type: ignore[arg-type]
    assert isinstance(exit_code, int)


def test_analyze_nonexistent_file(handler: AnalyzeCommand, temp_workspace: tuple[AgentSettings, Path]) -> None:
    settings, workspace = temp_workspace
    target_file = workspace / "missing.py"

    with pytest.raises(FileOperationError):
        handler.handle([str(target_file)])  # type: ignore[arg-type]


def test_analyze_invalid_python_syntax(handler: AnalyzeCommand, temp_workspace: tuple[AgentSettings, Path]) -> None:
    settings, workspace = temp_workspace
    target_file = workspace / "broken.py"
    _write_file(target_file, "def broken(:\n    pass\n")

    exit_code = handler.handle([str(target_file)])  # type: ignore[arg-type]
    assert isinstance(exit_code, int)


def test_analyze_no_args(handler: AnalyzeCommand) -> None:
    with pytest.raises(IndexError):
        handler.handle([])  # type: ignore[arg-type]


def test_handler_name_property(handler: AnalyzeCommand) -> None:
    assert handler.name == "analyze"