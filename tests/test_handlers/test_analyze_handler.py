from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent_core.handlers.analyze_handler import AnalyzeCommand


@pytest.fixture
def handler() -> AnalyzeCommand:
    return AnalyzeCommand()


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_analyze_valid_python_file(handler: AnalyzeCommand, tmp_path: Path) -> None:
    target_file = tmp_path / "sample.py"
    _write_file(target_file, "def greet(name):\n    return f'Hello {name}'\n")

    exit_code = asyncio.run(handler.handle([str(target_file)]))
    assert isinstance(exit_code, int)
    assert exit_code == 0


def test_analyze_nonexistent_file(handler: AnalyzeCommand, tmp_path: Path) -> None:
    target_file = tmp_path / "missing.py"

    exit_code = asyncio.run(handler.handle([str(target_file)]))
    assert exit_code == 1


def test_analyze_invalid_python_syntax(handler: AnalyzeCommand, tmp_path: Path) -> None:
    target_file = tmp_path / "broken.py"
    _write_file(target_file, "def broken(:\n    pass\n")

    exit_code = asyncio.run(handler.handle([str(target_file)]))
    assert isinstance(exit_code, int)
    assert exit_code == 2


def test_analyze_no_args(handler: AnalyzeCommand) -> None:
    exit_code = asyncio.run(handler.handle([]))
    assert exit_code == 1


def test_handler_name_property(handler: AnalyzeCommand) -> None:
    assert handler.name == "analyze"
