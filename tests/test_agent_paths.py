from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent_core.path_utils import normalize_path

WS = Path(__file__).resolve().parent.parent


@pytest.fixture
def workspace() -> Path:
    return WS


def test_normalize_relative(workspace: Path) -> None:
    result = normalize_path("agent.py", workspace)
    assert result.is_absolute()
    assert (workspace / "agent.py").resolve() == result


def test_normalize_absolute(workspace: Path) -> None:
    result = normalize_path(str(workspace / "agent.py"), workspace)
    assert result == (workspace / "agent.py").resolve()


def test_normalize_posix_slash(workspace: Path) -> None:
    result = normalize_path("agent_core/entities.py", workspace)
    assert result == (workspace / "agent_core" / "entities.py").resolve()


def test_block_traversal_double_dot(workspace: Path) -> None:
    from agent_core.entities import SecurityViolationError
    with pytest.raises(SecurityViolationError):
        normalize_path("../etc/passwd", workspace)


def test_block_traversal_absolute_escape(workspace: Path) -> None:
    from agent_core.entities import SecurityViolationError
    with pytest.raises(SecurityViolationError):
        normalize_path("C:\\Windows\\System32", workspace)


def test_reject_empty_path(workspace: Path) -> None:
    from agent_core.entities import FileOperationError
    with pytest.raises(FileOperationError):
        normalize_path("", workspace)


def test_reject_non_string(workspace: Path) -> None:
    from agent_core.entities import FileOperationError
    with pytest.raises(FileOperationError):
        normalize_path(None, workspace)  # type: ignore[arg-type]
