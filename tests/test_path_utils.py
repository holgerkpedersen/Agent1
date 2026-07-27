from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent_core.entities import FileOperationError, SecurityViolationError
from agent_core.path_utils import normalize_path


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Provide a temporary workspace directory."""
    return tmp_path


def test_normalize_valid_existing_file(workspace: Path) -> None:
    target_file = workspace / "test.txt"
    target_file.touch()

    result = normalize_path(str(target_file), workspace)
    assert result is not None
    assert result.resolve() == target_file.resolve()


def test_normalize_valid_string_input(workspace: Path) -> None:
    target_file = workspace / "nested" / "file.py"
    target_file.parent.mkdir(parents=True)
    target_file.touch()

    result = normalize_path(str(target_file), workspace)
    assert result is not None
    assert result == target_file.resolve()


def test_normalize_nonexistent_resolvable_path(workspace: Path) -> None:
    future_path = workspace / "does_not_exist.py"

    result = normalize_path(str(future_path), workspace)
    assert result is not None
    assert result.parent == workspace.resolve()


def test_block_traversal_attack(workspace: Path) -> None:
    with pytest.raises(SecurityViolationError):
        normalize_path("../../../etc/passwd", workspace)


def test_block_absolute_outside_workspace(workspace: Path) -> None:
    outside_target = workspace.parent / "outside.txt"

    with pytest.raises(SecurityViolationError):
        normalize_path(str(outside_target), workspace)


def test_block_symlink_escape(workspace: Path) -> None:
    link_name = workspace / "escape_link"
    target_outside = workspace.parent

    try:
        os.symlink(str(target_outside), str(link_name))
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unsupported on this platform")

    with pytest.raises(SecurityViolationError):
        normalize_path(str(link_name), workspace)


def test_reject_invalid_input_type(workspace: Path) -> None:
    with pytest.raises(FileOperationError):
        normalize_path(12345, workspace)  # type: ignore[arg-type]


def test_normalize_returns_resolved_absolute(workspace: Path) -> None:
    target_file = workspace / "sub" / "dir" / "file.txt"
    target_file.parent.mkdir(parents=True)
    target_file.touch()

    result = normalize_path(str(target_file), workspace)
    assert result is not None
    assert result.is_absolute()


def test_normalize_preserves_relative_within_workspace(workspace: Path) -> None:
    target_file = workspace / "relative.txt"
    target_file.touch()

    result = normalize_path("relative.txt", workspace)
    assert result is not None
    assert result.resolve().is_relative_to(workspace.resolve())


def test_normalize_deeply_nested_traversal_blocked(workspace: Path) -> None:
    with pytest.raises(SecurityViolationError):
        normalize_path("../../../../../root/secret", workspace)
