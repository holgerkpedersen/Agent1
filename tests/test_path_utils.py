from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final, Tuple

import pytest

from agent_core.config import AgentSettings
from agent_core.exceptions import FileOperationError, SecurityViolationError
from agent_core.path_utils import normalize_path


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Tuple[AgentSettings, Path]:
    """Provide a settings object rooted at the temporary workspace."""
    settings = AgentSettings(workspace_root=tmp_path)
    return settings, tmp_path


def test_normalize_valid_existing_file(temp_workspace: Tuple[AgentSettings, Path]) -> None:
    settings, workspace = temp_workspace
    target_file = workspace / "test.txt"
    target_file.touch()

    result = normalize_path(target_file, settings)
    assert result is not None
    assert result.resolve() == target_file.resolve()


def test_normalize_valid_string_input(temp_workspace: Tuple[AgentSettings, Path]) -> None:
    settings, workspace = temp_workspace
    target_file = workspace / "nested" / "file.py"
    target_file.parent.mkdir(parents=True)
    target_file.touch()

    result = normalize_path(str(target_file), settings)
    assert result is not None
    assert result == target_file.resolve()


def test_normalize_nonexistent_resolvable_path(
    temp_workspace: Tuple[AgentSettings, Path]
) -> None:
    settings, workspace = temp_workspace
    future_path = workspace / "does_not_exist.py"

    # Resolution with strict=False succeeds even if the file is absent.
    result = normalize_path(future_path, settings)
    assert result is not None
    assert result.parent == workspace.resolve()


def test_block_traversal_attack(temp_workspace: Tuple[AgentSettings, Path]) -> None:
    settings, _ = temp_workspace

    with pytest.raises(SecurityViolationError):
        normalize_path("../../../etc/passwd", settings)


def test_block_absolute_outside_workspace(
    temp_workspace: Tuple[AgentSettings, Path]
) -> None:
    settings, workspace = temp_workspace
    outside_target = tmp_path_factory.getbasetemp().parent / "outside.txt"

    with pytest.raises(SecurityViolationError):
        normalize_path(outside_target, settings)


def test_block_symlink_escape(temp_workspace: Tuple[AgentSettings, Path]) -> None:
    settings, workspace = temp_workspace
    link_name = workspace / "escape_link"
    target_outside = tmp_path_factory.getbasetemp().parent

    try:
        os.symlink(str(target_outside), str(link_name))
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unsupported on this platform")

    with pytest.raises(SecurityViolationError):
        normalize_path(link_name, settings)


def test_reject_invalid_input_type(temp_workspace: Tuple[AgentSettings, Path]) -> None:
    settings, _ = temp_workspace

    with pytest.raises(TypeError):
        normalize_path(12345, settings)  # type: ignore[arg-type]


def test_unix_absolute_path_handling_on_windows(
    temp_workspace: Tuple[AgentSettings, Path]
) -> None:
    settings, workspace = temp_workspace

    if not (sys.platform.startswith("win") or os.name == "nt"):
        pytest.skip("Unix-to-Windows conversion only applies on Windows/WSL")

    unix_path = "/c/Dev/test.txt"
    converted = normalize_path(unix_path, settings)
    assert converted is not None


def test_normalize_returns_resolved_absolute(
    temp_workspace: Tuple[AgentSettings, Path]
) -> None:
    settings, workspace = temp_workspace
    target_file = workspace / "sub" / "dir" / "file.txt"
    target_file.parent.mkdir(parents=True)
    target_file.touch()

    result = normalize_path(target_file, settings)
    assert result is not None
    assert result.is_absolute()


def test_normalize_preserves_relative_within_workspace(
    temp_workspace: Tuple[AgentSettings, Path]
) -> None:
    settings, workspace = temp_workspace
    target_file = workspace / "relative.txt"
    target_file.touch()

    result = normalize_path("relative.txt", settings)
    assert result is not None
    assert result.resolve().is_relative_to(workspace.resolve())


def test_normalize_deeply_nested_traversal_blocked(
    temp_workspace: Tuple[AgentSettings, Path]
) -> None:
    settings, _ = temp_workspace

    with pytest.raises(SecurityViolationError):
        normalize_path("../../../../../root/secret", settings)