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


class TestConsolidatedPathHelpers:
    """agent.py, FileSystem and FileSearcher must all delegate to the shared
    agent_core.path_utils.safe_path/resolve_path (candidate 8) — a single
    implementation, identical absolute result for the same input."""

    def test_all_callers_share_one_resolution(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from agent_core.file_searcher import FileSearcher
        from agent_core.file_system import FileSystem
        from agent_core.path_utils import resolve_path, safe_path

        (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        abs_mod = str((tmp_path / "mod.py").resolve())

        assert resolve_path("mod.py") == abs_mod
        assert safe_path("./mod.py") == abs_mod
        assert FileSystem(str(tmp_path)).normalize_path("mod.py") == abs_mod
        assert FileSystem(str(tmp_path)).safe_path("./mod.py") == abs_mod
        assert FileSearcher()._safe_path("./mod.py") == abs_mod

    def test_relative_search_path_now_resolves(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """FileSearcher._safe_path previously returned the raw relative string
        (to_windows_path only) — regression: it must resolve like the others."""
        from agent_core.file_searcher import FileSearcher

        (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        resolved = FileSearcher()._safe_path("mod.py")
        assert os.path.isabs(resolved)
        assert Path(resolved) == (tmp_path / "mod.py").resolve()
