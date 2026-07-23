import pytest
from pathlib import Path

from agent_core.path_utils import _validate_path, WorkspaceSandbox
from agent_core.entities import SecurityViolationError, FileOperationError


@pytest.fixture()
def workspace_root(tmp_path: Path) -> Path:
    """Create a deterministic temporary workspace for path validation tests."""
    root = tmp_path / "workspace"
    root.mkdir(parents=True)

    # Valid internal structure
    (root / "allowed.txt").write_text("safe")
    subdir = root / "subdir"
    subdir.mkdir()
    (subdir / "nested.txt").write_text("also safe")

    # External target for symlink boundary testing
    external_target = tmp_path / "external_secret.txt"
    external_target.write_text("leaked!")

    return root


class TestValidatePath:
    """Unit tests for _validate_path() security and normalization logic."""

    def test_accepts_valid_relative_paths(self, workspace_root: Path) -> None:
        assert _validate_path("allowed.txt", workspace_root) == workspace_root / "allowed.txt"

    def test_accepts_valid_nested_paths(self, workspace_root: Path) -> None:
        expected = workspace_root / "subdir" / "nested.txt"
        assert _validate_path("subdir/nested.txt", workspace_root) == expected

    def test_blocks_directory_traversal_escape(self, workspace_root: Path) -> None:
        with pytest.raises(SecurityViolationError):
            _validate_path("../outside.txt", workspace_root)

    def test_blocks_deep_traversal_escape(self, workspace_root: Path) -> None:
        with pytest.raises(SecurityViolationError):
            _validate_path("subdir/../../escape.txt", workspace_root)

    def test_blocks_absolute_paths_outside_workspace(self, workspace_root: Path) -> None:
        external = workspace_root.parent / "external.txt"
        with pytest.raises(SecurityViolationError):
            _validate_path(str(external), workspace_root)

    def test_accepts_absolute_path_inside_workspace(self, workspace_root: Path) -> None:
        internal_abs = str(workspace_root / "allowed.txt")
        assert _validate_path(internal_abs, workspace_root) == workspace_root / "allowed.txt"

    def test_rejects_empty_string_input(self, workspace_root: Path) -> None:
        with pytest.raises(FileOperationError):
            _validate_path("", workspace_root)

    def test_rejects_non_string_input(self, workspace_root: Path) -> None:
        with pytest.raises(FileOperationError):
            _validate_path(None, workspace_root)  # type: ignore[arg-type]

    def test_symlink_policy_blocks_links_when_disabled(self, workspace_root: Path) -> None:
        (workspace_root / "link_to_outside").symlink_to(workspace_root.parent / "secret.txt")
        
        with pytest.raises(SecurityViolationError):
            _validate_path("link_to_outside", workspace_root, follow_symlinks=False)

    def test_symlink_policy_allows_links_when_enabled(self, workspace_root: Path) -> None:
        (workspace_root / "internal_link").symlink_to(workspace_root / "allowed.txt")
        
        result = _validate_path("internal_link", workspace_root, follow_symlinks=True)
        assert result == workspace_root / "allowed.txt"


class TestWorkspaceSandbox:
    """Unit tests for WorkspaceSandbox context manager and resolve_path."""

    def test_sandbox_context_manager_scoping(self, workspace_root: Path) -> None:
        with WorkspaceSandbox(workspace_root) as sandbox:
            resolved = sandbox.resolve_path("subdir/nested.txt")
            assert resolved == workspace_root / "subdir" / "nested.txt"

    def test_sandbox_blocks_escape_via_resolve_path(self, workspace_root: Path) -> None:
        with WorkspaceSandbox(workspace_root) as sandbox:
            with pytest.raises(SecurityViolationError):
                sandbox.resolve_path("../escape.txt")

    def test_sandbox_propagates_symlink_policy(self, workspace_root: Path) -> None:
        (workspace_root / "link").symlink_to(workspace_root.parent / "outside.txt")
        
        with WorkspaceSandbox(workspace_root, follow_symlinks=False) as sandbox:
            with pytest.raises(SecurityViolationError):
                sandbox.resolve_path("link")

    def test_sandbox_graceful_exit_after_exception(self, workspace_root: Path) -> None:
        """Ensure context manager cleans up and allows re-entry after internal failures."""
        try:
            with WorkspaceSandbox(workspace_root) as sandbox:
                _ = sandbox.resolve_path("../bad.txt")
        except SecurityViolationError:
            pass  # Expected failure inside scope

        # Re-entering should work correctly without residual state
        with WorkspaceSandbox(workspace_root) as sandbox:
            assert sandbox.resolve_path("allowed.txt") == workspace_root / "allowed.txt"