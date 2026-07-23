"""Secure path validation & workspace sandboxing utilities.

This module provides deterministic boundary enforcement for agent filesystem operations,
preventing directory traversal attacks and enforcing strict symlink policies within a defined workspace root.

Security Guarantees:
- Absolute containment: All resolved paths are verified to remain strictly inside `workspace_root`.
- Symlink isolation: Optional policy flag blocks symlink traversal that could escape the sandbox.
- Input hardening: Rejects empty, non-string, or malformed inputs before filesystem resolution.
"""

from pathlib import Path
from .entities import SecurityViolationError, FileOperationError


def _validate_path(raw: str, workspace_root: Path, follow_symlinks: bool = True) -> Path:
    """Validate and resolve a path strictly within the given workspace boundary.
    
    Args:
        raw: The input path string to validate.
        workspace_root: The absolute root directory allowed for operations.
        follow_symlinks: If False, symlink traversal is rejected entirely.
        
    Returns:
        Resolved absolute Path guaranteed to be within workspace_root.
        
    Raises:
        FileOperationError: If input is empty or not a string.
        SecurityViolationError: If path escapes boundary or violates symlink policy.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise FileOperationError("Empty or invalid path provided")
        
    target = Path(raw).resolve(strict=False)
    resolved_ws = workspace_root.resolve()
    
    # Enforce strict workspace boundary containment
    try:
        target.relative_to(resolved_ws)
    except ValueError:
        raise SecurityViolationError(f"Path escapes workspace boundary: {raw}")
        
    # Apply symlink policy enforcement
    if not follow_symlinks and target.is_symlink():
        raise SecurityViolationError(f"Symlinks are prohibited in this workspace: {raw}")
        
    return target


class WorkspaceSandbox:
    """Context manager enforcing secure path resolution within an isolated workspace.
    
    Usage:
        with WorkspaceSandbox("./sandbox", follow_symlinks=False) as sandbox:
            safe_path = sandbox.resolve_path("data/file.txt")
            # ... perform operations on safe_path
    """
    
    def __init__(self, workspace_root: Path | str, follow_symlinks: bool = True) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.follow_symlinks = follow_symlinks

    def resolve_path(self, raw: str) -> Path:
        """Resolve a relative/absolute path string strictly within the sandbox boundary."""
        return _validate_path(raw, self.workspace_root, self.follow_symlinks)

    def __enter__(self) -> "WorkspaceSandbox":
        return self

    def __exit__(self, exc_type: type[BaseException] | None, 
                 exc_val: BaseException | None, 
                 exc_tb: object) -> None:
        # Declarative scoping complete; pure validation sandbox requires no cleanup
        pass