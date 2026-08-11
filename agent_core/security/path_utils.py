"""Centralized path normalization and workspace sandboxing.

All file-system interactions must pass through ``normalize_path`` to ensure
that no operation escapes the configured workspace root directory.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

# Sentinel used when the resolved path is outside the sandbox.
_OUTSIDE_SANDBOX: Final[str] = "<outside_sandbox>"


class SecurityViolationError(Exception):
    """Raised when a path operation violates workspace sandbox rules."""

    pass


def normalize_path(workspace_root: Path, target_path: str) -> Path:
    """Resolve *target_path* relative to *workspace_root*.

    The result is guaranteed to be strictly inside *workspace_root* (or equal).
    
    Parameters
    ----------
    workspace_root:
        The root directory that serves as the sandbox boundary.  Should be an
        absolute, existing path for predictable behaviour.
    target_path:
        User-supplied or tool-generated path string.  May be relative or
        absolute; may contain ``..`` segments.

    Returns
    -------
    :class:`pathlib.Path`
        The resolved, sandboxed path.

    Raises
    ------
    SecurityViolationError
        If the resolved path escapes *workspace_root*.
    """
    # 1. Normalise inputs --------------------------------------------------
    root: Path = workspace_root.resolve(strict=False)
    
    # Handle empty strings gracefully by treating them as current-dir relative
    if not target_path or not target_path.strip():
        raise SecurityViolationError(
            f"Empty path string is not allowed (workspace={root})"
        )

    # 2. Resolve the candidate path ----------------------------------------
    try:
        resolved: Path = (root / target_path).resolve(strict=False)
    except OSError as exc:
        raise SecurityViolationError(
            f"Could not resolve path '{target_path}': {exc}"
        ) from exc

    # 3. Sandbox enforcement -----------------------------------------------
    _ensure_under_root(root, resolved, target_path)

    logger.debug("Resolved '%s' -> %s", target_path, resolved)
    return resolved


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_under_root(
    root: Path,
    candidate: Path,
    original: str,
) -> None:
    """Raise ``SecurityViolationError`` if *candidate* is outside *root*.

    Uses os.path.commonpath for cross-platform correctness (handles Windows
    drive letters and POSIX symlinks correctly).
    """
    try:
        common = os_common_path([root, candidate])
    except ValueError as exc:
        # commonpath raises ValueError on different drives (Windows) or empty.
        raise SecurityViolationError(
            f"Path '{original}' escapes workspace root {root}: "
            f"{exc}"
        ) from exc

    if Path(common) != root:
        raise SecurityViolationError(
            f"Path '{original}' resolves to '{candidate}', which is outside "
            f"the allowed workspace '{root}'."
        )


def os_common_path(paths: list[Path]) -> str:
    """Thin wrapper around ``os.path.commonpath`` for type clarity."""
    return os.path.commonpath([str(p) for p in paths])