"""Safe file read/write operations using workspace sandboxing."""

from pathlib import Path

import logging

from agent_core.security.path_utils import (
    SecurityViolationError,
    normalize_path,
)

_file_logger = logging.getLogger(__name__)


def read_file(workspace_root: Path, target_path: str) -> str:
    """Read a file safely inside *workspace_root*.

    Parameters
    ----------
    workspace_root : Path
        The root directory that all operations are sandboxed to.
    target_path : str
        User-supplied path (relative or absolute). Will be resolved and
        validated against ``workspace_root``.

    Returns
    -------
    str
        The text contents of the file.

    Raises
    ------
    SecurityViolationError
        If *target_path* resolves outside *workspace_root*.
    FileNotFoundError
        If the normalised path does not exist.
    """
    safe_path = normalize_path(workspace_root, target_path)

    if not safe_path.is_file():
        raise FileNotFoundError(f"Not a file: {safe_path}")

    _file_logger.debug("Reading file %s", safe_path)
    return safe_path.read_text(encoding="utf-8")


def write_file(workspace_root: Path, target_path: str, content: str) -> None:
    """Write *content* to a file safely inside *workspace_root*.

    Parameters
    ----------
    workspace_root : Path
        The root directory that all operations are sandboxed to.
    target_path : str
        User-supplied path (relative or absolute). Will be resolved and
        validated against ``workspace_root``.
    content : str
        Text content to write.

    Raises
    ------
    SecurityViolationError
        If *target_path* resolves outside *workspace_root*.
    """
    safe_path = normalize_path(workspace_root, target_path)

    # Ensure parent directories exist (still within the sandbox).
    safe_path.parent.mkdir(parents=True, exist_ok=True)

    _file_logger.debug("Writing file %s (%d bytes)", safe_path, len(content))
    safe_path.write_text(content, encoding="utf-8")