"""
path_utils.py
Strict workspace sandboxing, cross-platform normalization, and structured exception hierarchy.
Aligns with [PATH-01], [EXC-01], and security hardening requirements.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union

# ---------------------------------------------------------------------------
# 🛡️ Exception Hierarchy ([EXC-01])
# ---------------------------------------------------------------------------

class AgentError(Exception):
    """Base exception for all agent framework errors."""
    pass


class FileOperationError(AgentError):
    """Raised on read/write/delete failures, encoding issues, or size limits."""
    def __init__(self, path: Path | str, message: str = "File operation failed"):
        self.path = Path(path) if isinstance(path, str) else path
        super().__init__(f"{message}: {self.path}")


class SecurityViolationError(AgentError):
    """Raised on path traversal attempts, sandbox escapes, or unsafe inputs."""
    def __init__(self, attempted_path: str, reason: str = "Path outside workspace boundary"):
        self.attempted_path = attempted_path
        super().__init__(f"Security violation ({reason}): {attempted_path}")


class ToolExecutionError(AgentError):
    """Raised when a tool call fails validation or runtime execution."""
    def __init__(self, tool_name: str, message: str = "Tool execution failed"):
        self.tool_name = tool_name
        super().__init__(f"[{tool_name}] {message}")


# ---------------------------------------------------------------------------
# 📦 Path Validation & Normalization ([PATH-01])
# ---------------------------------------------------------------------------

def _validate_path(raw: str, workspace_root: Path) -> Path:
    """
    Strictly validates and resolves a path within the workspace sandbox.
    Raises SecurityViolationError on any escape attempt (.., symlinks, absolute jumps).
    """
    if not raw or not isinstance(raw, str):
        raise FileOperationError("", "Empty path provided")

    # Normalize slashes to OS standard for consistent resolution
    normalized_str = raw.replace("/", os.sep).replace("\\", os.sep)
    target_path = Path(normalized_str)

    # If relative, anchor to workspace root before resolving
    if not target_path.is_absolute():
        target_path = workspace_root / target_path

    try:
        resolved_target = target_path.resolve(strict=False)
        resolved_workspace = workspace_root.resolve()

        ws_str = str(resolved_workspace)
        # Prevent prefix collision attacks (e.g., /workspace matches /workspace_evil)
        if not str(resolved_target).startswith(ws_str + os.sep) and resolved_target != resolved_workspace:
            raise SecurityViolationError(raw, "Path escapes workspace boundary")

        return resolved_target
    except (ValueError, OSError) as e:
        if isinstance(e, SecurityViolationError):
            raise
        raise FileOperationError(raw, f"Failed to resolve path: {e}")


def _normalize_to_posix(path_input: Union[str, Path]) -> str:
    """
    Converts paths to POSIX-style strings for consistent LLM consumption & storage.
    Cross-platform safe via pathlib abstraction.
    """
    p = Path(str(path_input))
    return str(p.as_posix())


def _resolve_workspace_path(raw: str, workspace_root: Path) -> Path:
    """
    Unified path resolver handling POSIX/Windows slash variations.
    Single source of truth for all agent path operations.
    """
    # Strip surrounding whitespace and quotes that LLMs sometimes emit
    cleaned = raw.strip().strip("\"'")
    return _validate_path(cleaned, workspace_root)