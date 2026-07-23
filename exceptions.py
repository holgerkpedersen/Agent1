"""
exceptions.py
Structured exception hierarchy for the agent framework.
Replaces fragile string-based error returns with typed exceptions 
for predictable control flow, safer LLM interaction, and robust logging.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Optional


class AgentError(Exception):
    """
    Base exception class for all agent framework errors.
    
    Catching `AgentError` allows callers to handle any framework-specific 
    failure uniformly while preserving the specific subclass context.
    """
    pass


class FileOperationError(AgentError):
    """
    Raised when file I/O operations fail (read, write, delete, stat).
    
    Attributes:
        path: The target file or directory path involved in the operation.
        raw_error: The underlying OS or encoding exception that triggered this error.
    """
    def __init__(
        self, 
        path: str | Path, 
        message: str = "File operation failed", 
        raw_error: Optional[Exception] = None
    ):
        self.path = Path(path) if isinstance(path, str) else path
        self.raw_error = raw_error
        
        error_context = f" (caused by {type(raw_error).__name__}: {raw_error})" if raw_error else ""
        super().__init__(f"{message}: '{self.path}'{error_context}")


class ToolExecutionError(AgentError):
    """
    Raised when a tool call fails validation, execution, or returns an unexpected state.
    
    Attributes:
        tool_name: Identifier of the tool that failed (e.g., 'read_file', 'search').
        raw_error: The underlying exception or error context from the tool logic.
    """
    def __init__(
        self, 
        tool_name: str, 
        message: str = "Tool execution failed", 
        raw_error: Optional[Exception] = None
    ):
        self.tool_name = tool_name
        self.raw_error = raw_error
        
        error_context = f" (caused by {type(raw_error).__name__}: {raw_error})" if raw_error else ""
        super().__init__(f"[{tool_name}] {message}{error_context}")


class PathEscapeError(AgentError):
    """
    Raised when a path resolution attempt tries to escape the sandboxed workspace boundary.
    
    Attributes:
        attempted_path: The original unsafe path string provided by the LLM or user.
        workspace_root: The resolved boundary that was violated (if known).
    """
    def __init__(
        self, 
        attempted_path: str, 
        workspace_root: Optional[str | Path] = None, 
        message: str = "Path traversal outside workspace boundary"
    ):
        self.attempted_path = attempted_path
        self.workspace_root = Path(workspace_root) if isinstance(workspace_root, str) else workspace_root
        
        context = f" (workspace: '{self.workspace_root}')" if self.workspace_root else ""
        super().__init__(f"{message}: '{attempted_path}'{context}")


class AgentTimeoutError(AgentError, TimeoutError):
    """
    Raised when an async operation exceeds its allocated timeout budget.
    
    Inherits from both `AgentError` and built-in `TimeoutError` to support 
    generic timeout catching while remaining part of the agent hierarchy.
    
    Attributes:
        operation: Description of the timed-out action (e.g., 'LLM call', 'file read').
        timeout_seconds: The duration limit that was exceeded.
        raw_error: The underlying httpx or asyncio timeout exception.
    """
    def __init__(
        self, 
        operation: str = "unknown operation", 
        timeout_seconds: float = 0.0, 
        raw_error: Optional[Exception] = None
    ):
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        self.raw_error = raw_error
        
        super().__init__(
            f"Timeout during '{operation}' after {timeout_seconds}s"
        )


# ---------------------------------------------------------------------------
# Helper Utilities for Exception Context Enrichment
# ---------------------------------------------------------------------------

def enrich_with_traceback(exc: Exception) -> None:
    """
    Attaches the current traceback string to an exception instance 
    before re-raising or logging, aiding in async debugging.
    """
    if not hasattr(exc, '__traceback__'):
        exc.__traceback__ = traceback.extract_stack()  # type: ignore[attr-defined]