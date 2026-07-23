"""
agent_core - Core framework for AI agent execution, workspace sandboxing, and observability.

Provides secure path validation, async-safe correlation tracking, structured logging,
and centralized configuration/exception management.

Version: 1.0.0
Python Target: >= 3.10

Usage Example:
    >>> from pathlib import Path
    >>> from agent_core import (
    ...     WorkspaceSandbox, validate_path, setup_logging, CorrelationIdContext, AgentConfig
    ... )
    >>> setup_logging()
    >>> config = AgentConfig.default()
    >>> with CorrelationIdContext("req-123") as ctx_id:
    ...     sandbox = WorkspaceSandbox(Path("./workspace"))
    ...     safe_path = validate_path("./data/file.txt", sandbox.root)
"""

from .entities import (
    AgentConfig,
    AgentError,
    FileOperationError,
    FileSystemConfig,
    LLMConfig,
    SemanticIndexError,
    SecurityViolationError,
    ToolExecutionError,
)
from .path_utils import WorkspaceSandbox, _validate_path as validate_path
from .context_management import CorrelationIdContext
from .logging_config import setup_logging

__all__ = [
    "AgentError",
    "FileOperationError",
    "ToolExecutionError",
    "SecurityViolationError",
    "SemanticIndexError",
    "validate_path",
    "WorkspaceSandbox",
    "CorrelationIdContext",
    "setup_logging",
    "AgentConfig",
    "FileSystemConfig",
    "LLMConfig",
]

__version__ = "1.0.0"