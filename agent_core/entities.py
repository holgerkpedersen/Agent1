"""
Core entities for the Agent Framework.

This module serves as the single source of truth for all domain models,
configuration dataclasses, exception hierarchies, and shared type aliases.
All downstream modules must import from this file; zero redefinition is allowed.

Target: Python 3.10+ (PEP 604 unions, modern pathlib, strict typing)
"""

import contextvars
import json
import logging
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar

# =========================================================================== #
# Exception Hierarchy
# =========================================================================== #

class AgentError(Exception):
    """Base exception for all agent-related errors."""

    def __init__(self, message: str = "An unexpected agent error occurred", details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class FileOperationError(AgentError):
    """Raised when a file system operation fails."""


class ToolExecutionError(AgentError):
    """Raised when an external tool execution fails."""


class SecurityViolationError(AgentError):
    """Raised when a security boundary or sandbox constraint is violated."""


class SemanticIndexError(AgentError):
    """Raised when semantic search or indexing operations fail."""


# =========================================================================== #
# Configuration Models (Frozen Dataclasses)
# =========================================================================== #

@dataclass(frozen=True)
class FileSystemConfig:
    """Configuration for workspace and file system behavior."""
    workspace_root: Path = field(default_factory=lambda: Path.cwd())
    follow_symlinks: bool = True
    max_file_size_bytes: int = 10 * 1024 * 1024  # 10MB default


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for Large Language Model interactions."""
    model_name: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout_seconds: float = 30.0


@dataclass(frozen=True)
class AgentConfig:
    """Top-level configuration aggregating subsystem settings."""
    fs_config: FileSystemConfig = field(default_factory=FileSystemConfig)
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    log_level: int = logging.INFO
    enable_sandbox: bool = True


# =========================================================================== #
# Type Aliases & Protocols
# =========================================================================== #

T = TypeVar("T")

class Serializable(Protocol):
    """Protocol for objects that can be safely serialized to JSON."""
    def as_dict(self) -> dict[str, Any]: ...


# =========================================================================== #
# Shared Context Variables
# =========================================================================== #

CORRELATION_ID_CTX: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=str(uuid.uuid4())
)


# =========================================================================== #
# Utilities
# =========================================================================== #

def safe_json_dumps(obj: Any, **kwargs: Any) -> str:
    """Serialize an object to JSON with fallback string conversion for unknown types."""
    def _default_encoder(o: Any) -> Any:
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    return json.dumps(obj, default=_default_encoder, **kwargs)


def enrich_exception_with_traceback(exc: Exception) -> None:
    """Safely attach a formatted traceback string to an exception for structured logging."""
    exc.traceback_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))  # type: ignore[attr-defined]


# =========================================================================== #
# Public API Surface
# =========================================================================== #

__all__ = [
    # Exceptions
    "AgentError",
    "FileOperationError",
    "ToolExecutionError",
    "SecurityViolationError",
    "SemanticIndexError",
    # Configs
    "FileSystemConfig",
    "LLMConfig",
    "AgentConfig",
    # Context & Types
    "CORRELATION_ID_CTX",
    "Serializable",
    # Utilities
    "safe_json_dumps",
    "enrich_exception_with_traceback",
]