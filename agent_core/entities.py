"""Core entity definitions for the agent system.

This module provides configuration dataclasses and serialization helpers used
across the agent infrastructure. Exception hierarchies have been consolidated
into ``agent_core/exceptions`` to avoid duplicate/conflicting signatures; this
module re-exports them so existing import paths remain valid.
"""


import json
from typing import Any, Protocol

# Re-export canonical exceptions from the single source-of-truth module.
from agent_core.exceptions import (  # noqa: F401
    AgentBaseError as AgentError,
    ConfigurationError,
    FileOperationError,
    SecurityViolationError,
    ToolExecutionError,
    SemanticIndexError,
)


class Serializable(Protocol):
    """Structural protocol for objects that can be serialized to dicts."""

    def as_dict(self) -> dict[str, Any]: ...


def _default_encoder(o: Any) -> Any:
    """Fallback encoder used by ``safe_json_dumps`` for non-native types."""
    if hasattr(o, "as_dict"):
        return o.as_dict()
    if isinstance(o, Exception):
        return {"type": type(o).__name__, "message": str(o)}
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def safe_json_dumps(obj: Any, **kwargs: Any) -> str:
    """Serialize ``obj`` to a JSON string with graceful fallback handling."""
    return json.dumps(obj, default=_default_encoder, **kwargs)


def enrich_exception_with_traceback(exc: Exception) -> None:
    """Attach traceback context to an exception for richer diagnostics."""
    import traceback

    exc.__traceback__ = traceback.format_exc()  # type: ignore[attr-defined]
    if not hasattr(exc, "_agent_trace"):
        setattr(exc, "_agent_trace", traceback.format_exc())


class LLMConfig:
    """Configuration container for LLM client settings."""

    def __init__(self) -> None:
        self.model_name: str = "gpt-3.5-turbo"
        self.temperature: float = 0.7
        self.max_tokens: int = 2048


class FileSystemConfig:
    """Configuration container for filesystem/workspace settings."""

    def __init__(self) -> None:
        self.workspace: str = "."
        self.allow_create: bool = True
        self.encoding: str = "utf-8"


class AgentConfig(Serializable):
    """Top-level configuration aggregating LLM and filesystem sub-configs."""

    def __init__(self, workspace: str | None = None) -> None:
        self.llm: LLMConfig = LLMConfig()
        self.fs: FileSystemConfig = FileSystemConfig()
        if workspace is not None:
            self.fs.workspace = workspace

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary representation of the config."""
        return {
            "llm": {"model_name": self.llm.model_name},
            "fs": {"workspace": self.fs.workspace},
        }


__all__ = [
    "AgentConfig",
    "AgentError",
    "ConfigurationError",
    "FileOperationError",
    "FileSystemConfig",
    "LLMConfig",
    "SemanticIndexError",
    "SecurityViolationError",
    "Serializable",
    "ToolExecutionError",
    "enrich_exception_with_traceback",
    "safe_json_dumps",
]