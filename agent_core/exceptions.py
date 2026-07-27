from __future__ import annotations
from typing import Final

class AgentBaseError(Exception):
    """Base exception for all agent-related errors."""


class FileOperationError(AgentBaseError):
    def __init__(self, message: str, path: str | None = None) -> None:
        self.path = path
        super().__init__(message)


class SecurityViolationError(AgentBaseError):
    pass


class ConfigurationError(AgentBaseError):
    pass


class ToolExecutionError(AgentBaseError): 
    def __init__(self, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' failed: {message}")


# Type alias for consistent error handling
ErrorHandlerResult = tuple[bool, str | None]  # (success, optional_error_message)

__all__: Final[list[str]] = [
    "AgentBaseError",
    "FileOperationError",
    "SecurityViolationError",
    "ConfigurationError",
    "ToolExecutionError",
    "ErrorHandlerResult",
]