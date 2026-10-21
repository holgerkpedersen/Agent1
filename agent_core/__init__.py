"""Agent core package — exceptions, configs, path utilities."""
from .entities import (
    AgentError,
    FileOperationError,
    SecurityViolationError,
    ToolExecutionError,
    SemanticIndexError,
    FileSystemConfig,
    LLMConfig,
    AgentConfig,
    Serializable,
    safe_json_dumps,
    enrich_exception_with_traceback,
)
from .context_management import CORRELATION_ID_CTX
from .path_utils import (
    normalize_path, to_windows_path, workspace_path, WorkspaceSandbox
)
from .constants import KNOWN_MODELS, DEFAULT_MODEL
from .file_system import FileSystem
from .file_searcher import FileSearcher
from .tool_dispatcher import ToolDispatcher

__all__ = [
    "AgentError",
    "FileOperationError",
    "SecurityViolationError",
    "ToolExecutionError",
    "SemanticIndexError",
    "FileSystemConfig",
    "LLMConfig",
    "AgentConfig",
    "CORRELATION_ID_CTX",
    "Serializable",
    "safe_json_dumps",
    "enrich_exception_with_traceback",
    "normalize_path",
    "to_windows_path",
    "workspace_path",
    "WorkspaceSandbox",
    "KNOWN_MODELS",
    "DEFAULT_MODEL",
    "FileSystem",
    "FileSearcher",
    "ToolDispatcher",
    "Agent",
    "AgentCore",
]


def __getattr__(name):
    """Lazy exports: ``Agent``/``AgentCore`` pull in the full agent stack.

    Resolved on first access only, so lightweight submodule imports
    (``import agent_core.path_utils`` etc.) never drag in the LLM layer
    or risk circular imports through this package root.
    """
    if name in ("Agent", "AgentCore"):
        from .agent import Agent, AgentCore

        return Agent if name == "Agent" else AgentCore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
