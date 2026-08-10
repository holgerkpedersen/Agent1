
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentSettings:
    """Immutable configuration settings for the agent."""

    workspace_root: Path = field(default_factory=lambda: Path.cwd())
    llm_api_url: str = "http://localhost:1234/v1"
    max_concurrent_tools: int = 5
    search_command_timeout_sec: float = 30.0
    compilation_check_timeout_sec: float = 30.0


def _validate_settings(settings: AgentSettings) -> None:
    """Validate the given settings and raise ConfigurationError if invalid."""

    if not isinstance(settings.workspace_root, Path):
        raise ConfigurationError(
            f"workspace_root must be a Path instance, got {type(settings.workspace_root).__name__}"
        )

    if not settings.workspace_root.exists():
        raise ConfigurationError(
            f"Workspace root does not exist: {settings.workspace_root}"
        )

    if settings.max_concurrent_tools <= 0:
        raise ConfigurationError("max_concurrent_tools must be positive")


DEFAULT_SETTINGS: Final[AgentSettings] = AgentSettings()

try:
    _validate_settings(DEFAULT_SETTINGS)
except ConfigurationError as exc:
    logger.error("Default agent settings validation failed: %s", exc)
    raise