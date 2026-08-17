import enum
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class AgentDisplayMode(str, enum.Enum):
    """How much tool-call activity is printed to the end user during NLP turns.

    VERBOSE (default) — current behaviour: every call prints ``[tool] name(args)``
    and its full (truncated) result ``[result] ...``.
    CLEAN   — a short human-readable reason precedes each call, results are
    summarized for display only; the model always receives the full payload.
    QUIET   — tool calls/results are hidden from stdout; only the final answer
    is printed (useful when piping output or running headless).
    """

    VERBOSE = "verbose"
    CLEAN = "clean"
    QUIET = "quiet"


@dataclass(frozen=True)
class AgentSettings:
    """Immutable configuration settings for the agent."""

    workspace_root: Path = field(default_factory=lambda: Path.cwd())
    llm_api_url: str = "http://localhost:1234/v1"
    max_concurrent_tools: int = 5
    search_command_timeout_sec: float = 30.0
    compilation_check_timeout_sec: float = 30.0
    display_mode: AgentDisplayMode = field(
        default_factory=lambda: _parse_display_mode(os.environ.get("AGENT_DISPLAY_MODE"))
    )
    #: LLM provider selection (decision #009/#010): lmstudio (default) or
    #: opencode.  A model prefix ("opencode-go/...") overrides this.
    llm_provider: str = field(
        default_factory=lambda: os.environ.get("AGENT_LLM_PROVIDER", "lmstudio").strip().lower()
    )
    #: opencode server connection (decision #008 configured provider).
    opencode_server_url: str = field(
        default_factory=lambda: os.environ.get("OPENCODE_SERVER_URL", "http://127.0.0.1:4096")
    )
    opencode_password: str = field(
        default_factory=lambda: os.environ.get("OPENCODE_SERVER_PASSWORD", "")
    )
    opencode_model: str = field(
        default_factory=lambda: os.environ.get(
            "AGENT_OPENCODE_MODEL", "opencode-go/deepseek-v4-flash"
        )
    )
    #: Direct hosted API mode (OpenAI-compatible, the opencode-go gateway).
    #: When a key is available (OPENCODE_API_KEY or opencode's auth.json
    #: store) the provider uses direct API mode with NATIVE tool calling
    #: instead of a local opencode server.
    opencode_api_url: str = field(
        default_factory=lambda: os.environ.get("OPENCODE_API_URL", "https://opencode.ai/zen/go/v1")
    )
    opencode_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENCODE_API_KEY", "")
    )


_LLM_PROVIDERS = ("lmstudio", "opencode")


def _store_secret(name: str) -> str:
    """Secure-store lookup (OS keyring / encrypted file) without env fallback.

    Used as the last tier in secret resolution: env var -> .env -> store
    (plan OPS item 4).  Never raises — settings load must not break.
    """
    try:
        from agent_core.security.secrets import get_secret
        return get_secret(name, "")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Secret store lookup failed for %s: %s", name, exc)
        return ""


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

    if settings.llm_provider not in _LLM_PROVIDERS:
        raise ConfigurationError(
            f"llm_provider must be one of {', '.join(_LLM_PROVIDERS)}, got '{settings.llm_provider}'"
        )


def _load_env_file(env_path: Path | None = None) -> dict[str, str]:
    """Load environment variables from a .env file into a dictionary."""
    env_vars: dict[str, str] = {}

    if env_path is None:
        search_dir = Path.cwd()
        while True:
            candidate = search_dir / ".env"
            if candidate.is_file():
                env_path = candidate
                break
            parent = search_dir.parent
            if parent == search_dir:
                break
            search_dir = parent

    if env_path and env_path.is_file():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip("\"'")
                        if key:
                            env_vars[key] = value
        except OSError as exc:
            logger.warning("Failed to read .env file %s: %s", env_path, exc)

    return env_vars


def _parse_int(value: str | None, default: int) -> int:
    """Safely parse an integer from a string."""
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        logger.warning("Invalid integer value '%s', using default %d", value, default)
        return default


def _parse_float(value: str | None, default: float) -> float:
    """Safely parse a float from a string."""
    if value is None:
        return default
    try:
        return float(value.strip())
    except ValueError:
        logger.warning("Invalid float value '%s', using default %f", value, default)
        return default


def _parse_display_mode(value: str | None, default: AgentDisplayMode = AgentDisplayMode.VERBOSE) -> AgentDisplayMode:
    """Safely parse the display mode from a string."""
    if value is None:
        return default
    try:
        return AgentDisplayMode(value.strip().lower())
    except ValueError:
        logger.warning(
            "Invalid AGENT_DISPLAY_MODE '%s', using default %s", value, default.value
        )
        return default


def load_agent_settings(env_path: Path | None = None) -> AgentSettings:
    """Load agent settings from environment variables with type-safe defaults.

    Environment variable mapping:
        AGENT_WORKSPACE_ROOT -> workspace_root (Path)
        AGENT_LLM_API_URL -> llm_api_url (str)
        AGENT_MAX_CONCURRENT_TOOLS -> max_concurrent_tools (int)
        AGENT_SEARCH_COMMAND_TIMEOUT_SEC -> search_command_timeout_sec (float)
        AGENT_COMPILATION_CHECK_TIMEOUT_SEC -> compilation_check_timeout_sec (float)
        AGENT_DISPLAY_MODE -> display_mode (AgentDisplayMode: verbose|clean|quiet, default verbose)
    """
    env_vars = _load_env_file(env_path)

    merged: dict[str, str] = {**env_vars}
    for key in ("AGENT_WORKSPACE_ROOT", "AGENT_LLM_API_URL",
                "AGENT_MAX_CONCURRENT_TOOLS", "AGENT_SEARCH_COMMAND_TIMEOUT_SEC",
                "AGENT_COMPILATION_CHECK_TIMEOUT_SEC"):
        if key in os.environ:
            merged[key] = os.environ[key]

    # AGENT_DISPLAY_MODE may also live in the .env file (merged already covers it).
    display_mode_raw = os.environ.get("AGENT_DISPLAY_MODE") or env_vars.get("AGENT_DISPLAY_MODE")

    workspace_root_str = merged.get("AGENT_WORKSPACE_ROOT")
    workspace_root = Path(workspace_root_str) if workspace_root_str else Path.cwd()

    settings = AgentSettings(
        workspace_root=workspace_root,
        llm_api_url=merged.get("AGENT_LLM_API_URL", "http://localhost:1234/v1"),
        max_concurrent_tools=_parse_int(merged.get("AGENT_MAX_CONCURRENT_TOOLS"), 5),
        search_command_timeout_sec=_parse_float(merged.get("AGENT_SEARCH_COMMAND_TIMEOUT_SEC"), 30.0),
        compilation_check_timeout_sec=_parse_float(merged.get("AGENT_COMPILATION_CHECK_TIMEOUT_SEC"), 30.0),
        display_mode=_parse_display_mode(display_mode_raw, AgentDisplayMode.VERBOSE),
        llm_provider=(os.environ.get("AGENT_LLM_PROVIDER") or env_vars.get("AGENT_LLM_PROVIDER") or "lmstudio").strip().lower(),
        opencode_server_url=os.environ.get("OPENCODE_SERVER_URL") or env_vars.get("OPENCODE_SERVER_URL") or "http://127.0.0.1:4096",
        opencode_password=os.environ.get("OPENCODE_SERVER_PASSWORD") or env_vars.get("OPENCODE_SERVER_PASSWORD") or _store_secret("OPENCODE_SERVER_PASSWORD"),
        opencode_model=os.environ.get("AGENT_OPENCODE_MODEL") or env_vars.get("AGENT_OPENCODE_MODEL") or "opencode-go/deepseek-v4-flash",
        opencode_api_url=os.environ.get("OPENCODE_API_URL") or env_vars.get("OPENCODE_API_URL") or "https://opencode.ai/zen/go/v1",
        opencode_api_key=os.environ.get("OPENCODE_API_KEY") or env_vars.get("OPENCODE_API_KEY") or _store_secret("OPENCODE_API_KEY"),
    )

    _validate_settings(settings)
    return settings


DEFAULT_SETTINGS: Final[AgentSettings] = AgentSettings()

try:
    _validate_settings(DEFAULT_SETTINGS)
except ConfigurationError as exc:
    logger.error("Default agent settings validation failed: %s", exc)
    raise