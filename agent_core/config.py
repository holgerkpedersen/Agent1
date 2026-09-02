from __future__ import annotations
import enum
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

#: Default LM Studio server port (OpenAI-compatible + management REST API).
DEFAULT_LMSTUDIO_PORT: Final[int] = 1234


def lmstudio_port() -> int:
    """LM Studio server port.

    Single source of truth: ``LMSTUDIO_PORT`` env var (or ``.env`` entry),
    falling back to :data:`DEFAULT_LMSTUDIO_PORT`.  Invalid values warn and
    use the default — settings resolution must never raise.
    """
    raw = os.environ.get("LMSTUDIO_PORT")
    if not raw:
        return DEFAULT_LMSTUDIO_PORT
    try:
        return int(raw.strip())
    except ValueError:
        logger.warning(
            "Invalid LMSTUDIO_PORT '%s', using default %d", raw, DEFAULT_LMSTUDIO_PORT
        )
        return DEFAULT_LMSTUDIO_PORT


def lmstudio_base_url() -> str:
    """LM Studio OpenAI-compatible base URL (e.g. ``http://localhost:1234/v1``).

    Resolution order:
      1. ``LMSTUDIO_URL`` — explicit full-URL override (backward compatible).
      2. ``LMSTUDIO_PORT`` via :func:`lmstudio_port` — port-only override.
      3. :data:`DEFAULT_LMSTUDIO_PORT`.
    """
    url = os.environ.get("LMSTUDIO_URL")
    if url:
        return url.strip().rstrip("/")
    return f"http://localhost:{lmstudio_port()}/v1"


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
    llm_api_url: str = field(default_factory=lmstudio_base_url)
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
    #: Ordered LLM provider fallback chain (decision #013).  When the active
    #: provider loses connectivity, the agent fails over to the next entry in
    #: this list (e.g. ``lmstudio,opencode`` -> local first, hosted on outage).
    #: Defaults to a single-element list built from ``AGENT_LLM_PROVIDER`` so
    #: existing single-provider setups keep working unchanged.  Set
    #: ``AGENT_LLM_PROVIDERS`` (comma-separated) to enable multi-provider
    #: failover.
    llm_providers: tuple[str, ...] = field(
        default_factory=lambda: _parse_provider_chain(
            os.environ.get("AGENT_LLM_PROVIDERS")
            or os.environ.get("AGENT_LLM_PROVIDER")
            or "lmstudio"
        )
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
    #: llama.cpp (llama-server) OpenAI-compatible endpoint (decision #014).
    #: When AGENT_LLM_PROVIDER=llama the agent talks to a local llama.cpp
    #: ``llama-server`` over the standard /v1/chat/completions protocol
    #: instead of LM Studio.  Override with AGENT_LLAMA_URL.
    llama_base_url: str = field(
        default_factory=lambda: os.environ.get("AGENT_LLAMA_URL", "http://127.0.0.1:8080/v1")
    )
    #: Extra CLI args passed to ``llama-server`` when the agent auto-launches
    #: or relaunches it (e.g. "--gpu-layers 999 --ctx-size 262144").  These are
    #: appended verbatim after the agent's own --host/--port/--model/--alias
    #: flags so a manual launch's tuning survives a model switch.  Space-
    #: separated string; override with AGENT_LLAMA_EXTRA_ARGS.  Empty = none.
    llama_extra_args: str = field(
        default_factory=lambda: os.environ.get("AGENT_LLAMA_EXTRA_ARGS", "")
    )
    #: OpenRouter hosted gateway (OpenAI-compatible, native tool calling).
    #: When AGENT_LLM_PROVIDER=openrouter (or a model is prefixed openrouter/)
    #: the agent talks to OpenRouter over the standard /v1/chat/completions
    #: protocol instead of LM Studio / opencode / llama.cpp.  Override the base
    #: URL with OPENROUTER_API_URL (e.g. a proxy); the key resolves from
    #: OPENROUTER_API_KEY first, then the secure store.
    openrouter_api_url: str = field(
        default_factory=lambda: os.environ.get("OPENROUTER_API_URL", "https://openrouter.ai/api/v1")
    )
    openrouter_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", "") or _store_secret("OPENROUTER_API_KEY")
    )
    #: Default OpenRouter model (persisted model choice in model.json wins).
    #: The owner only uses the FREE tier, so the default is a free model; the
    #: user overrides it with `model openrouter/<vendor>/<model>` or the
    #: AGENT_OPENROUTER_MODEL env var.  (Free models carry a ``:free`` suffix —
    #: `model list` only shows free models by default.)
    openrouter_model: str = field(
        default_factory=lambda: os.environ.get(
            "AGENT_OPENROUTER_MODEL", "openrouter/meta-llama/llama-3.1-8b-instruct:free"
        )
    )
    #: Default opencode-zen FREE model used for catalog listing / probing
    #: (keyless tier — no API key needed).  NO specific model is hardcoded (a
    #: machine/account may not have it); when unset we construct the keyless
    #: catalog probe with a generic zen-prefixed placeholder.  Override with
    #: AGENT_ZEN_FREE_DEFAULT.
    zen_free_default: str = field(
        default_factory=lambda: os.environ.get("AGENT_ZEN_FREE_DEFAULT", "")
    )
    #: Ordered fallback models for the opencode-zen FREE tier, tried in order
    #: when the user's chosen free model is temporarily unavailable on the
    #: backend.  Comma-separated; override with AGENT_ZEN_FREE_FALLBACKS.  When
    #: unset the live catalog is discovered at retry time (no hardcoded list).
    zen_free_fallbacks: tuple[str, ...] = field(
        default_factory=lambda: _parse_zen_fallbacks(
            os.environ.get("AGENT_ZEN_FREE_FALLBACKS")
        )
    )

    def __post_init__(self) -> None:
        # Enforce the invariant on every construction (not just at the two
        # load entry points) so misconfigured settings fail fast everywhere.
        _validate_settings(self)


_LLM_PROVIDERS = ("lmstudio", "opencode", "llama", "openrouter")


def _parse_provider_chain(raw: str | None) -> tuple[str, ...]:
    """Parse a comma-separated provider list into a clean ordered tuple.

    Strips whitespace, lowercases each entry, and drops empties so
    ``"lmstudio, opencode"`` and ``"lmstudio,,opencode,"`` both yield
    ``("lmstudio", "opencode")``.  Falls back to ``("lmstudio",)`` when the
    input is empty/None.  Validation against :data:`_LLM_PROVIDERS` happens
    later in :func:`_validate_settings`.
    """
    if not raw:
        return ("lmstudio",)
    cleaned = tuple(
        part.strip().lower() for part in raw.split(",") if part.strip()
    )
    return cleaned or ("lmstudio",)


def _parse_zen_fallbacks(raw: str | None) -> tuple[str, ...]:
    """Parse a comma-separated opencode-zen FREE fallback list.

    Strips whitespace and drops empties so ``"opencode-zen/hy3-free,
    opencode-zen/laguna-s-2.1-free"`` yields a clean ordered tuple.  Returns
    an empty tuple when the input is empty/None — no model is assumed to
    exist; the retry path discovers the live catalog instead.
    """
    if not raw:
        return ()
    cleaned = tuple(
        part.strip() for part in raw.split(",") if part.strip()
    )
    return cleaned


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

    invalid_providers = tuple(
        p.split(":", 1)[0].strip()
        for p in settings.llm_providers
        if p.split(":", 1)[0].strip() not in _LLM_PROVIDERS
    )
    if invalid_providers:
        raise ConfigurationError(
            f"llm_providers must contain only {', '.join(_LLM_PROVIDERS)} "
            f"(optionally 'provider:model' per entry), "
            f"got {', '.join(settings.llm_providers)} "
            f"(invalid: {', '.join(invalid_providers)})"
        )

    # The single-provider setting must match the first entry in the chain (the
    # provider part of a possible 'provider:model' entry) so the "active"
    # provider and the failover order agree.
    if settings.llm_providers:
        first_provider = settings.llm_providers[0].split(":", 1)[0].strip()
        if settings.llm_provider != first_provider:
            raise ConfigurationError(
                f"llm_provider ('{settings.llm_provider}') must match the first "
                f"entry of llm_providers ({settings.llm_providers[0]})"
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

    # The failover chain drives the active provider: its first entry IS the
    # active provider, so llm_provider is derived from it (never diverges).
    # AGENT_LLM_PROVIDERS wins; otherwise fall back to the single-provider
    # AGENT_LLM_PROVIDER setting; finally the default cloud-first / local-
    # fallback chain (decision #013 variant): opencode-zen free tier (primary
    # cloud), opencode-go (secondary cloud), LM Studio (primary local), then
    # llama.cpp (secondary local).  Per-entry "provider:model" overrides let
    # the same provider appear twice in different modes (zen vs go).
    DEFAULT_LLM_CHAIN = (
        "opencode:opencode-zen/hy3-free,"
        "opencode:opencode-go/deepseek-v4-flash,"
        "lmstudio,llama"
    )
    raw_chain = (
        os.environ.get("AGENT_LLM_PROVIDERS")
        or env_vars.get("AGENT_LLM_PROVIDERS")
        or os.environ.get("AGENT_LLM_PROVIDER")
        or env_vars.get("AGENT_LLM_PROVIDER")
        or DEFAULT_LLM_CHAIN
    )
    llm_providers = _parse_provider_chain(raw_chain)
    llm_provider = llm_providers[0].split(":", 1)[0].strip()

    settings = AgentSettings(
        workspace_root=workspace_root,
        llm_api_url=merged.get("AGENT_LLM_API_URL") or lmstudio_base_url(),
        max_concurrent_tools=_parse_int(merged.get("AGENT_MAX_CONCURRENT_TOOLS"), 5),
        search_command_timeout_sec=_parse_float(merged.get("AGENT_SEARCH_COMMAND_TIMEOUT_SEC"), 30.0),
        compilation_check_timeout_sec=_parse_float(merged.get("AGENT_COMPILATION_CHECK_TIMEOUT_SEC"), 30.0),
        display_mode=_parse_display_mode(display_mode_raw, AgentDisplayMode.VERBOSE),
        llm_provider=llm_provider,
        llm_providers=llm_providers,
        opencode_server_url=os.environ.get("OPENCODE_SERVER_URL") or env_vars.get("OPENCODE_SERVER_URL") or "http://127.0.0.1:4096",
        opencode_password=os.environ.get("OPENCODE_SERVER_PASSWORD") or env_vars.get("OPENCODE_SERVER_PASSWORD") or _store_secret("OPENCODE_SERVER_PASSWORD"),
        opencode_model=os.environ.get("AGENT_OPENCODE_MODEL") or env_vars.get("AGENT_OPENCODE_MODEL") or "opencode-go/deepseek-v4-flash",
        opencode_api_url=os.environ.get("OPENCODE_API_URL") or env_vars.get("OPENCODE_API_URL") or "https://opencode.ai/zen/go/v1",
        opencode_api_key=os.environ.get("OPENCODE_API_KEY") or env_vars.get("OPENCODE_API_KEY") or _store_secret("OPENCODE_API_KEY"),
        llama_base_url=os.environ.get("AGENT_LLAMA_URL") or env_vars.get("AGENT_LLAMA_URL") or "http://127.0.0.1:8080/v1",
        llama_extra_args=os.environ.get("AGENT_LLAMA_EXTRA_ARGS") or env_vars.get("AGENT_LLAMA_EXTRA_ARGS") or "",
        openrouter_api_url=os.environ.get("OPENROUTER_API_URL") or env_vars.get("OPENROUTER_API_URL") or "https://openrouter.ai/api/v1",
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY") or env_vars.get("OPENROUTER_API_KEY") or _store_secret("OPENROUTER_API_KEY"),
        openrouter_model=os.environ.get("AGENT_OPENROUTER_MODEL") or env_vars.get("AGENT_OPENROUTER_MODEL") or "openrouter/meta-llama/llama-3.1-8b-instruct:free",
    )

    _validate_settings(settings)
    return settings


DEFAULT_SETTINGS: Final[AgentSettings] = AgentSettings()

try:
    _validate_settings(DEFAULT_SETTINGS)
except ConfigurationError as exc:
    logger.error("Default agent settings validation failed: %s", exc)
    raise