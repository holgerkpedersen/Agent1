"""LLM Provider Protocol - abstract interface for LLM backends."""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

from agent_core.constants import (
    DEFAULT_LLAMA_BASE_URL,
    DEFAULT_OPENCODE_API_BASE,
    DEFAULT_OPENCODE_MODEL,
    DEFAULT_OPENCODE_SERVER_URL,
    DEFAULT_OPENROUTER_API_BASE,
    DEFAULT_OPENROUTER_MODEL,
    ROUTER,
)
from agent_core.config import _CHEAPEST_CLOUD_KEYWORD
from .pricing import cheapest_opencode_go_model, cost_per_token

logger = logging.getLogger(__name__)


#: Mapping from concrete provider class names to their routing type.
_PROVIDER_TYPE_BY_CLASS = {
    "LlamaProvider": "llama",
    "LMStudioProvider": "lmstudio",
    "OpencodeProvider": "opencode",
    "OpenRouterProvider": "openrouter",
}


def _provider_type(provider: Any) -> str:
    """Best-effort routing type for *provider* (for cost lookups)."""
    return _PROVIDER_TYPE_BY_CLASS.get(type(provider).__name__, "")


@dataclass(frozen=True)
class ResponseMetrics:
    """Per-call token/latency/cost accounting (plan ARCH item 14).

    Providers that expose usage data set ``provider.last_response_metrics``
    after every ``chat`` call so callers can report per-turn token/latency/
    cost without changing the ``chat`` return contract (which stays a str).
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


#: Attribute name providers set with the last call's metrics.
LAST_METRICS_ATTR = "last_response_metrics"


def get_last_metrics(provider: Any) -> ResponseMetrics | None:
    """Return the provider's last-call metrics, or None when unsupported."""
    if provider is None:
        return None
    metrics = getattr(provider, LAST_METRICS_ATTR, None)
    return metrics if isinstance(metrics, ResponseMetrics) else None


def provider_for(
    model_name: str | None,
    provider_setting: str = "lmstudio",
    persisted_provider: str | None = None,
) -> str:
    """Provider selection (decisions #007/#009/#010).

    Priority: an explicit model prefix wins ("opencode-go/..." or "opencode/..."
    → opencode; known LM Studio prefixes → lmstudio); then the provider the
    user last persisted in model.json (``model`` command writes it) — this
    keeps LM Studio models on LM Studio even when AGENT_LLM_PROVIDER is
    opencode; finally the configured ``llm_provider`` setting; unknown values
    fall back to lmstudio.
    """
    m = (model_name or "").lower()
    for prefix, provider in ROUTER.items():
        if m.startswith(prefix):
            return provider
    if persisted_provider in ("lmstudio", "opencode", "llama", "openrouter"):
        return persisted_provider
    # A chain entry may carry a per-entry model override ("opencode:model");
    # the provider part is everything before the first colon.
    provider_setting = provider_setting.split(":", 1)[0].strip()
    return provider_setting if provider_setting in ("lmstudio", "opencode", "llama", "openrouter") else "lmstudio"


def _split_entry(entry: str) -> tuple[str, str | None]:
    """Split a failover-chain entry into ``(provider, model_or_None)``.

    Supports an optional per-entry model override so the same provider can
    appear twice in different modes — e.g. ``opencode:opencode-zen/hy3-free``
    (keyless free tier) and ``opencode:opencode-go/deepseek-v4-flash`` (keyed
    tier) are BOTH the ``opencode`` provider but in different modes.  The model
    is split on the FIRST colon only, so OpenRouter model ids that contain
    ``:free`` are not mangled.  Returns ``(provider, None)`` when no override
    is present.
    """
    entry = entry.strip()
    if ":" in entry:
        provider, _, model = entry.partition(":")
        return provider.strip(), model.strip() or None
    return entry, None


def _provider_part(entry: str) -> str:
    """Return just the provider name from a chain entry (drops any ``:model``)."""
    return entry.split(":", 1)[0].strip()


def _model_mode(model: str | None) -> str:
    """Return ``"zen"`` for free-tier model names, else ``"go"`` (keyed/other)."""
    from agent_core.constants import _ZEN_TIER_PREFIXES
    m = (model or "").lower()
    return "zen" if m.startswith(_ZEN_TIER_PREFIXES) else "go"


def _matches_slot(
    model_name: str,
    entry_provider: str,
    entry_override: str | None,
    persisted_provider: str,
) -> bool:
    """Whether the active *model_name* should drive this failover slot.

    True when the active model routes to the slot's provider, and — when the
    slot carries a per-entry override — when the active model is in the same
    tier (zen vs go) as that override.  This lets a user-selected zen model
    drive the zen slot while the go slot keeps its configured default, and
    vice versa, instead of the override blindly clobbering the selection.
    """
    if provider_for(model_name, entry_provider, persisted_provider) != entry_provider:
        return False
    if entry_override is None:
        return True
    return _model_mode(model_name) == _model_mode(entry_override)


def _effective_model(
    entry_provider: str,
    entry_override: str | None,
    model_name: str,
    persisted_provider: str,
) -> str | None:
    """Resolve the model a single failover slot should use.

    A per-entry override wins, except when the active model targets this slot
    *and* is the same tier (zen/go) as the override — then the active selection
    wins.  With no override, the active model is used when it routes here,
    otherwise the provider falls back to its own default (``None``).
    """
    if entry_override:
        if _matches_slot(model_name, entry_provider, entry_override, persisted_provider):
            return model_name
        return entry_override
    if provider_for(model_name, entry_provider, persisted_provider) == entry_provider:
        return model_name
    return None


def build_provider(
    settings: Any, model_name: str, provider_override: str | None = None
) -> "LLMProvider":
    """Configured provider factory (decision #008/#013).

    Builds the concrete provider for the selected provider (LM Studio default,
    #009).  When :attr:`settings.llm_providers` lists more than one provider,
    builds each entry in order and wraps them in a :class:`FailoverProvider`
    so a connectivity loss on the active provider fails over to the next.

    Each chain entry may carry a per-entry model override of the form
    ``provider:model`` (split on the first colon).  This lets the same
    provider appear twice in different modes — e.g. ``opencode:opencode-zen/
    hy3-free`` (keyless free tier, primary cloud) followed by ``opencode:
    opencode-go/deepseek-v4-flash`` (keyed tier, secondary cloud).  When an
    entry has no override, the active ``model_name`` is used only if it
    already targets that provider (so a chosen zen model drives the zen slot
    and a chosen go model drives the go slot); otherwise the provider falls
    back to its own configured default model.

    When *provider_override* is given (``"lmstudio"`` or ``"opencode"``) it
    takes precedence over the model-name prefix and the persisted provider —
    this is how the ``model <name> --provider <p>`` command lets the user
    explicitly choose which provider a model name routes to, instead of the
    provider being inferred purely from the model name.
    """
    from agent_core.constants import load_model_json

    persisted = load_model_json()
    persisted_provider = str(persisted.get("provider") or "")

    def _build_one(
        provider_name: str,
        entry_model: str | None,
        user_explicit: bool = False,
    ) -> "LLMProvider":
        # The model each slot actually uses.
        #
        # When the user explicitly chose this provider (via ``-p``), their
        # model always wins — chain per-entry defaults are only fallbacks for
        # slots the user did NOT explicitly select.
        eff: str | None
        if user_explicit:
            eff = model_name
        else:
            # Per-entry override wins, except when the active model targets
            # this slot *and* is the same tier (zen/go) as the override.
            eff = _effective_model(
                provider_name, entry_model, model_name, persisted_provider
            )

        if provider_name == "opencode":
            from .opencode_provider import OpencodeProvider

            opencode_model: str = str(eff or getattr(settings, "opencode_model", DEFAULT_OPENCODE_MODEL))
            return OpencodeProvider(
                model_name=opencode_model,
                server_url=getattr(settings, "opencode_server_url", DEFAULT_OPENCODE_SERVER_URL),
                password=getattr(settings, "opencode_password", ""),
                api_url=getattr(settings, "opencode_api_url", DEFAULT_OPENCODE_API_BASE),
                api_key=getattr(settings, "opencode_api_key", ""),
            )
        # llama.cpp (llama-server) OpenAI-compatible provider.
        if provider_name == "llama":
            from .llama_provider import LlamaProvider
            return LlamaProvider(
                model_name=eff,
                api_url=getattr(settings, "llama_base_url", DEFAULT_LLAMA_BASE_URL),
            )

        # OpenRouter hosted gateway (OpenAI-compatible, native tool calling).
        if provider_name == "openrouter":
            from .openrouter_provider import OpenRouterProvider
            openrouter_model: str = str(eff or getattr(settings, "openrouter_model", DEFAULT_OPENROUTER_MODEL))
            return OpenRouterProvider(
                model_name=openrouter_model,
                api_url=getattr(settings, "openrouter_api_url", DEFAULT_OPENROUTER_API_BASE),
                api_key=getattr(settings, "openrouter_api_key", ""),
            )

        # Unknown entries fall back to LM Studio (the default provider).
        from .lmstudio import LMStudioProvider

        return LMStudioProvider(model_name=eff)

    chain = tuple(getattr(settings, "llm_providers", ()) or ())
    if not chain:
        chain = (getattr(settings, "llm_provider", "lmstudio"),)

    # The ``cheapest-cloud`` keyword resolves at build time to the cheapest
    # paid ``opencode-go/<id>`` model (see :func:`cheapest_opencode_go_model`),
    # so a chain entry like ``cheapest-cloud`` becomes
    # ``opencode:opencode-go/<cheapest>`` and flows through the normal opencode
    # slot logic below.  Without a priced cloud model it falls back to the
    # configured opencode default so the chain still builds.
    resolved_chain = []
    for entry in chain:
        if _provider_part(entry) == _CHEAPEST_CLOUD_KEYWORD:
            cheapest = cheapest_opencode_go_model()
            if cheapest:
                resolved_chain.append(f"opencode:{cheapest}")
            else:
                resolved_chain.append("opencode")
        else:
            resolved_chain.append(entry)
    chain = tuple(resolved_chain)

    # A single configured provider always yields the concrete provider —
    # routing (persisted/prefix/override) selects WHICH one, it never extends
    # the chain.  Only an explicit multi-provider chain builds a FailoverProvider.
    if len(chain) == 1:
        # An explicit override wins over prefix-based and persisted routing.
        if provider_override in ("lmstudio", "opencode", "llama", "openrouter"):
            return _build_one(provider_override, None, user_explicit=True)
        # No override: the model prefix / persisted provider still selects the
        # concrete provider (e.g. an opencode-go name routes to opencode even
        # when llm_provider defaults to lmstudio).
        routed = provider_for(model_name, chain[0], persisted_provider)
        return _build_one(routed, None)

    # Multi-provider failover.  The user's desired cloud-first / local-fallback
    # sequence is preserved, but the slot that carries the user's *explicit*
    # model selection is promoted to the front so it is tried first — the
    # configured chain defaults are only fallbacks for when the selected model
    # is unavailable or fails.  Without an explicit ``-p`` override the active
    # (persisted) model name still routes: a persisted llama/opencode/lmstudio
    # model keeps its slot first instead of the chain default sitting on top.
    #
    # ``explicit_entry`` names the exact chain entry (string) that is promoted
    # and must use the user's model directly; every other slot keeps its own
    # configured default via ``_effective_model``.  It is tracked by entry
    # (not just provider-name) so that a split opencode zen/go chain promotes
    # only the tier that matches the user's model.
    ordered_entries = list(chain)
    explicit_entry: str | None = None
    if provider_override in ("lmstudio", "opencode", "llama", "openrouter"):
        matching = [e for e in ordered_entries if _provider_part(e) == provider_override]
        rest = [e for e in ordered_entries if _provider_part(e) != provider_override]
        if matching:
            # Keep the matching entry's existing override (e.g. "opencode:…").
            explicit_entry = matching[0]
            ordered_entries = [matching[0], *rest]
        else:
            explicit_entry = provider_override
            ordered_entries = [provider_override, *rest]
    else:
        # No explicit override: promote the chain entry that the active model
        # routes to (and whose tier, for a split opencode zen/go chain, matches
        # the model).  The matching slot becomes the primary and uses the
        # user's model directly.
        routed = provider_for(model_name, chain[0], persisted_provider)
        candidates = []
        for e in ordered_entries:
            if _provider_part(e) != routed:
                continue
            if _matches_slot(
                model_name,
                _provider_part(e),
                _split_entry(e)[1],
                persisted_provider,
            ):
                candidates.append(e)
        if candidates:
            explicit_entry = candidates[0]
            ordered_entries = [
                explicit_entry,
                *(e for e in ordered_entries if e is not explicit_entry),
            ]

    providers = [
        _build_one(
            _provider_part(e),
            _split_entry(e)[1],
            user_explicit=(e == explicit_entry),
        )
        for e in ordered_entries
    ]
    return FailoverProvider(
        providers,
        model_name=model_name,
        strategy=getattr(settings, "failover_strategy", "ordered") or "ordered",
    )


# Transport/connectivity failure signals embedded in a provider's returned
# ``[Error: ...]`` string.  Both concrete providers swallow connection
# failures and return them as text instead of raising, so failover keys off
# these phrases rather than exception types.  Matching is case-insensitive.
_CONNECTION_FAILURE_RE = re.compile(
    r"\[Error:\s*(?:"
    r".*(?:unreachable|connection\s*(?:refused|reset|error)|connecterror|"
    r"timeout|timed out|urlerror|nameresolutionerror|failed to resolve|"
    r"getaddrinfo|http error 5\d\d|"
    r"opencode-zen free model \S+ is currently unavailable|"
    r"model\s+[^\"\]]*\bnot supported)"
    r")",
    re.IGNORECASE,
)


def is_connection_failure(text: str) -> bool:
    """Return True if *text* looks like a transport-level connectivity error.

    Providers return ``[Error: ...]`` strings on failure (not exceptions),
    so the failover wrapper uses this to decide whether to try the next
    provider.  We only treat transport/5xx failures as failover-worthy —
    a 4xx/auth error is permanent and must NOT be retried against another
    provider.
    """
    if not text or "[Error:" not in text:
        return False
    return bool(_CONNECTION_FAILURE_RE.search(text))


class FailoverProvider:
    """Wrap an ordered list of providers with connectivity failover (#013).

    Implements the :class:`LLMProvider` protocol by trying each wrapped
    provider in order for :meth:`chat`.  When a provider returns a
    transport-level failure (see :func:`is_connection_failure`), the next
    provider is tried.  The first non-failure result wins.  If every provider
    fails, the last error text is returned (the agent degrades gracefully
    rather than crashing — callers already handle ``[Error: ...]`` strings).

    ``chat_stream`` and ``analyze_code`` delegate to the FIRST provider only:
    streaming and one-shot analysis don't benefit from failover and this keeps
    their behavior identical to a single provider.
    """

    def __init__(
        self,
        providers: list["LLMProvider"],
        model_name: str,
        strategy: str = "ordered",
    ) -> None:
        if not providers:
            raise ValueError("FailoverProvider requires at least one provider")
        if strategy not in ("ordered", "cheapest"):
            raise ValueError(f"unknown failover strategy: {strategy!r}")
        self._strategy = strategy
        self._providers = list(providers)
        # For the "cheapest" strategy, pre-order the chain by ascending
        # per-token cost; the user's configured order is still the tie-break.
        # Each provider is priced by its OWN model id (providers in a chain
        # often serve different models), falling back to the failover's
        # model_name when a wrapped provider has none.
        if strategy == "cheapest":
            self._providers = sorted(
                self._providers,
                key=lambda p: cost_per_token(
                    getattr(p, "model_name", model_name), _provider_type(p)
                ),
            )
        self.model_name = model_name
        self.temperature = providers[0].temperature
        self.max_tokens = providers[0].max_tokens
        self._profile_name: str | None = getattr(providers[0], "_profile_name", None)
        #: Metrics of whichever provider actually answered (for reporting).
        self.last_response_metrics: ResponseMetrics | None = None

    @property
    def providers(self) -> list["LLMProvider"]:
        return self._providers

    def apply_profile(
        self, name: str, temperature: float, max_tokens: int,
    ) -> None:
        """Activate *name* on every wrapped provider (profile is global)."""
        self._profile_name = name
        self.temperature = temperature
        self.max_tokens = max_tokens
        for provider in self._providers:
            provider.apply_profile(name, temperature, max_tokens)

    async def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        disable_thinking: bool = False,
    ) -> str:
        last_error: str | None = None
        for index, provider in enumerate(self._providers):
            try:
                result = await provider.chat(
                    messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    disable_thinking=disable_thinking,
                )
            except Exception as exc:  # noqa: BLE001 - failover on any raise too
                result = f"[Error: {exc}]"
            if not is_connection_failure(result):
                self.last_response_metrics = getattr(
                    provider, "last_response_metrics", None
                )
                if index != 0:
                    logger.warning(
                        "LLM failover: provider %d/%d answered after %d failed",
                        index + 1,
                        len(self._providers),
                        index,
                    )
                return result
            last_error = result
            logger.warning(
                "LLM provider %d/%d unreachable (%s); failing over to next",
                index + 1,
                len(self._providers),
                result,
            )
        # All providers failed — return the last error text so callers treat
        # it exactly like a normal provider failure.
        return last_error if last_error is not None else "[Error: all LLM providers failed]"

    async def chat_stream(self, messages: list[dict[str, str]]) -> str:
        return await self._providers[0].chat_stream(messages)

    async def analyze_code(self, code: str) -> str:
        return await self._providers[0].analyze_code(code)


class LLMProvider(Protocol):
    """Abstract interface for LLM providers.

    This protocol defines the contract that all LLM providers must implement.
    It enables dependency inversion - Agent depends on this abstraction,
    not on concrete LMStudio implementation.

    Metrics contract: providers with usage data additionally expose
    ``last_response_metrics: ResponseMetrics | None`` after each ``chat``
    (see :func:`get_last_metrics`) — the return type itself stays ``str`` so
    every existing caller keeps working.

    Profile contract: callers never assign provider attributes directly —
    they call :meth:`apply_profile` (implemented by both concrete providers),
    which updates the profile name plus temperature/max-tokens atomically.
    """

    model_name: str
    _profile_name: str | None
    temperature: float
    max_tokens: int

    def apply_profile(
        self, name: str, temperature: float, max_tokens: int,
    ) -> None:
        """Activate *name* with its sampling parameters (see class docstring)."""
        ...

    async def chat(
        self, 
        messages: list[dict[str, str]], 
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        disable_thinking: bool = False,
    ) -> str:
        """Send chat request to LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of OpenAI-format tool schemas
            max_tokens: Optional output token cap for this call
            disable_thinking: If True, send thinking: disabled (reasoning
                models otherwise burn the output budget on reasoning and
                return empty content)
            
        Returns:
            LLM response text, or JSON string if tools present and tool_calls returned
        """
        ...
    
    async def chat_stream(self, messages: list[dict[str, str]]) -> str:
        """Chat with real-time token streaming to console.
        
        Args:
            messages: List of message dicts
            
        Returns:
            Complete response text
        """
        ...
    
    async def analyze_code(self, code: str) -> str:
        """Analyze code and return feedback.
        
        Args:
            code: Source code to analyze
            
        Returns:
            Analysis text with bugs, improvements, suggestions
        """
        ...
