"""LLM Provider Protocol - abstract interface for LLM backends."""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


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
    if m.startswith("opencode"):
        return "opencode"
    if m.startswith("zen"):
        return "opencode"
    if m.startswith(("laguna", "qwen", "kwaipilot", "gemma", "meta/", "prism", "lmstudio")):
        return "lmstudio"
    if persisted_provider in ("lmstudio", "opencode"):
        return persisted_provider
    return provider_setting if provider_setting in ("lmstudio", "opencode") else "lmstudio"


def build_provider(settings: Any, model_name: str) -> "LLMProvider":
    """Configured provider factory (decision #008/#013).

    Builds the concrete provider for the selected provider (LM Studio default,
    #009).  When :attr:`settings.llm_providers` lists more than one provider,
    builds each entry in order and wraps them in a :class:`FailoverProvider`
    so a connectivity loss on the active provider fails over to the next.
    """
    from agent_core.constants import load_model_json

    persisted = load_model_json()
    persisted_provider = str(persisted.get("provider") or "")

    def _build_one(provider_name: str) -> "LLMProvider":
        if provider_name == "opencode":
            from .opencode_provider import OpencodeProvider

            return OpencodeProvider(
                model_name=model_name,
                server_url=getattr(settings, "opencode_server_url", "http://127.0.0.1:4096"),
                password=getattr(settings, "opencode_password", ""),
                api_url=getattr(settings, "opencode_api_url", "https://opencode.ai/zen/go/v1"),
                api_key=getattr(settings, "opencode_api_key", ""),
            )
        # Unknown entries fall back to LM Studio (the default provider).
        from .lmstudio import LMStudioProvider

        return LMStudioProvider(model_name=model_name)

    chain = tuple(getattr(settings, "llm_providers", ()) or ())
    if not chain:
        chain = (getattr(settings, "llm_provider", "lmstudio"),)

    # The model prefix still overrides the provider for a single-provider
    # setup; with a failover chain the primary entry is the active provider.
    primary = provider_for(model_name, chain[0], persisted_provider)

    # A single configured provider always yields the concrete provider —
    # routing (persisted/prefix) selects WHICH one, it never extends the
    # chain.  Only an explicit multi-provider chain builds a FailoverProvider.
    if len(chain) == 1:
        return _build_one(primary)

    ordered = (primary, *[p for p in chain if p != primary])
    providers = [_build_one(p) for p in ordered]
    return FailoverProvider(providers, model_name=model_name)


# Transport/connectivity failure signals embedded in a provider's returned
# ``[Error: ...]`` string.  Both concrete providers swallow connection
# failures and return them as text instead of raising, so failover keys off
# these phrases rather than exception types.  Matching is case-insensitive.
_CONNECTION_FAILURE_RE = re.compile(
    r"\[Error:\s*(?:"
    r".*(?:unreachable|connection\s*(?:refused|reset|error)|connecterror|"
    r"timeout|timed out|urlerror|nameresolutionerror|failed to resolve|"
    r"getaddrinfo|http error 5\d\d)"
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

    def __init__(self, providers: list["LLMProvider"], model_name: str) -> None:
        if not providers:
            raise ValueError("FailoverProvider requires at least one provider")
        self._providers = list(providers)
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
