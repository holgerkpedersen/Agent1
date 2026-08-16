"""LLM Provider Protocol - abstract interface for LLM backends."""
from typing import Any, Protocol


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
    if m.startswith(("laguna", "qwen", "kwaipilot", "gemma", "meta/", "prism", "lmstudio")):
        return "lmstudio"
    if persisted_provider in ("lmstudio", "opencode"):
        return persisted_provider
    return provider_setting if provider_setting in ("lmstudio", "opencode") else "lmstudio"


def build_provider(settings: Any, model_name: str) -> "LLMProvider":
    """Configured provider factory (decision #008): build the concrete
    provider for the selected provider, keeping LM Studio the default (#009)."""
    from agent_core.constants import load_model_json

    persisted = load_model_json()
    persisted_provider = str(persisted.get("provider") or "")
    if provider_for(
        model_name,
        getattr(settings, "llm_provider", "lmstudio"),
        persisted_provider,
    ) == "opencode":
        from .opencode_provider import OpencodeProvider

        return OpencodeProvider(
            model_name=model_name,
            server_url=getattr(settings, "opencode_server_url", "http://127.0.0.1:4096"),
            password=getattr(settings, "opencode_password", ""),
            api_url=getattr(settings, "opencode_api_url", "https://opencode.ai/zen/go/v1"),
            api_key=getattr(settings, "opencode_api_key", ""),
        )
    from .lmstudio import LMStudioProvider

    return LMStudioProvider(model_name=model_name)


class LLMProvider(Protocol):
    """Abstract interface for LLM providers.
    
    This protocol defines the contract that all LLM providers must implement.
    It enables dependency inversion - Agent depends on this abstraction,
    not on concrete LMStudio implementation.
    """
    
    model_name: str
    
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
