import abc
from typing import Optional, Any, List, Dict


class AsyncLLMProvider(abc.ABC):
    """Abstract base class for asynchronous LLM providers."""
    pass


class LMStudioProvider(AsyncLLMProvider):
    """Provider implementation for LM Studio local inference."""

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = base_url or "http://localhost:1234/v1"


class LLMProviderRegistry:
    """Registry to manage and instantiate various LLM providers."""

    _providers: Dict[str, type[AsyncLLMProvider]] = {}

    @classmethod
    def register(cls, name: str, provider: AsyncLLMProvider) -> None:
        """Registers a provider instance by its name."""
        cls._providers[name] = type(provider)

    @classmethod
    def get(cls, name: str) -> AsyncLLMProvider:
        """Retrieves a default instance of the registered provider."""
        if name not in cls._providers:
            raise KeyError(f"Provider '{name}' is not registered.")
        return cls._providers[name]()

    @classmethod
    def list_providers(cls) -> List[str]:
        """Returns a list of all registered provider names."""
        return list(cls._providers.keys())

    @classmethod
    def create_provider(cls, name: str, **kwargs: Any) -> AsyncLLMProvider:
        """Creates a new instance of the specified provider with custom kwargs."""
        if name not in cls._providers:
            raise KeyError(f"Provider '{name}' is not registered.")
        return cls._providers[name](**kwargs)