import abc
import asyncio
import time
from typing import Optional, Any, List, Dict, Callable, Awaitable


class AsyncLLMProvider(abc.ABC):
    """Abstract base class for asynchronous LLM providers."""

    def __init__(self) -> None:
        self.max_retries = 3
        self.base_delay = 1.0
        self.max_delay = 60.0
        self.rate_limit_rps = 10.0
        self._last_request_time: float = 0.0

    async def _enforce_rate_limit(self) -> None:
        """Enforces rate limiting between requests."""
        min_interval = 1.0 / self.rate_limit_rps
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        self._last_request_time = time.monotonic()

    async def _retry_with_backoff(self, func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
        """Executes an async function with exponential backoff retries."""
        exc_to_raise: BaseException = RuntimeError("Unexpected retry failure")
        for attempt in range(self.max_retries + 1):
            try:
                await self._enforce_rate_limit()
                return await func(*args, **kwargs)
            except BaseException as exc:
                exc_to_raise = exc
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    await asyncio.sleep(delay)
        raise exc_to_raise


class LMStudioProvider(AsyncLLMProvider):
    """Provider implementation for LM Studio local inference."""

    def __init__(self, base_url: Optional[str] = None) -> None:
        super().__init__()
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