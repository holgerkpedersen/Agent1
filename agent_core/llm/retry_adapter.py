import time
import random
from typing import Callable, TypeVar, Any
from dataclasses import dataclass

R = TypeVar("R")

@dataclass(frozen=True)
class RetryConfig:
    """Configuration for exponential backoff retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,)

class RateLimiter:
    """Simple fixed-interval rate limiter based on monotonic time."""
    
    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self.min_interval = 1.0 / requests_per_second
        self._last_call_time: float = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call_time
        sleep_time = self.min_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        self._last_call_time = time.monotonic()

class RetryAdapter:
    """Applies exponential backoff retries and optional rate limiting to callable targets."""
    
    def __init__(self, config: RetryConfig | None = None, rate_limiter: RateLimiter | None = None) -> None:
        self.config = config or RetryConfig()
        self.rate_limiter = rate_limiter

    def call(self, func: Callable[..., R], *args: Any, **kwargs: Any) -> R:
        last_exception: Exception | None = None
        
        for attempt in range(self.config.max_retries + 1):
            if self.rate_limiter is not None:
                self.rate_limiter.wait()
                
            try:
                return func(*args, **kwargs)
            except self.config.retryable_exceptions as exc:
                last_exception = exc
                if attempt < self.config.max_retries:
                    delay = min(
                        self.config.base_delay * (self.config.backoff_factor ** attempt),
                        self.config.max_delay
                    )
                    if self.config.jitter:
                        delay *= random.uniform(0.5, 1.5)
                    time.sleep(delay)
                    
        if last_exception is not None:
            raise last_exception
        # Unreachable given loop logic, but satisfies strict type checking
        raise RuntimeError("Retry loop exhausted without capturing an exception")