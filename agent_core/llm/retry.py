"""Retry policy for transient LLM API errors."""
import asyncio
from typing import Awaitable, Callable, TypeVar, Any

T = TypeVar('T')


class RetryPolicy:
    """Configurable retry policy with exponential backoff.
    
    Extracted from LLMClient to separate retry concern from LLM communication.
    """
    
    def __init__(
        self, 
        max_retries: int = 3, 
        base_delay: float = 1.0,
        retryable_errors: tuple[type[Exception], ...] = (
            TimeoutError,
            ConnectionResetError,
            ConnectionRefusedError,
        )
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.retryable_errors = retryable_errors
    
    async def execute_with_retry(
        self, 
        func: Callable[..., Awaitable[T]], 
        *args: Any,
        on_retry: Callable[[int, str, float], None] | None = None,
        **kwargs: Any
    ) -> T:
        """Execute function with retry on transient errors.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for func
            on_retry: Optional callback(attempt, error_msg, wait_time) for logging
            **kwargs: Keyword arguments for func
            
        Returns:
            Result from func
            
        Raises:
            Last exception if all retries exhausted
        """
        last_error: BaseException | None = None
        
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except self.retryable_errors as e:
                last_error = e
                error_msg = str(e)
                
                if attempt < self.max_retries - 1:
                    wait_time = self.base_delay * (2 ** attempt)
                    if on_retry:
                        on_retry(attempt + 1, error_msg, wait_time)
                    await asyncio.sleep(wait_time)
            except Exception:
                # Non-retryable error, raise immediately
                raise
        
        if last_error is not None:
            raise last_error
        raise RuntimeError("all retries exhausted")
