"""
agent_core/context_management.py

Async-safe correlation tracking & context propagation utilities.
Provides thread/isolated task scoping for distributed tracing and request correlation.
"""

from __future__ import annotations

import contextvars
import uuid
from typing import Any

CORRELATION_ID_CTX: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)


class CorrelationIdContext:
    """
    Context manager for safely setting and resetting correlation IDs.
    
    Ensures that the correlation ID is isolated to the current execution scope
    (thread, asyncio task, or greenlet). Properly cleans up tokens even if an
    exception occurs within the context block.
    
    Usage:
        with CorrelationIdContext("req-123") as cid:
            print(f"Current ID: {CORRELATION_ID_CTX.get()}")  # req-123
            
        assert CORRELATION_ID_CTX.get() == ""  # Restored to default
    
    For thread pools / executors, propagate context explicitly:
        ctx = copy_correlation_context()
        loop.run_in_executor(executor, ctx.run, target_func)
    """

    def __init__(self, corr_id: str | None = None) -> None:
        self._corr_id = corr_id or str(uuid.uuid4())
        self._token: contextvars.Token[str] | None = None

    def __enter__(self) -> str:
        self._token = CORRELATION_ID_CTX.set(self._corr_id)
        return self._corr_id

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        if self._token is not None:
            CORRELATION_ID_CTX.reset(self._token)
        # Do not suppress exceptions
        return False


def copy_correlation_context() -> contextvars.Context:
    """
    Capture the current execution context including correlation ID.
    
    Returns a snapshot of the active context variables that can be used to
    propagate state into new threads, processes, or asyncio tasks using
    `ctx.run(callable)`.
    
    Example for ThreadPoolExecutor:
        ctx = copy_correlation_context()
        executor.submit(ctx.run, worker_function)
    """
    return contextvars.copy_context()