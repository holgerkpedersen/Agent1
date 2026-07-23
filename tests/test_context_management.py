"""Tests for async-safe correlation tracking & context propagation."""

import asyncio
import concurrent.futures
from typing import Any

import pytest

from agent_core.context_management import (
    CORRELATION_ID_CTX,
    CorrelationIdContext,
    copy_correlation_context,
)


class TestCorrelationIdContextBasic:
    def test_default_value_is_empty_string(self) -> None:
        assert CORRELATION_ID_CTX.get() == ""

    def test_enter_sets_and_returns_id(self) -> None:
        with CorrelationIdContext("test-123") as ctx_id:
            assert ctx_id == "test-123"
            assert CORRELATION_ID_CTX.get() == "test-123"

    def test_exit_resets_token_to_original(self) -> None:
        original = CORRELATION_ID_CTX.get()
        with CorrelationIdContext("temp-id"):
            pass
        assert CORRELATION_ID_CTX.get() == original

    def test_nested_contexts_restore_parent_correctly(self) -> None:
        with CorrelationIdContext("outer"):
            assert CORRELATION_ID_CTX.get() == "outer"
            with CorrelationIdContext("inner"):
                assert CORRELATION_ID_CTX.get() == "inner"
            assert CORRELATION_ID_CTX.get() == "outer"

    def test_exception_does_not_prevent_token_reset(self) -> None:
        original = CORRELATION_ID_CTX.get()
        with pytest.raises(ValueError):
            with CorrelationIdContext("fail-id"):
                raise ValueError("Intentional failure")
        
        assert CORRELATION_ID_CTX.get() == original


class TestCorrelationIdAsyncIsolation:
    @pytest.mark.asyncio
    async def test_concurrent_tasks_do_not_leak_context(self) -> None:
        results: list[str] = []

        async def worker(task_id: str) -> None:
            with CorrelationIdContext(f"ctx-{task_id}"):
                await asyncio.sleep(0.01)
                results.append(CORRELATION_ID_CTX.get())

        tasks = [asyncio.create_task(worker(str(i))) for i in range(5)]
        await asyncio.gather(*tasks)

        expected = {f"ctx-{i}" for i in range(5)}
        assert set(results) == expected
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_context_reset_after_async_task_completion(self) -> None:
        with CorrelationIdContext("parent-ctx"):
            await asyncio.sleep(0.01)
            
            async def child() -> str:
                with CorrelationIdContext("child-ctx"):
                    return CORRELATION_ID_CTX.get()
                    
            result = await child()
            assert result == "child-ctx"
            
        # After parent exits, should revert to default
        assert CORRELATION_ID_CTX.get() == ""


class TestCorrelationIdExecutorPropagation:
    @pytest.mark.asyncio
    async def test_copy_context_preserves_correlation_id(self) -> None:
        with CorrelationIdContext("propagated-id"):
            ctx = copy_correlation_context()
            
            result_holder: list[str] = []
            
            def target() -> None:
                result_holder.append(CORRELATION_ID_CTX.get())

            ctx.run(target)
            assert result_holder == ["propagated-id"]

    @pytest.mark.asyncio
    async def test_copy_context_outside_manager_returns_default(self) -> None:
        ctx = copy_correlation_context()
        
        result_holder: list[str] = []
        def target() -> None:
            result_holder.append(CORRELATION_ID_CTX.get())
            
        ctx.run(target)
        assert result_holder == [""]

    @pytest.mark.asyncio
    async def test_thread_pool_executor_receives_propagated_context(self) -> None:
        with CorrelationIdContext("executor-parent"):
            ctx = copy_correlation_context()
            
            def worker() -> str:
                return CORRELATION_ID_CTX.get()

            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                result = await loop.run_in_executor(pool, ctx.run, worker)
                
        assert result == "executor-parent"