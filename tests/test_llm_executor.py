from __future__ import annotations

import asyncio
from typing import Final, List, Dict, Any
from unittest.mock import AsyncMock, patch

import pytest

from fixcommand.core.executor.llm_executor import LLMExecutor, LLMExecutorError


SERVICE_ERROR_CODE: Final[str] = "SERVICE_ERROR"


def test_llm_executor_init_stores_config() -> None:
    """LLMExecutor preserves its model and other config."""
    configured = LLMExecutor(
        api_key="fake-key",
        model="qwen3-coder-30b-a3b-instruct",
        temperature=0.5,
        max_tokens=128,
    )
    assert configured.model == "qwen3-coder-30b-a3b-instruct"
    assert configured.temperature == 0.5
    assert configured.max_tokens == 128


def test_llm_executor_empty_description_raises_error() -> None:
    """LLMExecutor raises LLMExecutorError if issue_description is empty."""
    executor = LLMExecutor(api_key="fake-key", model="qwen3-coder-30b-a3b-instruct")
    with pytest.raises(LLMExecutorError):
        asyncio.run(executor.run_fix_command(""))


def test_llm_executor_no_tool_calls_returns_empty_list() -> None:
    """LLMExecutor returns empty list when no tool calls in response."""
    executor = LLMExecutor(api_key="fake-key", model="qwen3-coder-30b-a3b-instruct")
    
    mock_chat_completion = AsyncMock()
    mock_chat_completion.choices = []
    
    with patch.object(executor.client.chat.completions, "create", return_value=mock_chat_completion):
        # Should raise error not return empty list
        with pytest.raises(LLMExecutorError):
            asyncio.run(executor.run_fix_command("Test issue"))


def test_llm_executor_tool_execution_failure_returns_error_string() -> None:
    """LLMExecutor returns error string when tool call fails."""
    executor = LLMExecutor(api_key="fake-key", model="qwen3-coder-30b-a3b-instruct")
    
    mock_chat_completion = AsyncMock()
    mock_chat_completion.choices = [AsyncMock(message=AsyncMock(model_dump=lambda: {"tool_calls": [{"name": "read_file", "function": {"arguments": '{"filename": "test.txt"}'}}]}))]
    
    with patch.object(executor.client.chat.completions, "create", return_value=mock_chat_completion):
        # Mock the tool call to return an error string
        with patch("fixcommand.core.parser.structured.execute_tool_call", return_value="Error: tool failed"):
            result = asyncio.run(executor.run_fix_command("Test issue"))
            assert isinstance(result[0], str)
            assert "Error: tool failed" in result[0]


def test_llm_executor_api_failure_raises_error() -> None:
    """LLMExecutor raises LLMExecutorError if API call fails."""
    executor = LLMExecutor(api_key="fake-key", model="qwen3-coder-30b-a3b-instruct")
    
    with patch.object(executor.client.chat.completions, "create", side_effect=Exception("API failed")):
        # Should raise specific error not generic exception
        with pytest.raises(LLMExecutorError):
            asyncio.run(executor.run_fix_command("Test issue"))