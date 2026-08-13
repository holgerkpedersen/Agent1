from __future__ import annotations

import asyncio
import json
from typing import Final, List, Dict, Any
from unittest.mock import AsyncMock, patch

import pytest

from fixcommand.core.executor.llm_executor import LLMExecutor, LLMExecutorError
from fixcommand.core.parser.structured import ToolCallResult


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
        with patch("fixcommand.core.executor.llm_executor.execute_tool_call", return_value="Error: tool failed"):
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


def test_extract_response_dict_fallback_for_non_pydantic_message() -> None:
    """When the message has no model_dump (non-pydantic), dict(message) is used."""
    executor = LLMExecutor(api_key="fake-key", model="qwen3-coder-30b-a3b-instruct")

    # A plain object that does NOT expose model_dump — forces the fallback branch.
    class PlainMessage:
        def __init__(self) -> None:
            self.content = "no tool calls"

        def keys(self):  # makes dict(message) work like a mapping proxy
            return ["content"]

        def __getitem__(self, key):
            return getattr(self, key)

    result = executor._extract_response_dict(PlainMessage())
    assert isinstance(result, dict)
    assert result == {"content": "no tool calls"}


def test_run_fix_command_concurrent_tool_calls_with_mixed_results() -> None:
    """Multiple tool calls run concurrently; success and failure coexist in the list."""
    executor = LLMExecutor(api_key="fake-key", model="qwen3-coder-30b-a3b-instruct")

    # Build arguments as proper JSON strings so they parse cleanly regardless of quote style.
    a_args_json: str = json.dumps({"filename": "a.txt"})
    b_args_json: str = json.dumps({"filename": "b.txt"})

    def _response_dict() -> Dict[str, Any]:
        return {
            "tool_calls": [
                {"name": "read_file", "function": {"arguments": a_args_json}},
                {"name": "read_file", "function": {"arguments": b_args_json}},
            ],
        }

    mock_chat_completion = AsyncMock()
    mock_chat_completion.choices = [AsyncMock(message=AsyncMock(model_dump=_response_dict))]

    with patch.object(executor.client.chat.completions, "create", return_value=mock_chat_completion):
        # First tool call succeeds; second raises — gather must keep both as strings.
        def fake_execute_tool_call(call: ToolCallResult) -> str:
            if call["arguments"].get("filename") == "b.txt":
                raise RuntimeError("boom on b.txt")
            return "read a.txt OK"

        with patch("fixcommand.core.executor.llm_executor.execute_tool_call", side_effect=fake_execute_tool_call):
            result = asyncio.run(executor.run_fix_command("Test issue"))
            assert isinstance(result, list)
            assert len(result) == 2
            assert any(r == "read a.txt OK" for r in result)
            assert any("Error: boom on b.txt" in r for r in result)