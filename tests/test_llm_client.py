from __future__ import annotations

import asyncio
import json
from typing import Final
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from agent_core.llm_client import LLMClient, LlmResponse


SERVICE_ERROR_CODE: Final[str] = "SERVICE_ERROR"


@pytest.fixture
def client() -> LLMClient:
    """Provide a configured LLMClient for tests."""
    return LLMClient(api_url="http://localhost:1234/v1", timeout_sec=5.0)


def test_llm_response_defaults() -> None:
    """A successful-style LlmResponse defaults to non-error with no code."""
    resp = LlmResponse(content="Hello from LLM")
    assert resp.content == "Hello from LLM"
    assert resp.is_error is False
    assert resp.error_code is None


def test_llm_response_error_variant() -> None:
    """An error-style LlmResponse carries the service error code."""
    resp = LlmResponse(
        content="Service error occurred",
        is_error=True,
        error_code=SERVICE_ERROR_CODE,
    )
    assert resp.is_error is True
    assert resp.error_code == SERVICE_ERROR_CODE


def test_llm_client_init_stores_config() -> None:
    """LLMClient preserves its api_url and timeout_sec configuration."""
    configured = LLMClient(api_url="http://example.com/v1", timeout_sec=7.5)
    assert configured.api_url == "http://example.com/v1"
    assert configured.timeout_sec == 7.5


def test_chat_returns_success_response(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """chat() parses a successful completion into a non-error LlmResponse."""
    mock_response = httpx.Response(
        status_code=200,
        json={"choices": [{"message": {"content": "Hello from LLM"}}]},
        request=httpx.Request("POST", "http://localhost:1234/v1/chat/completions"),
    )
    mock_post = AsyncMock(return_value=mock_response)
    monkeypatch.setattr("agent_core.llm_client.httpx.AsyncClient.post", mock_post)

    result = asyncio.run(client.chat("Say hello"))
    assert isinstance(result, LlmResponse)
    assert result.is_error is False
    assert result.error_code is None
    assert result.content == "Hello from LLM"


def test_chat_handles_service_error(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """chat() recovers from a network failure into an error LlmResponse."""
    monkeypatch.setattr(
        "agent_core.llm_client.httpx.AsyncClient.post",
        AsyncMock(side_effect=httpx.ConnectError("connection refused")),
    )
    result = asyncio.run(client.chat("Say hello"))
    assert isinstance(result, LlmResponse)
    assert result.is_error is True
    assert result.error_code == SERVICE_ERROR_CODE


def test_chat_returns_llm_response_type(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """chat() always returns an LlmResponse instance regardless of outcome."""
    mock_response = httpx.Response(
        status_code=200,
        json={"choices": [{"message": {"content": "Hello from LLM"}}]},
        request=httpx.Request("POST", "http://localhost:1234/v1/chat/completions"),
    )
    monkeypatch.setattr(
        "agent_core.llm_client.httpx.AsyncClient.post",
        AsyncMock(return_value=mock_response),
    )
    result = asyncio.run(client.chat("ping"))
    assert isinstance(result, LlmResponse)


def test_llm_response_is_frozen_dataclass() -> None:
    """LlmResponse is an immutable frozen dataclass."""
    resp = LlmResponse(content="Hello from LLM")
    with pytest.raises((AttributeError, TypeError)):
        resp.content = "mutated"  # type: ignore[misc]
