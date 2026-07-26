from __future__ import annotations

import asyncio
import json
from typing import Final, NoReturn

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


def _patch_http_client(
    monkeypatch: pytest.MonkeyPatch, transport_handler: object
) -> None:
    """Replace httpx.AsyncClient construction in llm_client with a mocked transport."""
    def factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=transport_handler)  # type: ignore[arg-type]

    monkeypatch.setattr("agent_core.llm_client.httpx.AsyncClient", factory)


def _success_handler(request: httpx.Request) -> httpx.Response:
    """Mock transport returning a valid LLM chat completion."""
    payload = {"choices": [{"message": {"content": "Hello from LLM"}}]}
    return httpx.Response(
        status_code=200,
        headers={"content-type": "application/json"},
        content=json.dumps(payload).encode("utf-8"),
        request=request,
    )


def _failure_handler(request: httpx.Request) -> NoReturn:
    """Mock transport that simulates a network/service failure."""
    raise httpx.ConnectError("connection refused")


def test_chat_returns_success_response(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """chat() parses a successful completion into a non-error LlmResponse."""
    _patch_http_client(monkeypatch, _success_handler)
    result = asyncio.run(client.chat("Say hello"))
    assert isinstance(result, LlmResponse)
    assert result.is_error is False
    assert result.error_code is None
    assert result.content == "Hello from LLM"


def test_chat_handles_service_error(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """chat() recovers from a service failure into an error LlmResponse."""
    _patch_http_client(monkeypatch, _failure_handler)
    result = asyncio.run(client.chat("Say hello"))
    assert isinstance(result, LlmResponse)
    assert result.is_error is True
    assert result.error_code == SERVICE_ERROR_CODE


def test_chat_returns_llm_response_type(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """chat() always returns an LlmResponse instance regardless of outcome."""
    _patch_http_client(monkeypatch, _success_handler)
    result = asyncio.run(client.chat("ping"))
    assert isinstance(result, LlmResponse)


def test_llm_response_is_frozen_dataclass() -> None:
    """LlmResponse is an immutable frozen dataclass."""
    resp = LlmResponse(content="Hello from LLM")
    with pytest.raises((AttributeError, TypeError)):
        resp.content = "mutated"  # type: ignore[misc]