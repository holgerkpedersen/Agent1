"""Configurable LM Studio port — single source of truth.

Regression tests for the centralized resolver in ``agent_core.config``:
``LMSTUDIO_PORT`` (port-only override), ``LMSTUDIO_URL`` (full-URL override,
backward compatible) and the ``1234`` default.  Before this, every consumer
hardcoded its own URL (and ``module_similarity`` even defaulted to the wrong
port, 1235).
"""
from __future__ import annotations

import pytest

from agent_core.config import (
    DEFAULT_LMSTUDIO_PORT,
    lmstudio_base_url,
    lmstudio_port,
    load_agent_settings,
)


@pytest.fixture(autouse=True)
def _clean_lmstudio_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LMSTUDIO_URL", raising=False)
    monkeypatch.delenv("LMSTUDIO_PORT", raising=False)
    monkeypatch.delenv("AGENT_LLM_API_URL", raising=False)


def test_default_port_and_url() -> None:
    assert lmstudio_port() == DEFAULT_LMSTUDIO_PORT
    assert lmstudio_base_url() == f"http://localhost:{DEFAULT_LMSTUDIO_PORT}/v1"


def test_lmstudio_port_env_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LMSTUDIO_PORT", "9999")
    assert lmstudio_port() == 9999
    assert lmstudio_base_url() == "http://localhost:9999/v1"


def test_invalid_lmstudio_port_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LMSTUDIO_PORT", "not-a-port")
    assert lmstudio_port() == DEFAULT_LMSTUDIO_PORT
    assert f":{DEFAULT_LMSTUDIO_PORT}/v1" in lmstudio_base_url()


def test_lmstudio_url_wins_over_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LMSTUDIO_URL", "http://127.0.0.1:4321/v1/")
    monkeypatch.setenv("LMSTUDIO_PORT", "9999")
    assert lmstudio_base_url() == "http://127.0.0.1:4321/v1"


def test_providers_use_configured_port(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_core.llm.lmstudio import LMStudioProvider
    from agent_core.llm.async_provider import LMStudioProvider as AsyncProvider

    monkeypatch.setenv("LMSTUDIO_PORT", "8765")
    sync_prov = LMStudioProvider(model_name="laguna-s-2.1")
    assert sync_prov.lmstudio_url == "http://localhost:8765/v1"
    assert AsyncProvider().base_url == "http://localhost:8765/v1"


def test_management_url_follows_configured_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_core.llm.lmstudio import _management_url

    monkeypatch.setenv("LMSTUDIO_PORT", "8765")
    assert _management_url() == "http://localhost:8765/api/v1"


def test_settings_llm_api_url_derives_from_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LMSTUDIO_PORT", "8123")
    settings = load_agent_settings()
    assert settings.llm_api_url == "http://localhost:8123/v1"


def test_explicit_agent_llm_api_url_still_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_LLM_API_URL", "http://example.org:7777/v1")
    settings = load_agent_settings()
    assert settings.llm_api_url == "http://example.org:7777/v1"


def test_module_similarity_backend_uses_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_core.utils.module_similarity import _EmbeddingBackend

    monkeypatch.setenv("LMSTUDIO_PORT", "8765")
    backend = _EmbeddingBackend()
    assert backend.base_url == "http://localhost:8765/v1"
