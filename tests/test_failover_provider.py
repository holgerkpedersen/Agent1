"""Tests for LLM provider failover (decision #013).

Covers: config parsing/validation of the llm_providers chain, the
connection-failure detector, and the FailoverProvider failover loop.
"""
from __future__ import annotations

import logging
from typing import Any

import pytest

from agent_core.config import AgentSettings, ConfigurationError, load_agent_settings
from agent_core.llm.provider import (
    FailoverProvider,
    build_provider,
    is_connection_failure,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeProvider:
    """Minimal LLMProvider stand-in for failover tests.

    Returns a canned response, or an ``[Error: ...]`` string when its index
    is listed in ``fail_indices`` (simulating a connectivity outage).
    """

    def __init__(self, name: str, fail_indices: set[int], result: str = "ok") -> None:
        self.name = name
        self._fail_indices = fail_indices
        self._result = result
        self.model_name = "fake-model"
        self.temperature = 0.7
        self.max_tokens = 100
        self._profile_name: str | None = None
        self.last_response_metrics = None
        self.call_count = 0

    def apply_profile(self, _name: str, _t: float, _m: int) -> None:
        self._profile_name = _name

    async def chat(self, messages: list[dict[str, str]], **_: Any) -> str:
        idx = self.call_count
        self.call_count += 1
        if idx in self._fail_indices:
            return f"[Error: {self.name} unreachable at attempt {idx}]"
        return f"{self._result}:{self.name}"

    async def chat_stream(self, messages: list[dict[str, str]]) -> str:
        return "streamed"

    async def analyze_code(self, code: str) -> str:
        return "analyzed"


def _make_chain(fail_map: dict[str, set[int]], result: str = "ok") -> FailoverProvider:
    providers = [
        _FakeProvider("lmstudio", fail_map.get("lmstudio", set()), result),
        _FakeProvider("opencode", fail_map.get("opencode", set()), result),
    ]
    return FailoverProvider(providers, model_name="fake-model")


@pytest.mark.anyio
async def test_single_provider_succeeds() -> None:
    fp = _make_chain({})
    out = await fp.chat([{"role": "user", "content": "hi"}])
    assert out == "ok:lmstudio"
    assert fp.providers[0].call_count == 1
    assert fp.providers[1].call_count == 0


@pytest.mark.anyio
async def test_failover_to_second_provider() -> None:
    # lmstudio fails on its only attempt; opencode succeeds.
    fp = _make_chain({"lmstudio": {0}})
    out = await fp.chat([{"role": "user", "content": "hi"}])
    assert out == "ok:opencode"
    assert fp.providers[0].call_count == 1
    assert fp.providers[1].call_count == 1
    assert fp.last_response_metrics is None


@pytest.mark.anyio
async def test_all_providers_fail_returns_last_error() -> None:
    fp = _make_chain({"lmstudio": {0}, "opencode": {0}})
    out = await fp.chat([{"role": "user", "content": "hi"}])
    assert out.startswith("[Error: opencode unreachable")
    assert fp.providers[0].call_count == 1
    assert fp.providers[1].call_count == 1


@pytest.mark.anyio
async def test_failover_warns_on_unreachable(caplog: pytest.LogCaptureFixture) -> None:
    fp = _make_chain({"lmstudio": {0}})
    with caplog.at_level(logging.WARNING):
        await fp.chat([{"role": "user", "content": "hi"}])
    assert any("failing over" in rec.message for rec in caplog.records)
    assert any("answered after" in rec.message for rec in caplog.records)


@pytest.mark.anyio
async def test_chat_stream_and_analyze_delegate_to_first() -> None:
    fp = _make_chain({})
    assert await fp.chat_stream([{"role": "user", "content": "hi"}]) == "streamed"
    assert await fp.analyze_code("x=1") == "analyzed"
    assert fp.providers[1].call_count == 0


@pytest.mark.anyio
async def test_apply_profile_fans_out() -> None:
    fp = _make_chain({})
    fp.apply_profile("fast", 0.1, 10)
    assert fp._profile_name == "fast"
    for p in fp.providers:
        assert p._profile_name == "fast"


def test_is_connection_failure_detects_transport_errors() -> None:
    assert is_connection_failure("[Error: LM Studio unreachable at http://localhost:1234]")
    assert is_connection_failure("[Error: Connection refused]")
    assert is_connection_failure("[Error: HTTP Error 503: gateway]")
    assert is_connection_failure("[Error: timed out after 30s]")


def test_is_connection_failure_rejects_auth_errors() -> None:
    # 4xx/auth failures are permanent and must NOT trigger failover.
    assert not is_connection_failure("[Error: HTTP Error 401: Unauthorized]")
    assert not is_connection_failure("[Error: model is not loaded]")
    assert not is_connection_failure("normal response text")
    assert not is_connection_failure("")


def test_config_default_single_provider_chain() -> None:
    settings = AgentSettings()
    assert settings.llm_providers == ("lmstudio",)
    assert settings.llm_providers[0] == settings.llm_provider


def test_config_parses_multi_provider_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("AGENT_LLM_PROVIDERS", "lmstudio, opencode")
    settings = load_agent_settings()
    assert settings.llm_providers == ("lmstudio", "opencode")
    # The active provider is derived from the chain's first entry.
    assert settings.llm_provider == "lmstudio"


def test_config_rejects_invalid_provider() -> None:
    with pytest.raises(ConfigurationError):
        AgentSettings(llm_provider="lmstudio", llm_providers=("lmstudio", "bogus"))


def test_config_rejects_mismatched_first_entry() -> None:
    with pytest.raises(ConfigurationError):
        AgentSettings(llm_provider="opencode", llm_providers=("lmstudio", "opencode"))


def test_build_provider_returns_failover_for_multi_chain() -> None:
    settings = AgentSettings(
        llm_provider="lmstudio", llm_providers=("lmstudio", "opencode")
    )
    provider = build_provider(settings, "laguna-s-2.1")
    assert isinstance(provider, FailoverProvider)
    assert [p.__class__.__name__ for p in provider.providers] == [
        "LMStudioProvider",
        "OpencodeProvider",
    ]


def test_build_provider_single_chain_no_failover() -> None:
    settings = AgentSettings(llm_provider="lmstudio", llm_providers=("lmstudio",))
    provider = build_provider(settings, "laguna-s-2.1")
    assert not isinstance(provider, FailoverProvider)
    assert provider.__class__.__name__ == "LMStudioProvider"
