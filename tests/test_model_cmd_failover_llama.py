"""Regression tests: `model ... -p llama` must work even when the user has
multiple `llm_providers` configured (which wraps the LlamaProvider in a
FailoverProvider).  Previously `provider.api_url` raised AttributeError."""
from types import SimpleNamespace

import pytest

from agent_core.commands.model_cmd import ModelCommand, _concrete_llama_provider
from agent_core.llm.provider import build_provider


def _multi_settings():
    return SimpleNamespace(
        llm_provider="lmstudio",
        llm_providers=("lmstudio", "llama"),
        opencode_server_url="http://127.0.0.1:4096",
        opencode_password="",
        opencode_api_url="https://opencode.ai/zen/go/v1",
        opencode_api_key="",
        llama_base_url="http://127.0.0.1:8080/v1",
    )


def test_build_provider_llama_wrapped_in_failover_when_multi():
    """With >1 llm_providers, build_provider returns a FailoverProvider."""
    settings = _multi_settings()
    provider = build_provider(settings, "llama/Bonsai-27B-Q1_0", provider_override="llama")
    assert type(provider).__name__ == "FailoverProvider"
    # The helper must unwrap to the concrete LlamaProvider (with api_url).
    concrete = _concrete_llama_provider(provider)
    assert type(concrete).__name__ == "LlamaProvider"
    assert concrete.api_url == "http://127.0.0.1:8080/v1"


def test_concrete_llama_passthrough_and_no_match():
    settings = _multi_settings()
    llama = build_provider(settings, "llama/Bonsai-27B-Q1_0", provider_override="llama")
    concrete = _concrete_llama_provider(llama)
    # idempotent
    assert _concrete_llama_provider(concrete) is concrete
    # unknown wrapper returns the original
    fake = SimpleNamespace()
    assert _concrete_llama_provider(fake) is fake


@pytest.mark.anyio
async def test_switch_llama_unwraps_failover(monkeypatch):
    """`model llama/<x> -p llama` must not crash on a FailoverProvider."""
    settings = _multi_settings()
    agent = SimpleNamespace()
    agent.llm = SimpleNamespace()
    agent.llm.model_name = "llama/gemma"
    agent.llm._provider = build_provider(settings, "llama/gemma", provider_override="llama")

    captured = {}

    def _fake_ensure(api_url, name):
        captured["api_url"] = api_url
        captured["name"] = name
        return True, "served"

    fake_props = {"default_generation_settings": {}, "model_path": "", "role": "router"}
    monkeypatch.setattr(
        "agent_core.llm.llama_server.is_server_up", lambda api_url: True
    )
    monkeypatch.setattr(
        "agent_core.llm.llama_server.get_role", lambda api_url: "router"
    )
    monkeypatch.setattr(
        "agent_core.llm.llama_server.list_served_models", lambda api_url: ["gemma"]
    )
    monkeypatch.setattr(
        "agent_core.llm.llama_server.ensure_model_served", _fake_ensure
    )

    cmd = ModelCommand()
    # _switch_model dispatches on provider_override via the parse path; call the
    # provider-aware helper directly with provider="llama".
    await cmd._switch_model_with_provider(
        ["llama/Bonsai-27B-Q1_0"], agent, provider="llama"
    )
    assert captured.get("api_url") == "http://127.0.0.1:8080/v1"
    assert captured.get("name") == "llama/Bonsai-27B-Q1_0"
    # The agent now has a concrete (non-failover) provider pinned.
    assert type(agent.llm._provider).__name__ == "LlamaProvider"
