"""Tests for the cloud-first / local-fallback failover chain (zen, go, lmstudio, llama).

Covers the decision #013 variant where the opencode-zen FREE tier is the
primary cloud provider, opencode-go the secondary cloud provider, and
LM Studio / llama.cpp are the primary / secondary local fallbacks.  Each
chain entry may carry a per-entry "provider:model" override so the same
provider appears twice in different modes (zen vs go).
"""
from __future__ import annotations

import pytest

from agent_core.config import (
    AgentSettings,
    ConfigurationError,
    load_agent_settings,
)
from agent_core.llm.provider import (
    FailoverProvider,
    _model_mode,
    _split_entry,
    build_provider,
    is_connection_failure,
)
from _helpers import _default_zen_free_model


def _settings(llm_providers: tuple[str, ...]) -> AgentSettings:
    return AgentSettings(
        llm_provider=llm_providers[0].split(":", 1)[0].strip(),
        llm_providers=llm_providers,
        opencode_model="opencode-go/deepseek-v4-flash",
    )


DEFAULT_CHAIN = (
    f"opencode:{_default_zen_free_model()}",
    "opencode:opencode-go/deepseek-v4-flash",
    "openrouter",
    "lmstudio",
    "llama",
)


def test_split_entry_strips_model_override() -> None:
    assert _split_entry(f"opencode:{_default_zen_free_model()}") == (
        "opencode",
        _default_zen_free_model(),
    )
    # OpenRouter model ids contain ':' — split on the FIRST colon only.
    assert _split_entry("openrouter:openrouter/meta-llama/llama-3.1-8b:free") == (
        "openrouter",
        "openrouter/meta-llama/llama-3.1-8b:free",
    )
    assert _split_entry("lmstudio") == ("lmstudio", None)


def test_model_mode_distinguishes_zen_from_go() -> None:
    assert _model_mode(_default_zen_free_model()) == "zen"
    assert _model_mode("zen/laguna-s-2.1-free") == "zen"
    assert _model_mode("opencode-go/deepseek-v4-flash") == "go"
    assert _model_mode("laguna-s-2.1") == "go"


def test_config_default_load_chain_is_cloud_first() -> None:
    """load_agent_settings with no provider env vars uses the 5-entry chain."""
    settings = load_agent_settings()
    assert settings.llm_providers == DEFAULT_CHAIN
    # The active provider (first entry's provider part) is opencode (zen tier).
    assert settings.llm_provider == "opencode"


def test_build_provider_default_chain_order_and_modes() -> None:
    """zen -> go -> openrouter -> lmstudio -> llama, with zen in free mode."""
    settings = _settings(DEFAULT_CHAIN)
    provider = build_provider(settings, _default_zen_free_model())
    assert isinstance(provider, FailoverProvider)
    classes = [type(p).__name__ for p in provider.providers]
    assert classes == [
        "OpencodeProvider",
        "OpencodeProvider",
        "OpenRouterProvider",
        "LMStudioProvider",
        "LlamaProvider",
    ]
    zen, go, _or, lm, llama = provider.providers
    # zen slot uses the free model in keyless mode; go slot uses the keyed model.
    assert zen.model_name == _default_zen_free_model()
    assert zen.zen_mode is True
    assert go.model_name == "opencode-go/deepseek-v4-flash"
    assert go.zen_mode is False


def test_build_provider_single_override_is_concrete() -> None:
    """A single 'provider:model' entry yields the concrete provider (no failover)."""
    settings = _settings((f"opencode:{_default_zen_free_model()}",))
    provider = build_provider(settings, _default_zen_free_model())
    assert type(provider).__name__ == "OpencodeProvider"
    assert provider.zen_mode is True


def test_active_zen_model_drives_zen_slot_only() -> None:
    """Selecting a different zen model must drive the zen slot, not the go slot."""
    settings = _settings(DEFAULT_CHAIN)
    provider = build_provider(settings, "opencode-zen/laguna-s-2.1-free")
    zen, go = provider.providers[0], provider.providers[1]
    assert zen.model_name == "opencode-zen/laguna-s-2.1-free"
    # go slot keeps its configured default — the override isn't clobbered.
    assert go.model_name == "opencode-go/deepseek-v4-flash"


def test_active_go_model_drives_go_slot_only() -> None:
    settings = _settings(DEFAULT_CHAIN)
    provider = build_provider(settings, "opencode-go/gpt-oss")
    # The selected model's slot is promoted to the front (tried first), so the
    # go slot carries the active selection; the zen slot keeps its default.
    go, zen = provider.providers[0], provider.providers[1]
    assert go.model_name == "opencode-go/gpt-oss"
    assert zen.model_name == _default_zen_free_model()


def test_local_selection_drives_only_that_local_slot() -> None:
    """A LM Studio model selection must drive the lmstudio slot, not llama."""
    settings = _settings(DEFAULT_CHAIN)
    provider = build_provider(settings, "laguna-s-2.1")
    lm, llama = provider.providers[0], provider.providers[3]
    # Selected local model is promoted to the front (tried first).
    assert lm.model_name == "laguna-s-2.1"
    # llama slot keeps its own default model, NOT the active LM Studio selection.
    assert llama.model_name != "laguna-s-2.1"


def test_provider_override_moves_entry_to_front_preserving_override() -> None:
    """`model <x> -p llama` must reorder so llama is first, keeping its slot."""
    settings = _settings(DEFAULT_CHAIN)
    provider = build_provider(settings, "llama/Bonsai-27B", provider_override="llama")
    assert [type(p).__name__ for p in provider.providers] == [
        "LlamaProvider",
        "OpencodeProvider",
        "OpencodeProvider",
        "OpenRouterProvider",
        "LMStudioProvider",
    ]
    # The zen slot override is preserved (not collapsed to bare 'opencode').
    assert provider.providers[1].model_name == _default_zen_free_model()


def test_zen_exhaustion_triggers_failover_to_go() -> None:
    """A fully-down zen free tier must fail over to the go slot."""

    class _Stub:
        def __init__(self, name: str, fail: bool) -> None:
            self.name = name
            self.model_name = name
            self.temperature = 0.7
            self.max_tokens = 100
            self._profile_name = None
            self.last_response_metrics = None
            self.call_count = 0

        async def chat(self, messages, **kwargs):
            self.call_count += 1
            if self.name == "zen":
                return (
                    f"[Error: opencode-zen free model {_default_zen_free_model()} is "
                    "currently unavailable on the backend. Try another free model "
                    f"with 'model opencode-zen/<id>-free' (e.g. {_default_zen_free_model()}). "
                    f"Checked: {_default_zen_free_model()}.]"
                )
            return f"ok:{self.name}"

        async def chat_stream(self, messages):
            return "streamed"

        async def analyze_code(self, code):
            return "analyzed"

        def apply_profile(self, *a):
            pass

    zen = _Stub("zen", fail=True)
    go = _Stub("go", fail=False)
    fp = FailoverProvider([zen, go], model_name=_default_zen_free_model())
    import asyncio

    out = asyncio.run(fp.chat([{"role": "user", "content": "hi"}]))
    assert out == "ok:go"
    assert zen.call_count == 1
    assert go.call_count == 1


def test_is_connection_failure_detects_zen_exhaustion() -> None:
    msg = (
        f"[Error: opencode-zen free model {_default_zen_free_model()} is currently "
        "unavailable on the backend. Try another free model with "
        f"'model opencode-zen/<id>-free' (e.g. {_default_zen_free_model()}). "
        f"Checked: {_default_zen_free_model()}.]"
    )
    assert is_connection_failure(msg)


def test_config_rejects_invalid_provider_with_model() -> None:
    with pytest.raises(Exception):
        AgentSettings(
            llm_provider="opencode",
            llm_providers=(f"opencode:{_default_zen_free_model()}", "bogus:x"),
        )
