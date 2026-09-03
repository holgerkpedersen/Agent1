"""Tests for the OpenRouter LLM provider integration.

Covers: model-id prefix handling, provider routing (provider_for), provider
construction (build_provider), config validation of the openrouter chain, the
chat contract (plain text / tool_calls JSON / metrics / error strings), and
that transport failures are recognized as failover-worthy while auth/key
failures are not.
"""
from __future__ import annotations

import json
import urllib.error
from typing import Any
from unittest.mock import patch

import pytest

from agent_core.config import AgentSettings, ConfigurationError
from agent_core.llm.openrouter_provider import OpenRouterProvider, _http_model_id
from agent_core.llm.provider import (
    build_provider,
    is_connection_failure,
    provider_for,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------------------
#  Model-id prefix handling
# ---------------------------------------------------------------------------

class TestHttpModelId:
    def test_strips_openrouter_prefix(self):
        assert (
            _http_model_id("openrouter/anthropic/claude-3.5-haiku")
            == "anthropic/claude-3.5-haiku"
        )

    def test_keeps_bare_vendor_slash_model(self):
        # A bare vendor/model id (no openrouter/ prefix) is sent unchanged —
        # OpenRouter ids are already namespaced.
        assert (
            _http_model_id("deepseek/deepseek-chat-v3-0324")
            == "deepseek/deepseek-chat-v3-0324"
        )


class TestProviderNameNormalization:
    def test_constructor_adds_prefix_when_missing(self):
        prov = OpenRouterProvider(
            model_name="anthropic/claude-3.5-haiku",
            api_key="sk-or-test",
        )
        assert prov.model_name == "openrouter/anthropic/claude-3.5-haiku"

    def test_constructor_keeps_prefix(self):
        prov = OpenRouterProvider(
            model_name="openrouter/deepseek/deepseek-chat-v3-0324",
            api_key="sk-or-test",
        )
        assert prov.model_name == "openrouter/deepseek/deepseek-chat-v3-0324"


# ---------------------------------------------------------------------------
#  Routing + construction
# ---------------------------------------------------------------------------

class TestProviderFor:
    def test_openrouter_prefix_wins_over_setting(self):
        assert (
            provider_for("openrouter/anthropic/claude-3.5-haiku", "lmstudio")
            == "openrouter"
        )

    def test_persisted_openrouter_applies_without_prefix(self):
        assert provider_for("model-x", "lmstudio", "openrouter") == "openrouter"

    def test_setting_applies_without_prefix(self):
        assert provider_for("model-x", "openrouter") == "openrouter"

    def test_other_prefixes_unaffected(self):
        # Regression: adding openrouter must not hijack existing routing.
        assert provider_for("opencode-go/glm-5.2", "lmstudio") == "opencode"
        assert provider_for("laguna-s-2.1", "openrouter") == "lmstudio"


class TestBuildProvider:
    def test_openrouter_provider_built_with_settings(self):
        settings = AgentSettings(
            llm_provider="openrouter",
            llm_providers=("openrouter",),
            openrouter_api_url="https://proxy.example/v1/",
            openrouter_api_key="sk-or-test",
        )
        prov = build_provider(settings, "openrouter/anthropic/claude-3.5-haiku")
        assert isinstance(prov, OpenRouterProvider)
        assert prov.model_name == "openrouter/anthropic/claude-3.5-haiku"
        assert prov.api_url == "https://proxy.example/v1"  # trailing slash stripped
        assert prov.api_key == "sk-or-test"

    def test_override_builds_openrouter(self):
        settings = AgentSettings(llm_provider="lmstudio")
        prov = build_provider(
            settings, "openrouter/anthropic/claude-3.5-haiku",
            provider_override="openrouter",
        )
        assert isinstance(prov, OpenRouterProvider)


# ---------------------------------------------------------------------------
#  Config validation
# ---------------------------------------------------------------------------

def test_config_accepts_openrouter_as_active_provider():
    settings = AgentSettings(llm_provider="openrouter", llm_providers=("openrouter",))
    assert settings.llm_provider == "openrouter"


def test_config_accepts_openrouter_in_failover_chain():
    # Must not raise ConfigurationError.
    settings = AgentSettings(
        llm_provider="lmstudio",
        llm_providers=("lmstudio", "openrouter"),
    )
    assert settings.llm_provider == "lmstudio"


def test_config_still_rejects_unknown_provider():
    with pytest.raises(ConfigurationError):
        AgentSettings(llm_provider="lmstudio", llm_providers=("lmstudio", "bogus"))


# ---------------------------------------------------------------------------
#  Chat contract (fake HTTP)
# ---------------------------------------------------------------------------

class _FakeHttp:
    """Stub urllib.request.urlopen response returning a canned JSON payload."""

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def _provider(max_retries: int = 0) -> OpenRouterProvider:
    # max_retries=0 + zero base delay keep failure-path tests fast (no backoff).
    return OpenRouterProvider(
        model_name="openrouter/anthropic/claude-3.5-haiku",
        api_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test",
        max_retries=max_retries,
        retry_base_delay=0.0,
    )


@pytest.mark.anyio
async def test_chat_returns_plain_text():
    prov = _provider()
    payload = {
        "choices": [
            {"message": {"content": "Hello from OpenRouter"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    with patch("urllib.request.urlopen", return_value=_FakeHttp(payload)):
        out = await prov.chat([{"role": "user", "content": "hi"}])
    assert out == "Hello from OpenRouter"


@pytest.mark.anyio
async def test_chat_metrics_set_from_usage():
    prov = _provider()
    payload = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7},
    }
    with patch("urllib.request.urlopen", return_value=_FakeHttp(payload)):
        await prov.chat([{"role": "user", "content": "hi"}])
    m = prov.last_response_metrics
    assert m is not None
    assert m.prompt_tokens == 12
    assert m.completion_tokens == 7
    assert m.total_tokens == 19


@pytest.mark.anyio
async def test_chat_tool_calls_returned_as_json():
    prov = _provider()
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "run", "arguments": "{\"cmd\": \"ls\"}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {},
    }
    with patch("urllib.request.urlopen", return_value=_FakeHttp(payload)):
        out = await prov.chat(
            [{"role": "user", "content": "list files"}],
            tools=[{"type": "function", "function": {"name": "run"}}],
        )
    parsed = json.loads(out)
    assert "tool_calls" in parsed
    assert parsed["tool_calls"][0]["function"]["name"] == "run"


@pytest.mark.anyio
async def test_chat_sends_bare_model_id_and_key_header():
    prov = _provider()
    captured: dict[str, Any] = {}

    def fake_urlopen(req, timeout=None):
        captured["model"] = json.loads(req.data.decode())["model"]
        captured["auth"] = req.get_header("Authorization")
        return _FakeHttp({
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {},
        })

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        await prov.chat([{"role": "user", "content": "hi"}])
    # The openrouter/ routing prefix must NOT reach the wire.
    assert captured["model"] == "anthropic/claude-3.5-haiku"
    assert captured["auth"] == "Bearer sk-or-test"


@pytest.mark.anyio
async def test_disable_thinking_sends_reasoning_dict():
    prov = _provider()
    captured: dict[str, Any] = {}

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data.decode())
        captured["reasoning"] = body.get("reasoning")
        return _FakeHttp({
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {},
        })

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        await prov.chat([{"role": "user", "content": "hi"}])
        assert captured["reasoning"] is None  # not sent by default
        await prov.chat([{"role": "user", "content": "hi"}], disable_thinking=True)
    # OpenRouter expects a DICT ({"enabled": false}) — the LM Studio string
    # knob ("off") would be rejected.
    assert captured["reasoning"] == {"enabled": False}


@pytest.mark.anyio
async def test_disable_thinking_recovers_from_reasoning_mandatory_400():
    """Regression: some OpenRouter endpoints MANDATE reasoning and reject
    ``reasoning: {"enabled": false}`` with HTTP 400 "Reasoning is mandatory
    for this endpoint and cannot be disabled".  The fixer/implement/fix
    subagents always pass disable_thinking=True, so this used to hard-fail the
    whole turn.  The provider must transparently retry with reasoning
    explicitly ENABLED (``{"enabled": true}``) so the request still succeeds —
    deepseek/deepseek-r1 is a pure-reasoning model that only answers via reasoning."""
    prov = _provider(max_retries=0)
    calls: list[dict[str, Any]] = []

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data.decode())
        calls.append(body)
        if body.get("reasoning") == {"enabled": False}:
            # First attempt (disable requested) is rejected as mandatory.
            raise _http_error(
                400,
                '{"error":{"message":"Reasoning is mandatory for this endpoint '
                'and cannot be disabled.","code":400,"metadata":{}}}',
            )
        # Second attempt (reasoning explicitly enabled) succeeds.
        return _FakeHttp({
            "choices": [{"message": {"content": "fixed"}, "finish_reason": "stop"}],
            "usage": {},
        })

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        out = await prov.chat(
            [{"role": "user", "content": "improve it"}],
            disable_thinking=True,
        )
    assert out == "fixed"
    # Exactly two requests: the disabled one, then the recovered one.
    assert len(calls) == 2
    assert calls[0]["reasoning"] == {"enabled": False}
    assert calls[1]["reasoning"] == {"enabled": True}


@pytest.mark.anyio
async def test_disable_thinking_recovers_from_credits_402():
    """Regression: a hosted OpenRouter reasoning model (deepseek/deepseek-r1)
    with the default max_tokens (50000) exceeds the account's remaining credit
    budget, so OpenRouter returns HTTP 402 'can only afford N'.  The provider
    must transparently retry with a smaller max_tokens instead of surfacing a
    hard failure."""
    prov = _provider(max_retries=0)
    calls: list[dict[str, Any]] = []

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data.decode())
        calls.append(body)
        if len(calls) == 1:
            raise _http_error(
                402,
                '{"error":{"message":"This request requires more credits, or '
                'fewer max_tokens. You requested up to 50000 tokens, but can '
                'only afford 34782.","code":402,"metadata":{}}}',
            )
        return _FakeHttp({
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {},
        })

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        out = await prov.chat(
            [{"role": "user", "content": "improve it"}],
            disable_thinking=True,
        )
    assert out == "ok"
    assert len(calls) == 2
    # The retry must request a smaller budget than the original 50000.
    assert calls[1]["max_tokens"] < calls[0]["max_tokens"]
    assert calls[1]["max_tokens"] == 34782 - 256  # affordable - safety margin


@pytest.mark.anyio
async def test_pure_reasoning_model_returns_reasoning_as_content():
    """Regression: a hosted OpenRouter reasoning model (deepseek/deepseek-r1)
    returns content=None and only a reasoning block.  The provider must fall
    back to the reasoning text as content so the turn is not silently collapsed
    to '(no output)'."""
    prov = _provider(max_retries=0)

    def fake_urlopen(req, timeout=None):
        return _FakeHttp({
            "choices": [{
                "message": {
                    "content": None,
                    "reasoning_content": "Here is my analysis and fix.",
                },
                "finish_reason": "stop",
            }],
            "usage": {},
        })

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        out = await prov.chat(
            [{"role": "user", "content": "improve it"}],
            disable_thinking=False,
        )
    assert out == "Here is my analysis and fix."


@pytest.mark.anyio
async def test_disable_thinking_other_400_still_errors():
    """A non-reasoning HTTP 400 (e.g. invalid model) must NOT be swallowed by
    the mandatory-reasoning recovery path."""
    prov = _provider(max_retries=0)

    def boom(req, timeout=None):
        raise _http_error(400, '{"error":{"message":"invalid model","code":400}}')

    with patch("urllib.request.urlopen", side_effect=boom):
        out = await prov.chat(
            [{"role": "user", "content": "improve it"}],
            disable_thinking=True,
        )
    assert "[Error:" in out
    assert "invalid model" in out


@pytest.mark.anyio
async def test_chat_connection_failure_is_failover_worthy():
    prov = _provider(max_retries=0)

    def boom(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    with patch("urllib.request.urlopen", side_effect=boom):
        out = await prov.chat([{"role": "user", "content": "hi"}])
    assert out.startswith("[Error:")
    # Transport-level failure — is_connection_failure must recognize it so a
    # FailoverProvider moves on to the next provider.
    assert is_connection_failure(out)


@pytest.mark.anyio
async def test_chat_http_4xx_is_permanent_not_failover():
    prov = _provider(max_retries=0)

    def boom(req, timeout=None):
        raise urllib.error.HTTPError(
            url="https://openrouter.ai/api/v1/chat/completions",
            code=401, msg="Unauthorized", hdrs=None, fp=None,
        )

    with patch("urllib.request.urlopen", side_effect=boom):
        out = await prov.chat([{"role": "user", "content": "hi"}])
    assert "[Error:" in out and "401" in out
    # Auth failures are permanent — must NOT trigger failover.
    assert not is_connection_failure(out)


def _http_error(code: int, body: str) -> urllib.error.HTTPError:
    """Build an HTTPError whose read() returns *body* (chat() reads it)."""
    import io

    return urllib.error.HTTPError(
        url="https://openrouter.ai/api/v1/chat/completions",
        code=code,
        msg="",
        hdrs=None,
        fp=io.BytesIO(body.encode("utf-8")),
    )


def _boom_403_harness(req, timeout=None):
    raise _http_error(
        403,
        '{"error":{"message":"thinkingmachines/inkling-small:free is only '
        'available on agentic harnesses. Try plugging it into a coding agent",'
        '"code":403}}',
    )


def _boom_429(req, timeout=None):
    raise _http_error(
        429,
        '{"error":{"message":"Provider returned error","code":429,"metadata":'
        '{"raw":"z-ai/glm-5.2:free is temporarily rate-limited upstream",'
        '"is_byok":false}}',
    )


@pytest.mark.anyio
async def test_chat_403_agentic_harness_is_actionable_not_raw_json():
    """Regression: a restricted free model (\"only available on agentic
    harnesses\") must produce an actionable message, not the raw upstream JSON.
    Also triggers failover so the FailoverProvider can try the next chain entry."""
    prov = _provider(max_retries=0)
    with patch("urllib.request.urlopen", side_effect=_boom_403_harness):
        out = await prov.chat([{"role": "user", "content": "hi"}])
    assert "[Error:" in out
    # Actionable guidance — not a raw JSON blob.
    assert "agentic harnesses" in out.lower() or "restricted" in out.lower()
    assert '{"error"' not in out
    # Restricted models are failover-worthy — the FailoverProvider should try
    # the next chain entry instead of giving up.
    assert is_connection_failure(out)


@pytest.mark.anyio
async def test_chat_429_rate_limited_is_actionable():
    """Regression: an upstream-rate-limited free model (HTTP 429) must produce
    an actionable message after retries, not the raw upstream JSON."""
    prov = _provider(max_retries=0)
    with patch("urllib.request.urlopen", side_effect=_boom_429):
        out = await prov.chat([{"role": "user", "content": "hi"}])
    assert "[Error:" in out
    assert "rate-limited" in out.lower() or "429" in out
    # The raw upstream JSON must not leak into the user-facing error.
    assert 'is_byok' not in out


@pytest.mark.anyio
async def test_chat_missing_key_is_permanent_error():
    prov = _provider()
    prov.api_key = ""
    out = await prov.chat([{"role": "user", "content": "hi"}])
    assert out.startswith("[Error:")
    assert "OPENROUTER_API_KEY" in out
    # A missing key is a configuration problem, not a connectivity outage.
    assert not is_connection_failure(out)


@pytest.mark.anyio
async def test_chat_no_choices_is_error():
    prov = _provider()
    with patch("urllib.request.urlopen", return_value=_FakeHttp({"data": {}})):
        out = await prov.chat([{"role": "user", "content": "hi"}])
    assert out.startswith("[Error:")


# ---------------------------------------------------------------------------
#  Model listing (model command support)
# ---------------------------------------------------------------------------

def test_list_models_free_only_by_default():
    prov = _provider()
    payload = {
        "data": [
            # Paid model — must be excluded from the default free-only listing.
            {"id": "anthropic/claude-3.5-haiku", "pricing": {"prompt": "1", "completion": "3"}},
            # Free model (zero pricing) — must be included.
            {"id": "deepseek/deepseek-chat-v3-0324:free", "pricing": {"prompt": "0", "completion": "0"}},
            # No pricing info but :free suffix — included as a fallback.
            {"id": "meta-llama/llama-3.1-8b-instruct:free"},
        ]
    }
    with patch("urllib.request.urlopen", return_value=_FakeHttp(payload)):
        models = prov.list_models()
    assert models == [
        "openrouter/deepseek/deepseek-chat-v3-0324:free",
        "openrouter/meta-llama/llama-3.1-8b-instruct:free",
    ]


def test_list_models_full_catalog_with_free_only_false():
    prov = _provider()
    payload = {
        "data": [
            {"id": "anthropic/claude-3.5-haiku", "pricing": {"prompt": "1", "completion": "3"}},
            {"id": "deepseek/deepseek-chat-v3-0324:free", "pricing": {"prompt": "0", "completion": "0"}},
        ]
    }
    with patch("urllib.request.urlopen", return_value=_FakeHttp(payload)):
        models = prov.list_models(free_only=False)
    assert models == [
        "openrouter/anthropic/claude-3.5-haiku",
        "openrouter/deepseek/deepseek-chat-v3-0324:free",
    ]


def test_list_models_returns_empty_on_failure():
    prov = _provider()

    def boom(req, timeout=None):
        raise urllib.error.URLError("no route")

    with patch("urllib.request.urlopen", side_effect=boom):
        assert prov.list_models() == []
