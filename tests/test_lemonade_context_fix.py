"""Lemonade provider caps max_tokens at the model's context window.

These tests drive the real ``LemonadeProvider.chat`` code path with the HTTP
layer stubbed out, so they do not depend on a running Lemonade server.
"""
import asyncio

from agent_core.llm.lemonade_provider import LemonadeProvider


def _make_provider(context_length):
    """Return (provider, captured_payloads).

    ``context_length`` is what the fake ``/models`` endpoint reports; ``None``
    simulates an unreachable server (no context info).
    """
    provider = LemonadeProvider(model_name="lemonade/qwen3.5-9b-FLM", max_retries=0)
    captured = []

    def fake_request(method, url, payload=None, timeout=None):
        if method == "GET" and url.endswith("/models"):
            if context_length is None:
                raise OSError("server unreachable")
            return {"data": [{"id": "qwen3.5-9b-FLM", "context_length": context_length}]}
        captured.append(payload)
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    provider._request = fake_request
    return provider, captured


def _sent_max_tokens(provider_ctx, requested=None):
    provider, captured = _make_provider(provider_ctx)
    kwargs = {} if requested is None else {"max_tokens": requested}
    asyncio.run(provider.chat([{"role": "user", "content": "hi"}], **kwargs))
    assert len(captured) == 1
    return captured[0]["max_tokens"]


def test_context_limit_fetch():
    """context_limit is read from the server's /models response."""
    provider, _ = _make_provider(65536)
    assert provider.context_limit is None
    assert provider._fetch_context_limit() == 65536
    assert provider.context_limit == 65536


def test_max_tokens_capped_to_context_window():
    """A request larger than the context window is capped to the window."""
    assert _sent_max_tokens(provider_ctx=4096, requested=16384) == 4096


def test_max_tokens_kept_when_within_context_window():
    """A request that fits inside the window is sent unchanged."""
    assert _sent_max_tokens(provider_ctx=65536, requested=2048) == 2048


def test_max_tokens_uses_default_when_not_requested():
    """Without an explicit max_tokens the provider default is capped too."""
    provider, captured = _make_provider(4096)
    asyncio.run(provider.chat([{"role": "user", "content": "hi"}]))
    assert captured[0]["max_tokens"] == min(provider.max_tokens, 4096)


def test_max_tokens_unbounded_when_context_unknown():
    """If the server gives no context info, max_tokens is not capped."""
    assert _sent_max_tokens(provider_ctx=None, requested=16384) == 16384
