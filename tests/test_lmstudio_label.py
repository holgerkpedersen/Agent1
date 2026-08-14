"""Tests for the [model: ...] status label in LMStudioProvider.chat().

The label must be printed once per session and only re-printed when the
model/profile/temperature/tokens change mid-session — not before every
LLM call in the tool loop.
"""
import asyncio
from unittest.mock import patch

from agent_core.llm.lmstudio import LMStudioProvider


def _provider() -> LMStudioProvider:
    return LMStudioProvider(model_name="laguna-s-2.1", api_key="fake-key")


def _fake_response() -> dict:
    return {
        "choices": [{
            "message": {"role": "assistant", "content": "Hello there"},
            "finish_reason": "stop",
        }]
    }


def test_label_printed_once_for_repeated_calls(capsys) -> None:
    provider = _provider()
    with patch.object(provider, "_make_request", return_value=_fake_response()):
        asyncio.run(provider.chat([{"role": "user", "content": "hi"}]))
        asyncio.run(provider.chat([{"role": "user", "content": "hi again"}]))
        asyncio.run(provider.chat([{"role": "user", "content": "third"}]))
        asyncio.run(provider.chat([{"role": "user", "content": "fourth"}]))

    out = capsys.readouterr().out
    assert out.count("[model:") == 1
    assert "[model: laguna-s-2.1]" in out


def test_label_reprinted_when_profile_changes(capsys) -> None:
    provider = _provider()
    with patch.object(provider, "_make_request", return_value=_fake_response()):
        asyncio.run(provider.chat([{"role": "user", "content": "hi"}]))
        provider._profile_name = "deep-analysis"  # profile switch mid-session
        asyncio.run(provider.chat([{"role": "user", "content": "hi"}]))
        provider.temperature = 0.9
        asyncio.run(provider.chat([{"role": "user", "content": "hi"}]))

    out = capsys.readouterr().out
    assert out.count("[model:") == 3
    assert "[model: laguna-s-2.1]" in out
    assert "profile=deep-analysis t=0.7" in out
    assert "t=0.9" in out


def test_label_changes_when_model_changes(capsys) -> None:
    provider = _provider()
    with patch.object(provider, "_make_request", return_value=_fake_response()):
        asyncio.run(provider.chat([{"role": "user", "content": "hi"}]))
        provider.model_name = "qwen3-coder-30b-a3b-instruct"
        asyncio.run(provider.chat([{"role": "user", "content": "hi"}]))

    out = capsys.readouterr().out
    assert out.count("[model:") == 2
    assert "laguna-s-2.1" in out
    assert "qwen3-coder-30b-a3b-instruct" in out
