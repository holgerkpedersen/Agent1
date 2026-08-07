from __future__ import annotations

from agent_core.constants import KNOWN_MODELS
from agent_core.llm.lmstudio import LMStudioProvider


def _provider(model: str) -> LMStudioProvider:
    return LMStudioProvider(model_name=model, api_key="fake-key")


def test_build_payload_default_has_no_thinking_field() -> None:
    payload = _provider("laguna-s-2.1")._build_payload([{"role": "user", "content": "hi"}])
    assert payload["model"] == "laguna-s-2.1"
    assert "thinking" not in payload
    assert "chat_template_kwargs" not in payload


def test_build_payload_disable_thinking_sets_standard_field() -> None:
    payload = _provider("laguna-s-2.1")._build_payload(
        [{"role": "user", "content": "hi"}], disable_thinking=True
    )
    assert payload["thinking"] == {"type": "disabled"}
    assert "chat_template_kwargs" not in payload


def test_build_payload_qwen_adds_chat_template_kwargs() -> None:
    payload = _provider("qwen3.6-27b-mtp")._build_payload(
        [{"role": "user", "content": "hi"}], disable_thinking=True
    )
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_build_payload_merges_tools_and_stream() -> None:
    payload = _provider("qwen3.6-27b-mtp")._build_payload(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function"}],
        stream=True,
        disable_thinking=True,
    )
    assert payload["tools"] == [{"type": "function"}]
    assert payload["stream"] is True
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_known_models_extra_only_for_qwen() -> None:
    assert KNOWN_MODELS["qwen3.6-27b-mtp"].get("disable_thinking_kwargs") == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    for name, info in KNOWN_MODELS.items():
        if name == "qwen3.6-27b-mtp":
            continue
        assert "disable_thinking_kwargs" not in info, name
