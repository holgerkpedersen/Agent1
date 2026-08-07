from __future__ import annotations

from agent_core.constants import KNOWN_MODELS
from agent_core.llm import lmstudio as _lmstudio_mod
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
    allowed = {"qwen3.6-27b-mtp", "qwen3-coder-30b-a3b-instruct"}
    for name, info in KNOWN_MODELS.items():
        if name in allowed:
            assert info.get("disable_thinking_kwargs") == {
                "chat_template_kwargs": {"enable_thinking": False}
            }, name
        else:
            assert "disable_thinking_kwargs" not in info, name


def test_load_model_default_eval_batch_size_is_4096() -> None:
    """The load request must use LM Studio's default eval batch size (4096),
    not an accidental 4."""
    import inspect

    sig = inspect.signature(_lmstudio_mod.load_model)
    default = sig.parameters["eval_batch_size"].default
    assert default == 4096


def test_load_model_payload_sends_eval_batch_size_4096(monkeypatch) -> None:
    sent: dict[str, dict] = {}

    def fake_post(url: str, body: dict, timeout: int = 30) -> dict:
        sent[url] = body
        return {"status": "loaded", "instance_id": "inst-1"}

    monkeypatch.setattr(_lmstudio_mod, "_http_post_json", fake_post)
    monkeypatch.setenv("LMSTUDIO_URL", "http://localhost:1234/v1")

    ok, msg = _lmstudio_mod.load_model("qwen3.6-27b-mtp")
    assert ok is True
    posted = sent.get("http://localhost:1234/api/v1/models/load", {})
    assert posted.get("model") == "qwen3.6-27b-mtp"
    assert posted.get("eval_batch_size") == 4096
