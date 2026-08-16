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
    payload = _provider("kwaipilot_kat-coder-v2.5-dev")._build_payload(
        [{"role": "user", "content": "hi"}], disable_thinking=True
    )
    assert payload["thinking"] == {"type": "disabled"}
    # Universal fallback: all thinking models now get chat_template_kwargs
    assert payload["chat_template_kwargs"] == {"enable_thinking": False, "preserve_thinking": False}


def test_build_payload_laguna_adds_chat_template_kwargs() -> None:
    payload = _provider("laguna-s-2.1")._build_payload(
        [{"role": "user", "content": "hi"}], disable_thinking=True
    )
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["chat_template_kwargs"] == {
        "enable_thinking": False,
        "preserve_thinking": False,
    }
    # Reasoning is disabled via several LM Studio switches for robustness.
    assert payload["enableThinking"] is False
    assert payload["preserve_thinking"] is False
    assert payload["reasoning"] == "off"


def test_build_payload_qwen_adds_chat_template_kwargs() -> None:
    payload = _provider("qwen3.6-27b-mtp")._build_payload(
        [{"role": "user", "content": "hi"}], disable_thinking=True
    )
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["chat_template_kwargs"] == {"enable_thinking": False, "preserve_thinking": False}
    assert payload["enableThinking"] is False
    assert payload["preserve_thinking"] is False
    assert payload["reasoning"] == "off"


def test_build_payload_merges_tools_and_stream() -> None:
    payload = _provider("qwen3.6-27b-mtp")._build_payload(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function"}],
        stream=True,
        disable_thinking=True,
    )
    assert payload["tools"] == [{"type": "function"}]
    assert payload["stream"] is True
    assert payload["chat_template_kwargs"] == {"enable_thinking": False, "preserve_thinking": False}


def test_known_models_extra_only_for_qwen_and_laguna() -> None:
    allowed = {"qwen3.5-9b-mtp", "qwen3.6-27b-mtp", "qwen3-coder-30b-a3b-instruct", "laguna-s-2.1"}
    for name, info in KNOWN_MODELS.items():
        if name in allowed:
            extra = info.get("disable_thinking_kwargs", {})
            assert extra.get("chat_template_kwargs", {}).get("enable_thinking") is False, name
            if name == "laguna-s-2.1":
                assert extra.get("reasoning") == "off", name
        else:
            assert "disable_thinking_kwargs" not in info, name


def test_build_payload_disable_thinking_always_adds_reasoning_off() -> None:
    """Every model, when disable_thinking is set, must request reasoning off."""
    for name in KNOWN_MODELS:
        payload = _provider(name)._build_payload(
            [{"role": "user", "content": "hi"}], disable_thinking=True
        )
        assert payload["reasoning"] == "off", name
        assert payload["enableThinking"] is False, name
        assert payload["preserve_thinking"] is False, name


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


class TestSanitizeMessageRoles:
    """Strict chat templates (qwen Jinja) reject system messages mid-list."""

    def _payload(self, messages, model="qwen3.6-27b-mtp"):
        return _provider(model)._build_payload(messages)

    def test_leading_system_block_preserved(self):
        payload = self._payload([
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hej"},
        ])
        assert payload["messages"][0] == {"role": "system", "content": "SYS"}

    def test_mid_conversation_system_converted_to_user(self):
        payload = self._payload([
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hej"},
            {"role": "system", "content": "BUDGET WARNING: wrap up"},
            {"role": "assistant", "content": "svar"},
        ])
        msgs = payload["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[2]["role"] == "user"
        assert "[System note]" in msgs[2]["content"]
        assert "BUDGET WARNING" in msgs[2]["content"]
        assert msgs[3]["role"] == "assistant"

    def test_consecutive_mid_list_systems_all_converted(self):
        payload = self._payload([
            {"role": "user", "content": "a"},
            {"role": "system", "content": "n1"},
            {"role": "system", "content": "n2"},
            {"role": "user", "content": "b"},
        ])
        roles = [m["role"] for m in payload["messages"]]
        assert roles == ["user", "user", "user", "user"]

    def test_history_without_system_untouched(self):
        payload = self._payload([
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ])
        assert [m["role"] for m in payload["messages"]] == ["user", "assistant"]

    def test_orphan_tool_messages_dropped(self):
        """tool messages without a matching assistant tool_calls are dropped —
        strict gateways 400 on orphans (opencode Console Go)."""
        payload = self._payload([
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "x", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "res"},
            {"role": "tool", "tool_call_id": "ghost", "content": "orphan"},
        ])
        ids = [m.get("tool_call_id") for m in payload["messages"]]
        assert "ghost" not in ids
        assert "c1" in ids

