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


def test_build_payload_disable_thinking_sets_reasoning_off_and_template() -> None:
    payload = _provider("kwaipilot_kat-coder-v2.5-dev")._build_payload(
        [{"role": "user", "content": "hi"}], disable_thinking=True
    )
    assert payload["reasoning"] == "off"
    # Safe minimal fallback: the aggressive switches (thinking.disabled /
    # enableThinking / preserve_thinking) caused a full-budget reasoning burn
    # on qwen/qwen3.8-27b (2026-08-18) and are only sent when a model declares
    # them explicitly in KNOWN_MODELS.
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "enableThinking" not in payload
    assert "preserve_thinking" not in payload
    assert "thinking" not in payload


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
    assert payload["chat_template_kwargs"] == {"enable_thinking": False, "preserve_thinking": False}
    assert payload["reasoning"] == "off"


def test_build_payload_unknown_qwen_model_uses_safe_minimal_knobs() -> None:
    """Regression (2026-08-18): qwen/qwen3.8-27b burned its whole 12k output
    budget on reasoning_content with zero content (finish_reason=length) when
    the aggressive disable set was sent; reasoning off + template
    enable_thinking lets it answer normally."""
    payload = _provider("qwen/qwen3.8-27b")._build_payload(
        [{"role": "user", "content": "hi"}], disable_thinking=True
    )
    assert payload["reasoning"] == "off"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "enableThinking" not in payload
    assert "preserve_thinking" not in payload
    assert "thinking" not in payload


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


def test_build_payload_sets_cache_prompt() -> None:
    """_build_payload must request prompt caching on every request.

    LM Studio (llama.cpp backend) caches the stable system+history prefix, so
    each tool-loop iteration only re-processes the new suffix instead of the
    whole growing transcript — this is what prevents the slow prefill ramp
    that tripped ``LMSTUDIO_CHAT_TIMEOUT``."""
    payload = _provider("qwen3.6-27b-mtp")._build_payload([{"role": "user", "content": "hi"}])
    assert payload.get("cache_prompt") is True


def test_stream_payload_also_sets_cache_prompt() -> None:
    """The streaming path (chat_stream) shares _build_payload, so caching must
    be present there too."""
    payload = _provider("qwen3.6-27b-mtp")._build_payload(
        [{"role": "user", "content": "hi"}], stream=True
    )
    assert payload.get("cache_prompt") is True


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
    # The aggressive top-level switches are opt-in per model (explicit
    # disable_thinking_kwargs) — never injected universally.
    payload = _provider("kwaipilot_kat-coder-v2.5-dev")._build_payload(
        [{"role": "user", "content": "hi"}], disable_thinking=True
    )
    assert "enableThinking" not in payload
    assert "preserve_thinking" not in payload


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


class TestScaledTimeout:
    """Regression for the autonomous-agent timeout spiral (2026-08-28): a
    single best-effort-except issue pulled ~575KB of full source into one
    non-streaming POST; on a local 27B model the prefill couldn't finish
    inside the 600s socket floor, so RetryPolicy resent the same giant
    prompt 4x (~40min) before failing over. Timeout must scale with payload
    and stay bounded."""

    def _prov(self, monkeypatch, floor="600"):
        monkeypatch.setenv("LMSTUDIO_CHAT_TIMEOUT", floor)
        return _provider("qwen/qwen3.8-27b")

    def test_small_payload_uses_floor(self, monkeypatch):
        p = self._prov(monkeypatch)
        assert p._scaled_timeout(100) == 600

    def test_floor_respects_env(self, monkeypatch):
        p = self._prov(monkeypatch, "120")
        assert p._scaled_timeout(100) == 120

    def test_scales_with_size(self, monkeypatch):
        p = self._prov(monkeypatch)
        small = p._scaled_timeout(50_000)
        big = p._scaled_timeout(575_000)
        assert small == 601            # 600 + 1 (per 50KB)
        assert big == 600 + (575_000 // 50_000)
        assert big > small

    def test_scaling_capped_at_3600(self, monkeypatch):
        p = self._prov(monkeypatch)
        # ~500MB -> +3000 (the per-call cap) -> 600 + 3000 = 3600
        assert p._scaled_timeout(500_000_000) == 3600

    def test_chat_passes_scaled_timeout_to_request(self, monkeypatch):
        import json
        p = self._prov(monkeypatch)
        captured = {}

        def fake_make_request(payload, timeout=None):
            captured["timeout"] = timeout
            captured["payload_bytes"] = len(json.dumps(payload).encode("utf-8"))
            return {"choices": [{"message": {"content": "ok"}}]}

        monkeypatch.setattr(p, "_make_request", fake_make_request)
        import asyncio
        out = asyncio.run(p.chat([{"role": "user", "content": "x" * 300_000}]))
        assert out == "ok"
        assert captured["timeout"] is not None
        assert captured["timeout"] == p._scaled_timeout(captured["payload_bytes"])
        # 300KB payload -> 600 + 6 = 606, and retries are bounded for it.
        assert captured["timeout"] == 606

