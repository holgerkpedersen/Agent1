"""Tests for the AMD Lemonade (NPU) provider and its routing/config wiring.

Lemonade is AMD's OpenAI-compatible local server that runs LLMs on the Ryzen AI
XDNA NPU (via FastFlowLM / Ryzen AI LLM) or the iGPU.  It is wired in as a
first-class provider: model ids are namespaced ``lemonade/<id>``, routing /
persistence / ``model list`` all agree, and transport failures fail over.
"""
from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent_core.config import AgentSettings, ConfigurationError, load_agent_settings
from agent_core.llm.lemonade_provider import LemonadeProvider
from agent_core.llm.provider import (
    FailoverProvider,
    build_provider,
    is_connection_failure,
    provider_for,
)


class _FakeResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(req, code: int, body: bytes) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(req.full_url, code, "Err", {}, BytesIO(body))


def _provider() -> LemonadeProvider:
    prov = LemonadeProvider(
        model_name="lemonade/qwen3.5-4b-FLM",
        api_url="http://localhost:13305/api/v1",
    )
    prov._retry_base_delay = 0.0  # no real sleeping in tests
    return prov


_TOOLS = [{"type": "function", "function": {"name": "x", "parameters": {}}}]


def _chat_response(content: str, **message_extra) -> bytes:
    message = {"content": content, **message_extra}
    return json.dumps(
        {
            "choices": [{"message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5},
        }
    ).encode()


def _capturing_urlopen(response: bytes, calls: list):
    def fake(req, timeout=None):
        calls.append(req)
        return _FakeResponse(response)

    return fake


# ---------------------------------------------------------------------------
#  provider_for routing
# ---------------------------------------------------------------------------

class TestRouting:
    def test_lemonade_prefix_routes_to_lemonade(self):
        assert provider_for("lemonade/qwen3.5-4b-FLM") == "lemonade"
        assert provider_for("lemonade/Qwen3-4B-Hybrid") == "lemonade"

    def test_lemonade_namespace_beats_lmstudio_family_substrings(self):
        """Regression guard: a qwen/gemma-family NPU id must NOT be hijacked
        by the LM Studio substring keys."""
        assert provider_for("lemonade/qwen3.5-4b-FLM") == "lemonade"
        assert provider_for("lemonade/gemma3-4b-FLM") == "lemonade"

    def test_persisted_and_setting(self):
        assert provider_for("model-x", "lmstudio", "lemonade") == "lemonade"
        assert provider_for("model-x", "lemonade") == "lemonade"

    def test_lmstudio_names_untouched(self):
        assert provider_for("laguna-s-2.1", "lmstudio") == "lmstudio"
        assert provider_for("llama-4-scout-17b-16e-instruct") == "lmstudio"


# ---------------------------------------------------------------------------
#  build_provider
# ---------------------------------------------------------------------------

def _multi_settings():
    return SimpleNamespace(
        llm_provider="lmstudio",
        llm_providers=("lmstudio", "lemonade"),
        failover_strategy="ordered",
        opencode_server_url="http://127.0.0.1:4096",
        opencode_password="",
        opencode_api_url="https://opencode.ai/zen/go/v1",
        opencode_api_key="",
        llama_base_url="http://127.0.0.1:8080/v1",
        openrouter_api_url="https://openrouter.ai/api/v1",
        openrouter_api_key="",
        openrouter_model="openrouter/meta-llama/llama-3.1-8b-instruct:free",
        lemonade_api_url="http://localhost:13305/api/v1",
        lemonade_model="",
    )


class TestBuildProvider:
    def test_builds_lemonade_provider(self):
        s = SimpleNamespace(
            llm_provider="lemonade", llm_providers=("lemonade",),
            failover_strategy="ordered",
            lemonade_api_url="http://localhost:13305/api/v1",
            lemonade_model="qwen3.5-4b-FLM",
        )
        prov = build_provider(s, "lemonade/qwen3.5-4b-FLM")
        assert isinstance(prov, LemonadeProvider)
        assert prov.model_name == "lemonade/qwen3.5-4b-FLM"
        assert prov.api_url == "http://localhost:13305/api/v1"

    def test_multi_chain_wraps_in_failover(self):
        prov = build_provider(_multi_settings(), "lemonade/qwen3.5-4b-FLM")
        assert isinstance(prov, FailoverProvider)
        assert isinstance(prov.providers[0], LemonadeProvider)

    def test_chain_without_lemonade_still_serves_the_npu_model(self):
        """Regression: a chain that lists only cloud/local providers must NOT
        drop an NPU model — the synthetic front entry serves it first (before
        this, `multillm` silently ran opencode-go/deepseek for a lemonade
        model because the chain had no `lemonade` entry)."""
        s = SimpleNamespace(
            llm_provider="opencode",
            llm_providers=("opencode:opencode-go/deepseek-v4.1-flash", "lmstudio"),
            failover_strategy="ordered",
            opencode_server_url="http://127.0.0.1:4096",
            opencode_password="",
            opencode_api_url="https://opencode.ai/zen/go/v1",
            opencode_api_key="",
            llama_base_url="http://127.0.0.1:8080/v1",
            openrouter_api_url="https://openrouter.ai/api/v1",
            openrouter_api_key="",
            openrouter_model="openrouter/meta-llama/llama-3.1-8b-instruct:free",
            lemonade_api_url="http://localhost:13305/api/v1",
            lemonade_model="",
        )
        prov = build_provider(s, "lemonade/qwen3.5-4b-FLM")
        assert isinstance(prov, FailoverProvider)
        assert isinstance(prov.providers[0], LemonadeProvider)
        assert prov.providers[0].model_name == "lemonade/qwen3.5-4b-FLM"

    def test_override_routes_to_lemonade(self):
        s = _multi_settings()
        prov = build_provider(s, "something-bare", provider_override="lemonade")
        concrete = next(
            (p for p in getattr(prov, "providers", [prov])
             if isinstance(p, LemonadeProvider)),
            None,
        )
        assert concrete is not None
        assert concrete.model_name == "lemonade/something-bare"


# ---------------------------------------------------------------------------
#  chat / payload / tool calls
# ---------------------------------------------------------------------------

class TestChat:
    @pytest.mark.anyio
    async def test_payload_shape_and_prefix_strip(self):
        prov = _provider()
        calls: list = []
        with patch(
            "urllib.request.urlopen",
            side_effect=_capturing_urlopen(_chat_response("hej"), calls),
        ):
            out = await prov.chat([{"role": "user", "content": "hi"}], tools=_TOOLS)
        assert out == "hej"
        payload = json.loads(calls[0].data.decode("utf-8"))
        assert payload["model"] == "qwen3.5-4b-FLM"  # prefix stripped
        assert payload["messages"][0]["content"] == "hi"
        assert payload["tools"][0]["function"]["name"] == "x"
        assert calls[0].full_url == "http://localhost:13305/api/v1/chat/completions"

    @pytest.mark.anyio
    async def test_tool_calls_passthrough(self):
        prov = _provider()
        tool_calls = [{"id": "c1", "type": "function",
                       "function": {"name": "read", "arguments": "{}"}}]
        body = _chat_response("", tool_calls=tool_calls)
        with patch("urllib.request.urlopen", side_effect=_capturing_urlopen(body, [])):
            out = await prov.chat([{"role": "user", "content": "hi"}])
        parsed = json.loads(out)
        assert parsed["tool_calls"] == tool_calls

    @pytest.mark.anyio
    async def test_reasoning_fallback(self):
        prov = _provider()
        body = _chat_response("", reasoning_content="the answer is 42")
        with patch("urllib.request.urlopen", side_effect=_capturing_urlopen(body, [])):
            out = await prov.chat([{"role": "user", "content": "hi"}])
        assert out == "the answer is 42"

    @pytest.mark.anyio
    async def test_empty_returns_no_output(self):
        prov = _provider()
        body = _chat_response("")
        with patch("urllib.request.urlopen", side_effect=_capturing_urlopen(body, [])):
            out = await prov.chat([{"role": "user", "content": "hi"}])
        assert out == "(no output)"

    @pytest.mark.anyio
    async def test_transport_error_marks_connection_failure(self):
        prov = _provider()

        def boom(req, timeout=None):
            raise urllib.error.URLError("connection refused")

        with patch("urllib.request.urlopen", side_effect=boom):
            out = await prov.chat([{"role": "user", "content": "hi"}])
        assert out.startswith("[Error:")
        assert is_connection_failure(out)

    @pytest.mark.anyio
    async def test_400_reported(self):
        prov = _provider()

        def boom(req, timeout=None):
            raise _http_error(req, 400, b'{"error":"bad request"}')

        with patch("urllib.request.urlopen", side_effect=boom):
            out = await prov.chat([{"role": "user", "content": "hi"}])
        assert out.startswith("[Error: lemonade API request failed")

    @pytest.mark.anyio
    async def test_context_overflow_reports_actionable_hint(self):
        """FastFlowLM's "Max length reached!" (context too small) gets a hint."""
        prov = _provider()
        body = (
            b'{"error":{"code":400,"details":{"backend":"FastFlowLM","response":'
            b'{"error":{"code":400,"message":"Max length reached!",'
            b'"type":"model_error"}}},"message":"Max length reached!",'
            b'"status_code":400,"type":"model_error"}}'
        )

        def boom(req, timeout=None):
            raise _http_error(req, 400, body)

        with patch("urllib.request.urlopen", side_effect=boom):
            out = await prov.chat([{"role": "user", "content": "hi"}])
        assert "context is too small" in out
        assert "--ctx-size" in out

    @pytest.mark.anyio
    async def test_tools_rejected_retries_without_tools(self):
        prov = _provider()
        calls = {"n": 0}

        def fake(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _http_error(req, 400, b'{"error":"tools are not supported"}')
            return _FakeResponse(_chat_response("ok"))

        with patch("urllib.request.urlopen", side_effect=fake):
            out = await prov.chat([{"role": "user", "content": "hi"}], tools=_TOOLS)
        assert out == "ok"
        assert calls["n"] == 2

    @pytest.mark.anyio
    async def test_transient_5xx_retried(self):
        prov = _provider()
        calls = {"n": 0}

        def flaky(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _http_error(req, 503, b"busy")
            return _FakeResponse(_chat_response("ok"))

        with patch("urllib.request.urlopen", side_effect=flaky):
            out = await prov.chat([{"role": "user", "content": "hi"}])
        assert out == "ok"
        assert calls["n"] == 2


# ---------------------------------------------------------------------------
#  list_models / health_check
# ---------------------------------------------------------------------------

class TestCatalog:
    def test_list_models_namespaced(self):
        prov = _provider()
        body = json.dumps(
            {"data": [{"id": "qwen3.5-4b-FLM"}, {"id": "Qwen3-4B-Hybrid"}]}
        ).encode()
        with patch("urllib.request.urlopen", side_effect=_capturing_urlopen(body, [])):
            models = prov.list_models()
        assert models == ["lemonade/Qwen3-4B-Hybrid", "lemonade/qwen3.5-4b-FLM"]

    def test_list_models_unreachable_returns_empty(self):
        prov = _provider()
        with patch("urllib.request.urlopen", side_effect=OSError("offline")):
            assert prov.list_models() == []

    def test_health_check_ok(self):
        prov = _provider()
        body = json.dumps({"data": [{"id": "qwen3.5-4b-FLM"}]}).encode()
        with patch("urllib.request.urlopen", side_effect=_capturing_urlopen(body, [])):
            ok, note = prov.health_check()
        assert ok and "1 model" in note

    def test_health_check_unreachable(self):
        prov = _provider()
        with patch("urllib.request.urlopen", side_effect=OSError("offline")):
            ok, note = prov.health_check()
        assert not ok and "unreachable" in note.lower()


# ---------------------------------------------------------------------------
#  config
# ---------------------------------------------------------------------------

class TestResidency:
    def test_context_length_reads_server_window(self):
        prov = _provider()
        body = json.dumps(
            {"data": [{"id": "qwen3.5-4b-FLM", "context_length": 65536}]}
        ).encode()
        with patch("urllib.request.urlopen", side_effect=_capturing_urlopen(body, [])):
            assert prov.context_length() == 65536
        assert prov.context_limit == 65536

    def test_ensure_model_loaded_posts_bare_name(self):
        prov = _provider()
        calls: list = []

        def fake(req, timeout=None):
            calls.append(req)
            if req.full_url.endswith("/load"):
                return _FakeResponse(
                    b'{"status":"success","model_name":"qwen3.5-4b-FLM"}'
                )
            return _FakeResponse(json.dumps(
                {"data": [{"id": "qwen3.5-4b-FLM", "context_length": 65536}]}
            ).encode())

        with patch("urllib.request.urlopen", side_effect=fake):
            ok, note = prov.ensure_model_loaded()
        assert ok
        assert prov.context_limit == 65536
        load_calls = [c for c in calls if c.full_url.endswith("/load")]
        assert load_calls and json.loads(load_calls[0].data)["model_name"] == "qwen3.5-4b-FLM"

    def test_ensure_model_loaded_unreachable(self):
        prov = _provider()
        with patch("urllib.request.urlopen", side_effect=OSError("offline")):
            ok, note = prov.ensure_model_loaded()
        assert not ok and "unreachable" in note.lower()


class TestJevNpuOffload:
    def test_build_jev_engine_uses_lemonade_provider(self):
        from agent_core.jev_engine import build_jev_engine

        settings = SimpleNamespace(
            jev_model="lemonade/qwen3.5-0.8b-FLM",
            jev_provider="",
            jev_samples=3,
            jev_temperature=0.7,
            jev_max_tokens=64,
            jev_timeout=5.0,
            llm_provider="lmstudio",
            llm_providers=("lmstudio", "lemonade"),
            failover_strategy="ordered",
            lemonade_api_url="http://localhost:13305/api/v1",
            lemonade_model="",
        )
        engine = build_jev_engine(settings=settings)
        assert isinstance(engine.provider, LemonadeProvider)
        # No logprobs on the NPU provider -> the engine votes (self-consistency).
        assert getattr(engine.provider, "chat_logprobs", None) is None

    @pytest.mark.anyio
    async def test_ensure_ready_calls_ensure_model_loaded(self):
        from agent_core.jev_engine import JevEngine

        seen = {}

        class _Prov:
            model_name = "lemonade/x"

            def apply_profile(self, *a):
                pass

            def ensure_model_loaded(self):
                seen["called"] = True
                return True, "ok"

        engine = JevEngine(_Prov(), model_name="lemonade/x", timeout=5)
        await engine.ensure_ready()
        assert seen.get("called") is True


class TestConfig:
    def test_chain_accepts_lemonade(self, monkeypatch):
        monkeypatch.delenv("AGENT_LLM_PROVIDER", raising=False)
        monkeypatch.setenv("AGENT_LLM_PROVIDERS", "lemonade, lmstudio")
        settings = load_agent_settings()
        assert settings.llm_providers == ("lemonade", "lmstudio")
        assert settings.llm_provider == "lemonade"

    def test_lemonade_url_default(self, monkeypatch):
        monkeypatch.delenv("LEMONADE_API_URL", raising=False)
        s = AgentSettings(llm_provider="lmstudio", llm_providers=("lmstudio",))
        assert s.lemonade_api_url == "http://localhost:13305/api/v1"

    def test_lemonade_url_env_override(self, monkeypatch):
        monkeypatch.setenv("LEMONADE_API_URL", "http://127.0.0.1:8123/api/v1")
        s = AgentSettings(llm_provider="lmstudio", llm_providers=("lmstudio",))
        assert s.lemonade_api_url == "http://127.0.0.1:8123/api/v1"

    def test_bogus_provider_still_rejected(self):
        with pytest.raises(ConfigurationError):
            AgentSettings(llm_provider="lmstudio", llm_providers=("lmstudio", "bogus"))


# ---------------------------------------------------------------------------
#  model command integration
# ---------------------------------------------------------------------------

def _agent(current: str):
    return SimpleNamespace(
        llm=SimpleNamespace(model_name=current, _provider=None),
    )


def _single_lemonade_settings():
    """Single-provider lemonade settings — build_provider never touches LM Studio."""
    return SimpleNamespace(
        llm_provider="lemonade",
        llm_providers=("lemonade",),
        failover_strategy="ordered",
        lemonade_api_url="http://localhost:13305/api/v1",
        lemonade_model="",
    )


class TestModelCommandLemonade:
    def test_resolve_lemonade_match(self):
        from agent_core.commands.model_cmd import ModelCommand

        cmd = ModelCommand()
        models = ["lemonade/Qwen3-4B-Hybrid", "lemonade/qwen3.5-4b-FLM"]
        bare = cmd._resolve_lemonade_match("qwen3.5-4b-FLM", models)
        assert bare == "lemonade/qwen3.5-4b-FLM"
        prefixed = cmd._resolve_lemonade_match("lemonade/Qwen3-4B-Hybrid", models)
        assert prefixed == "lemonade/Qwen3-4B-Hybrid"
        assert cmd._resolve_lemonade_match("nothing-like-this", models) is None

    @pytest.mark.anyio
    async def test_switch_model_prefix_persists_lemonade(self, monkeypatch):
        from agent_core.commands.model_cmd import ModelCommand

        monkeypatch.setattr(
            "agent_core.commands.model_cmd.persist_model_choice", lambda *a, **k: None
        )
        monkeypatch.setattr(
            "agent_core.config.load_agent_settings", lambda: _single_lemonade_settings()
        )
        agent = _agent("laguna-s-2.1")
        await ModelCommand()._switch_model(["lemonade/qwen3.5-4b-FLM"], agent)
        assert agent.llm.model_name == "lemonade/qwen3.5-4b-FLM"
        assert isinstance(agent.llm._provider, LemonadeProvider)

    @pytest.mark.anyio
    async def test_load_model_namespaced_delegates_to_switch(self, monkeypatch):
        """Regression: `model load lemonade/<id>` must NOT be fuzzy-matched
        against LM Studio (it loaded qwen/qwen3.5-9b instead of the NPU model)."""
        from agent_core.commands.model_cmd import ModelCommand

        cmd = ModelCommand()
        seen: dict = {}

        async def fake_switch(args, agent):
            seen["args"] = args

        monkeypatch.setattr(cmd, "_switch_model", fake_switch)
        monkeypatch.setattr(
            cmd, "_fetch_models",
            lambda: (_ for _ in ()).throw(AssertionError("must not touch LM Studio")),
        )
        await cmd._load_model(["lemonade/qwen3.5-4b-FLM"], _agent("qwen/qwen3.5-9b"))
        assert seen["args"] == ["lemonade/qwen3.5-4b-FLM"]

    @pytest.mark.anyio
    async def test_switch_provider_lemonade_uses_live_catalog(self, monkeypatch):
        from agent_core.commands.model_cmd import ModelCommand

        persisted: dict = {}
        monkeypatch.setattr(
            "agent_core.constants.persist_model_choice",
            lambda name, provider=None: persisted.update(name=name, provider=provider),
        )
        monkeypatch.setattr(
            "agent_core.config.load_agent_settings", lambda: _single_lemonade_settings()
        )
        cmd = ModelCommand()
        monkeypatch.setattr(
            cmd, "_lemonade_catalog", lambda agent: ["lemonade/qwen3.5-4b-FLM"]
        )
        agent = _agent("laguna-s-2.1")
        await cmd._handle_provider(["lemonade"], agent)
        assert agent.llm.model_name == "lemonade/qwen3.5-4b-FLM"
        assert persisted == {"name": "lemonade/qwen3.5-4b-FLM", "provider": "lemonade"}
