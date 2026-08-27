"""Regression tests for the llama.cpp (llama-server) provider.

Covers:
  * provider routing for `llama/...` model names and `--provider llama`,
  * build_provider constructing a LlamaProvider pointed at llama_base_url,
  * chat payload shape (model/messages/tools/reasoning off),
  * list_models() parsing GET /v1/models,
  * the explicit guarantee that LlamaProvider performs NO LM Studio model
    management (no load_model / _open_chat auto-reload / lms CLI fallback).
"""
import json
from types import SimpleNamespace

import pytest

from agent_core.llm.provider import build_provider, provider_for
from agent_core.llm.llama_provider import LlamaProvider, discover_local_gguf_models


def _settings(provider="llama", llama_base_url="http://127.0.0.1:8080/v1"):
    return SimpleNamespace(
        llm_provider=provider,
        opencode_server_url="http://127.0.0.1:4096",
        opencode_password="",
        opencode_api_url="https://opencode.ai/zen/go/v1",
        opencode_api_key="",
        llama_base_url=llama_base_url,
    )


# ---------------------------------------------------------------------------
#  provider_for routing
# ---------------------------------------------------------------------------

class TestProviderForRouting:
    def test_llama_prefix_routes_to_llama(self):
        assert provider_for("llama/qwen3.8-flash-next") == "llama"

    def test_llama_persisted_provider(self):
        assert provider_for("something-else", "lmstudio", "llama") == "llama"

    def test_llama_provider_setting(self):
        assert provider_for("something-else", "llama", None) == "llama"

    def test_non_llama_name_untouched(self):
        assert provider_for("laguna-s-2.1") == "lmstudio"
        assert provider_for("opencode-go/hy3") == "opencode"


# ---------------------------------------------------------------------------
#  build_provider
# ---------------------------------------------------------------------------

class TestBuildProvider:
    def test_builds_llama_provider(self):
        prov = build_provider(_settings(), "llama/qwen3.8-flash-next")
        assert isinstance(prov, LlamaProvider)
        assert prov.model_name == "llama/qwen3.8-flash-next"
        assert prov.api_url == "http://127.0.0.1:8080/v1"

    def test_override_routes_to_llama(self):
        """A non-llama model name can be forced onto llama via override."""
        prov = build_provider(_settings("lmstudio"), "laguna-s-2.1", provider_override="llama")
        assert isinstance(prov, LlamaProvider)

    def test_uses_settings_llama_base_url(self):
        prov = build_provider(_settings(llama_base_url="http://10.0.0.5:9000/v1"), "llama/x")
        assert prov.api_url == "http://10.0.0.5:9000/v1"


# ---------------------------------------------------------------------------
#  _switch_model via the ModelCommand
# ---------------------------------------------------------------------------

class TestSwitchModelLlama:
    def setup_method(self):
        from agent_core.commands.model_cmd import ModelCommand
        self.cmd = ModelCommand()

    def test_llama_prefix_switch(self, monkeypatch, capsys):
        """`model llama/qwen3.8-flash-next` routes to LlamaProvider and
        persists provider=llama.

        When the llama-server is unreachable, the switch keeps the typed
        routing label (chat will later self-heal on the first 400).  We mock
        the server probe so the test is deterministic regardless of whether a
        real llama-server happens to be running on 127.0.0.1:8080.
        """
        agent = SimpleNamespace(
            llm=SimpleNamespace(model_name="laguna-s-2.1", _provider=None)
        )
        monkeypatch.setattr("agent_core.config.load_agent_settings", lambda: _settings())
        monkeypatch.setattr("agent_core.constants.load_model_json", lambda: {})
        monkeypatch.setattr(
            "agent_core.llm.llama_provider.LlamaProvider.refresh_server_model_id",
            lambda self: None,
        )
        persisted = {}
        monkeypatch.setattr(
            "agent_core.commands.model_cmd.persist_model_choice",
            lambda name, provider=None: persisted.update(model=name, provider=provider),
        )

        import asyncio
        asyncio.get_event_loop().run_until_complete(
            self.cmd._switch_model(["llama/qwen3.8-flash-next"], agent)
        )

        assert isinstance(agent.llm._provider, LlamaProvider)
        assert agent.llm.model_name == "llama/qwen3.8-flash-next"
        assert persisted == {"model": "llama/qwen3.8-flash-next", "provider": "llama"}

    def test_explicit_provider_flag_llama(self, monkeypatch, capsys):
        """`model <other> --provider llama` is accepted and routes to LlamaProvider.

        With the server probe mocked as unreachable the typed label is kept.
        """
        agent = SimpleNamespace(
            llm=SimpleNamespace(model_name="laguna-s-2.1", _provider=None)
        )
        monkeypatch.setattr("agent_core.config.load_agent_settings", lambda: _settings())
        monkeypatch.setattr("agent_core.constants.load_model_json", lambda: {})
        monkeypatch.setattr(
            "agent_core.llm.llama_provider.LlamaProvider.refresh_server_model_id",
            lambda self: None,
        )
        monkeypatch.setattr("agent_core.commands.model_cmd.persist_model_choice", lambda *a, **k: None)

        import asyncio
        asyncio.get_event_loop().run_until_complete(
            self.cmd._switch_model(["qwen3.8-flash-next", "--provider", "llama"], agent)
        )

        assert isinstance(agent.llm._provider, LlamaProvider)
        assert agent.llm.model_name == "llama/qwen3.8-flash-next"


# ---------------------------------------------------------------------------
#  chat payload shape
# ---------------------------------------------------------------------------

class TestChatPayload:
    def test_basic_payload(self):
        prov = LlamaProvider(model_name="llama/x", api_url="http://h/v1")
        payload = prov._build_payload([{"role": "user", "content": "hi"}])
        # The routing prefix is stripped at the HTTP boundary (server id is bare).
        assert payload["model"] == "x"
        # But the provider keeps the prefix internally for routing/persistence.
        assert prov.model_name == "llama/x"
        assert payload["messages"] == [{"role": "user", "content": "hi"}]
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 50000
        assert "tools" not in payload

    def test_tools_included(self):
        prov = LlamaProvider(model_name="llama/x", api_url="http://h/v1")
        tools = [{"type": "function", "function": {"name": "f"}}]
        payload = prov._build_payload([{"role": "user", "content": "hi"}], tools=tools)
        assert payload["tools"] == tools

    def test_disable_thinking_sets_reasoning_off(self):
        prov = LlamaProvider(model_name="llama/x", api_url="http://h/v1")
        payload = prov._build_payload(
            [{"role": "user", "content": "hi"}], disable_thinking=True
        )
        assert payload["reasoning"] == "off"
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}

    def test_stream_flag(self):
        prov = LlamaProvider(model_name="llama/x", api_url="http://h/v1")
        payload = prov._build_payload([{"role": "user", "content": "hi"}], stream=True)
        assert payload["stream"] is True

    def test_request_strips_llama_prefix(self):
        """The llama-server only knows the bare model id; the routing prefix
        must be stripped in the actual HTTP request payload."""
        prov = LlamaProvider(model_name="llama/ggml-org/gemma-4-e4b-it-GGUF:Q4_0", api_url="http://h/v1")
        payload = prov._build_payload([{"role": "user", "content": "hi"}])
        assert payload["model"] == "ggml-org/gemma-4-e4b-it-GGUF:Q4_0"
        assert prov.model_name == "llama/ggml-org/gemma-4-e4b-it-GGUF:Q4_0"

    def test_init_normalizes_prefixed_name(self):
        """A bare id passed to the constructor is normalized to the llama/ prefix."""
        prov = LlamaProvider(model_name="gemma-4-e4b", api_url="http://h/v1")
        assert prov.model_name == "llama/gemma-4-e4b"
        assert prov._server_model_id() == "gemma-4-e4b"

    def test_list_models_returns_prefixed_ids(self, monkeypatch):
        body = json.dumps({"data": [{"id": "ggml-org/gemma-4-e4b-it-GGUF:Q4_0"}]}).encode()

        class _Resp:
            def read(self):
                return body

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=15: _Resp())
        monkeypatch.setattr(
            "agent_core.llm.llama_provider.discover_local_gguf_models",
            lambda: [],
        )
        prov = LlamaProvider(model_name="llama/x", api_url="http://h/v1")
        assert prov.list_models() == ["llama/ggml-org/gemma-4-e4b-it-GGUF:Q4_0"]


# ---------------------------------------------------------------------------
#  list_models
# ---------------------------------------------------------------------------

class TestListModels:
    def test_parses_models_endpoint(self, monkeypatch):
        body = json.dumps({"data": [{"id": "qwen3.8-flash-next"}, {"id": "llama-3.1-8b"}]}).encode()

        class _Resp:
            def read(self):
                return body

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(
            "urllib.request.urlopen", lambda req, timeout=15: _Resp()
        )
        monkeypatch.setattr(
            "agent_core.llm.llama_provider.discover_local_gguf_models",
            lambda: [],
        )
        prov = LlamaProvider(model_name="llama/x", api_url="http://h/v1")
        assert prov.list_models() == ["llama/llama-3.1-8b", "llama/qwen3.8-flash-next"]

    def test_empty_on_failure(self, monkeypatch):
        def _boom(*a, **k):
            raise OSError("connection refused")

        monkeypatch.setattr("urllib.request.urlopen", _boom)
        monkeypatch.setattr(
            "agent_core.llm.llama_provider.discover_local_gguf_models",
            lambda: [],
        )
        prov = LlamaProvider(model_name="llama/x", api_url="http://h/v1")
        assert prov.list_models() == []

    def test_merges_server_and_local(self, monkeypatch):
        """list_models merges live server models with local GGUF discovery."""
        body = json.dumps({"data": [{"id": "qwen3.8-flash-next"}]}).encode()

        class _Resp:
            def read(self):
                return body

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=15: _Resp())
        monkeypatch.setattr(
            "agent_core.llm.llama_provider.discover_local_gguf_models",
            lambda: ["llama/unsloth/Qwen3-Coder-30B-A3B-Instruct-Q4_K_S"],
        )
        prov = LlamaProvider(model_name="llama/x", api_url="http://h/v1")
        result = prov.list_models()
        # Server model first, local-only model second.
        assert result == [
            "llama/qwen3.8-flash-next",
            "llama/unsloth/Qwen3-Coder-30B-A3B-Instruct-Q4_K_S",
        ]

    def test_dedup_server_and_local(self, monkeypatch):
        """A model that is both server-loaded and locally discovered appears once."""
        body = json.dumps({"data": [{"id": "gemma-4-e4b-it-Q4_K_S"}]}).encode()

        class _Resp:
            def read(self):
                return body

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=15: _Resp())
        monkeypatch.setattr(
            "agent_core.llm.llama_provider.discover_local_gguf_models",
            lambda: ["llama/gemma-4-e4b-it-Q4_K_S"],
        )
        prov = LlamaProvider(model_name="llama/x", api_url="http://h/v1")
        assert prov.list_models() == ["llama/gemma-4-e4b-it-Q4_K_S"]

    def test_local_only_when_server_unreachable(self, monkeypatch):
        """If the server is down, local GGUF models are still listed."""
        def _boom(*a, **k):
            raise OSError("connection refused")

        monkeypatch.setattr("urllib.request.urlopen", _boom)
        monkeypatch.setattr(
            "agent_core.llm.llama_provider.discover_local_gguf_models",
            lambda: ["llama/unsloth/Qwen3.5-9B-Q8_0"],
        )
        prov = LlamaProvider(model_name="llama/x", api_url="http://h/v1")
        assert prov.list_models() == ["llama/unsloth/Qwen3.5-9B-Q8_0"]


# ---------------------------------------------------------------------------
#  discover_local_gguf_models
# ---------------------------------------------------------------------------

class TestDiscoverLocalGguf:
    def test_returns_empty_when_dir_missing(self, monkeypatch):
        monkeypatch.setattr(
            "agent_core.llm.llama_provider._lmstudio_models_dir",
            lambda: None,
        )
        assert discover_local_gguf_models() == []

    def test_strips_gguf_extension(self, monkeypatch, tmp_path):
        models_dir = tmp_path / "models"
        (models_dir / "unsloth" / "Qwen3-Coder-30B-A3B-Instruct-GGUF").mkdir(parents=True)
        (models_dir / "unsloth" / "Qwen3-Coder-30B-A3B-Instruct-GGUF" / "Qwen3-Coder-30B-A3B-Instruct-Q4_K_S.gguf").touch()
        monkeypatch.setattr(
            "agent_core.llm.llama_provider._lmstudio_models_dir",
            lambda: str(models_dir),
        )
        result = discover_local_gguf_models()
        assert result == ["llama/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_S"]

    def test_collapses_shards(self, monkeypatch, tmp_path):
        """Sharded models (-00001-of-00003.gguf) collapse to a single id."""
        models_dir = tmp_path / "models"
        (models_dir / "unsloth" / "Qwen3.8-Flash-Next-GGUF").mkdir(parents=True)
        for i in range(1, 4):
            (models_dir / "unsloth" / "Qwen3.8-Flash-Next-GGUF" /
             f"Qwen3.8-Flash-Next-UD-IQ3_XXS-0000{i}-of-00003.gguf").touch()
        monkeypatch.setattr(
            "agent_core.llm.llama_provider._lmstudio_models_dir",
            lambda: str(models_dir),
        )
        result = discover_local_gguf_models()
        assert len(result) == 1
        assert result[0] == "llama/unsloth/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-IQ3_XXS"

    def test_skips_mmproj_files(self, monkeypatch, tmp_path):
        """Vision projection files (mmproj-*.gguf) are not chat models."""
        models_dir = tmp_path / "models"
        (models_dir / "gemma-4-12b-it-GGUF").mkdir(parents=True)
        (models_dir / "gemma-4-12b-it-GGUF" / "gemma-4-12b-it-Q4_K_M.gguf").touch()
        (models_dir / "gemma-4-12b-it-GGUF" / "mmproj-gemma-4-12b-it-BF16.gguf").touch()
        monkeypatch.setattr(
            "agent_core.llm.llama_provider._lmstudio_models_dir",
            lambda: str(models_dir),
        )
        result = discover_local_gguf_models()
        assert len(result) == 1
        assert result[0] == "llama/gemma-4-12b-it-GGUF/gemma-4-12b-it-Q4_K_M"

    def test_uses_env_override(self, monkeypatch, tmp_path):
        """LMSTUDIO_MODELS_DIR env var overrides the default path."""
        models_dir = tmp_path / "custom_models"
        models_dir.mkdir(parents=True)
        (models_dir / "my-model.gguf").touch()
        monkeypatch.setenv("LMSTUDIO_MODELS_DIR", str(models_dir))
        result = discover_local_gguf_models()
        assert result == ["llama/my-model"]

    def test_sorted_output(self, monkeypatch, tmp_path):
        """Results are sorted alphabetically."""
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)
        (models_dir / "z-model.gguf").touch()
        (models_dir / "a-model.gguf").touch()
        (models_dir / "m-model.gguf").touch()
        monkeypatch.setattr(
            "agent_core.llm.llama_provider._lmstudio_models_dir",
            lambda: str(models_dir),
        )
        result = discover_local_gguf_models()
        assert result == ["llama/a-model", "llama/m-model", "llama/z-model"]


# ---------------------------------------------------------------------------
#  No LM Studio management coupling
# ---------------------------------------------------------------------------

class TestNoLmStudioManagement:
    def test_chat_does_not_call_lmstudio_load(self, monkeypatch):
        """LlamaProvider.chat must never attempt LM Studio model loading."""
        calls = {"load": 0, "lms": 0}
        monkeypatch.setattr(
            "agent_core.llm.lmstudio.load_model",
            lambda *a, **k: calls.__setitem__("load", calls["load"] + 1),
        )
        import subprocess
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **k: calls.__setitem__("lms", calls["lms"] + 1),
        )

        body = json.dumps({
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }).encode()

        class _Resp:
            def read(self):
                return body

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=600: _Resp())

        prov = LlamaProvider(model_name="llama/x", api_url="http://h/v1")
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            prov.chat([{"role": "user", "content": "hi"}])
        )
        assert result == "ok"
        assert calls["load"] == 0
        assert calls["lms"] == 0


# ---------------------------------------------------------------------------
#  LlamaProvider.shutdown
# ---------------------------------------------------------------------------

class TestShutdown:
    def test_shutdown_calls_root_endpoint(self, monkeypatch):
        """shutdown() POSTs to the server root /shutdown (not /v1/shutdown)."""
        captured: dict = {}

        class _Resp:
            def read(self):
                return b""

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def _fake_urlopen(req, timeout=10):
            captured["url"] = req.full_url
            captured["method"] = req.get_method()
            return _Resp()

        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
        prov = LlamaProvider(model_name="llama/x", api_url="http://127.0.0.1:8080/v1")
        ok, msg = prov.shutdown()
        assert ok is True
        assert captured["url"] == "http://127.0.0.1:8080/shutdown"
        assert captured["method"] == "POST"

    def test_shutdown_returns_success_when_already_stopped(self, monkeypatch):
        """If the server is already down, shutdown reports success (no-op)."""
        import urllib.error

        def _fake_urlopen(req, timeout=10):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
        prov = LlamaProvider(model_name="llama/x", api_url="http://127.0.0.1:8080/v1")
        ok, msg = prov.shutdown()
        assert ok is True
        assert "already stopped" in msg

    def test_shutdown_returns_failure_on_http_error(self, monkeypatch):
        """A non-404 HTTP error is reported as a failure."""
        import urllib.error

        def _fake_urlopen(req, timeout=10):
            raise urllib.error.HTTPError("http://x/shutdown", 500, "Server Error", {}, None)

        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
        prov = LlamaProvider(model_name="llama/x", api_url="http://127.0.0.1:8080/v1")
        ok, msg = prov.shutdown()
        assert ok is False
        assert "500" in msg

    def test_server_base_url_strips_v1(self):
        """_server_base_url strips the /v1 suffix for management endpoints."""
        prov = LlamaProvider(model_name="llama/x", api_url="http://127.0.0.1:8080/v1")
        assert prov._server_base_url() == "http://127.0.0.1:8080"

    def test_server_base_url_without_v1(self):
        """_server_base_url handles URLs that don't end in /v1."""
        prov = LlamaProvider(model_name="llama/x", api_url="http://example.com")
        assert prov._server_base_url() == "http://example.com"


# ---------------------------------------------------------------------------
#  server model-id self-healing (the "model not found" 400 case)
# ---------------------------------------------------------------------------

class TestServerModelIdHealing:
    """llama-server serves exactly one model (the --model it started with) and
    registers it under that exact id.  The agent's model_name is a routing
    label that may not match, so the provider must discover the real served id
    from GET /v1/models and use it for the request `model` field — otherwise
    every request 400s with "model '<label>' not found".
    """

    def test_server_model_id_prefers_cached_real_id(self):
        """_server_model_id returns the discovered server id, not the label."""
        prov = LlamaProvider(model_name="llama/oss-20b", api_url="http://127.0.0.1:8080/v1")
        # Simulate discovery of the real served id.
        prov._cached_server_model_id = "gpt-oss-20b-MXFP4"
        assert prov._server_model_id() == "gpt-oss-20b-MXFP4"

    def test_server_model_id_falls_back_to_label(self):
        """Without a cached server id, fall back to the bare routing label."""
        prov = LlamaProvider(model_name="llama/oss-20b", api_url="http://127.0.0.1:8080/v1")
        assert prov._server_model_id() == "oss-20b"

    def test_refresh_server_model_id_returns_first_served(self, monkeypatch):
        """refresh_server_model_id parses GET /v1/models and caches the id."""
        import urllib.request

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps({
                    "data": [{"id": "gpt-oss-20b-MXFP4"}]
                }).encode()

        captured = {}

        def _fake_urlopen(req, timeout=10):
            captured["url"] = req.full_url
            return _Resp()

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
        prov = LlamaProvider(model_name="llama/oss-20b", api_url="http://127.0.0.1:8080/v1")
        got = prov.refresh_server_model_id()
        assert got == "gpt-oss-20b-MXFP4"
        assert prov._cached_server_model_id == "gpt-oss-20b-MXFP4"
        assert captured["url"] == "http://127.0.0.1:8080/v1/models"

    def test_chat_self_heals_model_not_found(self, monkeypatch, capsys):
        """A 400 'model not found' for the label triggers a retry with the
        real served id discovered from GET /v1/models.

        This is the exact scenario from the bug report: the server was started
        with --model gpt-oss-20b-MXFP4 but the agent requested 'oss-20b'.
        """
        import urllib.error
        import urllib.request

        real_id = "gpt-oss-20b-MXFP4"

        class _ModelsResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps({"data": [{"id": real_id}]}).encode()

        class _ChatOk:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps({
                    "choices": [{"message": {"content": "hello"}}]
                }).encode()

        calls: list[str] = []

        def _fake_urlopen(req, timeout=10):
            calls.append(req.full_url)
            if req.full_url.endswith("/models"):
                return _ModelsResp()
            # First chat attempt uses the label -> 400 model not found.
            if req.full_url.endswith("/chat/completions"):
                body = req.data.decode()
                if '"model": "oss-20b"' in body:
                    raise urllib.error.HTTPError(
                        req.full_url, 400,
                        '{"error":{"message":"model \'oss-20b\' not found"}}',
                        {}, None)
                return _ChatOk()
            raise urllib.error.HTTPError(req.full_url, 404, "nf", {}, None)

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
        prov = LlamaProvider(model_name="llama/oss-20b", api_url="http://127.0.0.1:8080/v1")
        out = prov._make_request({
            "model": prov._server_model_id(),
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert out["choices"][0]["message"]["content"] == "hello"
        assert prov._cached_server_model_id == real_id
        # The label was tried, then the real id succeeded.
        assert calls.count("http://127.0.0.1:8080/v1/chat/completions") == 2

    def test_chat_no_heal_when_server_id_matches_label(self, monkeypatch):
        """When the served id equals the label there is no extra retry."""
        import urllib.request

        class _ChatOk:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps({
                    "choices": [{"message": {"content": "ok"}}]
                }).encode()

        calls: list[str] = []

        def _fake_urlopen(req, timeout=10):
            calls.append(req.full_url)
            return _ChatOk()

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
        prov = LlamaProvider(model_name="llama/gpt-oss-20b-MXFP4",
                             api_url="http://127.0.0.1:8080/v1")
        out = prov._make_request({
            "model": prov._server_model_id(),
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert out["choices"][0]["message"]["content"] == "ok"
        # Only the chat call; no /models probe, no retry.
        assert calls == ["http://127.0.0.1:8080/v1/chat/completions"]
