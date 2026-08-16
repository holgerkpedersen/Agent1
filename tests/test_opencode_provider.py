"""Tests for the opencode LLM provider integration (decisions #007-#012)."""
import json
from unittest.mock import patch

from agent_core.constants import DEFAULT_OPENCODE_MODEL, persist_model_choice
from agent_core.llm.opencode_provider import OpencodeProvider, _TOOL_MAP
from agent_core.llm.provider import build_provider, provider_for


class TestProviderFor:
    def test_opencode_prefix_wins(self):
        assert provider_for("opencode-go/glm-5.2", "lmstudio") == "opencode"
        assert provider_for("opencode/deepseek-v4-flash", "lmstudio") == "opencode"

    def test_lmstudio_prefixes(self):
        assert provider_for("laguna-s-2.1", "opencode") == "lmstudio"
        assert provider_for("qwen3.8-27b", "opencode") == "lmstudio"

    def test_setting_applies_without_prefix(self):
        assert provider_for("model-x", "opencode") == "opencode"
        assert provider_for("model-x", "lmstudio") == "lmstudio"
        assert provider_for("model-x", "bogus") == "lmstudio"

    def test_persisted_provider_wins_over_setting(self):
        assert provider_for("reactagent-1.5b", "opencode", "lmstudio") == "lmstudio"
        assert provider_for("model-x", "lmstudio", "opencode") == "opencode"
        assert provider_for("model-x", "opencode", "bogus") == "opencode"
        assert provider_for("reactagent-1.5b", "opencode") == "opencode"  # unchanged without persisted


class TestBuildProvider:
    def test_lmstudio_default(self):
        settings = type("S", (), {"llm_provider": "lmstudio"})()
        prov = build_provider(settings, "laguna-s-2.1")
        assert type(prov).__name__ == "LMStudioProvider"

    def test_opencode_provider_built(self):
        settings = type("S", (), {
            "llm_provider": "opencode",
            "opencode_server_url": "http://127.0.0.1:4096",
            "opencode_password": "pw",
        })()
        prov = build_provider(settings, "opencode-go/glm-5.2")
        assert isinstance(prov, OpencodeProvider)
        assert prov.model_name == "opencode-go/glm-5.2"

    def test_persisted_lmstudio_keeps_lmstudio_provider(self, monkeypatch):
        settings = type("S", (), {"llm_provider": "opencode"})()
        monkeypatch.setattr(
            "agent_core.constants.load_model_json",
            lambda: {"model": "reactagent-1.5b", "provider": "lmstudio"},
        )
        prov = build_provider(settings, "reactagent-1.5b")
        assert type(prov).__name__ == "LMStudioProvider"


class _FakeHttp:
    """Stub urllib.request.urlopen returning a canned JSON response."""

    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class TestOpencodeChat:
    def _provider(self):
        return OpencodeProvider(
            model_name="opencode-go/glm-5.2",
            server_url="http://127.0.0.1:4096",
            read_store=False,
        )

    def test_returns_plain_text(self):
        prov = self._provider()
        session = {"id": "s1"}
        message = {
            "parts": [{"id": "p1", "type": "text", "text": "Hello from opencode"}],
        }
        responses = [session, message]

        def fake_urlopen(req, timeout=None):
            resp = _FakeHttp(responses.pop(0))
            return resp

        import asyncio
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            out = asyncio.run(prov.chat([{"role": "user", "content": "hi"}]))
        assert out == "Hello from opencode"

    def test_tool_calls_mapped_to_agent_tools(self):
        prov = self._provider()
        session = {"id": "s1"}
        message = {
            "parts": [{
                "id": "p1", "sessionID": "s1", "messageID": "m1", "type": "tool",
                "callID": "c1", "tool": "grep",
                "state": {"status": "pending", "input": {"path": ".", "pattern": "def x"},
                          "raw": "{}"},
            }],
        }

        def fake_urlopen(req, timeout=None):
            return _FakeHttp([session, message].pop(0)) if not hasattr(fake_urlopen, "done") else _FakeHttp(session)

        import asyncio
        calls = iter([session, message])

        def fake_urlopen2(req, timeout=None):
            return _FakeHttp(next(calls))

        with patch("urllib.request.urlopen", side_effect=fake_urlopen2):
            out = asyncio.run(prov.chat([{"role": "user", "content": "search"}]))
        parsed = json.loads(out)
        assert parsed["tool_calls"][0]["function"]["name"] == "search"  # grep -> search
        assert parsed["tool_calls"][0]["function"]["arguments"]

    def test_unmapped_tool_reported(self):
        prov = self._provider()
        session = {"id": "s1"}
        message = {
            "parts": [{
                "id": "p1", "type": "tool", "callID": "c1", "tool": "unknown_tool",
                "state": {"status": "pending", "input": {}},
            }],
        }

        import asyncio
        calls = iter([session, message])

        def fake_urlopen(req, timeout=None):
            return _FakeHttp(next(calls))

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            out = asyncio.run(prov.chat([{"role": "user", "content": "x"}]))
        parsed = json.loads(out)
        assert parsed["tool_calls"][0]["function"]["name"].startswith("unmapped:")

    def test_server_unreachable_returns_error(self):
        prov = self._provider()

        import asyncio
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            out = asyncio.run(prov.chat([{"role": "user", "content": "hi"}]))
        assert out.startswith("[Error")


class TestOpencodeRetry:
    """Retry-on-transient-5xx behavior (the workflow [analyze] HTTP 500 fix)."""

    def _provider(self, **kw):
        return OpencodeProvider(
            model_name="opencode-go/glm-5.2",
            api_key="sk-test",
            read_store=False,
            **kw,
        )

    @staticmethod
    def _http_error(code: int):
        import urllib.error
        return urllib.error.HTTPError("http://x", code, "boom", {}, None)

    def test_http_500_retried_then_succeeds(self):
        import asyncio
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            if calls["n"] < 3:
                raise self._http_error(500)
            return _FakeHttp({"choices": [{"message": {"content": "ok"}}]})

        prov = self._provider(max_retries=2, retry_base_delay=0.01)
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            out = asyncio.run(prov.chat([{"role": "user", "content": "hi"}]))
        assert out == "ok"
        assert calls["n"] == 3  # 2 failed attempts + 1 success

    def test_http_500_exhausted_returns_error(self):
        import asyncio

        def fake_urlopen(req, timeout=None):
            raise self._http_error(500)

        prov = self._provider(max_retries=2, retry_base_delay=0.01)
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            out = asyncio.run(prov.chat([{"role": "user", "content": "hi"}]))
        assert out.startswith("[Error: opencode API request failed")
        assert "500" in out

    def test_http_400_not_retried(self):
        import asyncio
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            raise self._http_error(400)

        prov = self._provider(max_retries=3, retry_base_delay=0.01)
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            out = asyncio.run(prov.chat([{"role": "user", "content": "hi"}]))
        assert out.startswith("[Error")
        assert calls["n"] == 1  # permanent error — no retry

    def test_timeout_retried_then_succeeds(self):
        import asyncio
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("read timed out")
            return _FakeHttp({"choices": [{"message": {"content": "recovered"}}]})

        prov = self._provider(max_retries=2, retry_base_delay=0.01)
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            out = asyncio.run(prov.chat([{"role": "user", "content": "hi"}]))
        assert out == "recovered"
        assert calls["n"] == 2


    def test_tool_map_covers_expected_names(self):
        for oc_name in ("bash", "read", "write", "edit", "list", "grep", "webfetch"):
            assert oc_name in _TOOL_MAP


class TestListModels:
    def test_lists_models_from_server(self):
        prov = OpencodeProvider("opencode-go/x", server_url="http://x:4096", read_store=False)
        payload = {
            "providers": [
                {"id": "opencode-go", "models": {"glm-5.2": {}, "kimi-k2.7-code": {}}},
                {"id": "opencode", "models": {"deepseek-v4-flash": {}}},
            ],
        }
        with patch("urllib.request.urlopen", return_value=_FakeHttp(payload)):
            models = prov.list_models()
        assert "opencode-go/glm-5.2" in models
        assert "opencode-go/kimi-k2.7-code" in models
        assert "opencode/deepseek-v4-flash" in models

    def test_offline_returns_empty(self):
        prov = OpencodeProvider("opencode-go/x", server_url="http://x:4096", read_store=False)
        with patch("urllib.request.urlopen", side_effect=OSError("offline")):
            assert prov.list_models() == []


class TestPersistProvider:
    def test_persist_model_choice_infers_provider(self, tmp_path, monkeypatch):
        from agent_core import constants as const
        monkeypatch.setattr(const, "MODEL_JSON_PATH", str(tmp_path / "model.json"))
        persist_model_choice("opencode-go/glm-5.2")
        data = const.load_model_json()
        assert data["model"] == "opencode-go/glm-5.2"
        assert data["provider"] == "opencode"
        persist_model_choice("laguna-s-2.1", provider="lmstudio")
        data = const.load_model_json()
        assert data["provider"] == "lmstudio"

    def test_resolve_model_opencode_provider(self, monkeypatch, tmp_path):
        from agent_core import constants as const
        monkeypatch.setattr(const, "MODEL_JSON_PATH", str(tmp_path / "model.json"))
        settings = type("S", (), {
            "llm_provider": "opencode",
            "opencode_model": "opencode-go/deepseek-v4-flash",
        })()
        with patch("agent_core.config.load_agent_settings", return_value=settings):
            assert const.resolve_model(None) == "opencode-go/deepseek-v4-flash"
        assert const.resolve_model("opencode-go/glm-5.2") == "opencode-go/glm-5.2"
        assert DEFAULT_OPENCODE_MODEL.startswith("opencode-go/")


class TestDirectApiMode:
    def test_api_mode_auto_detected_with_key(self):
        p = OpencodeProvider("opencode-go/x", api_key="sk-test", read_store=False)
        assert p.api_mode is True

    def test_server_mode_without_key(self):
        p = OpencodeProvider("opencode-go/x", read_store=False)
        assert p.api_mode is False

    def test_env_key_beats_store(self, monkeypatch):
        monkeypatch.setenv("OPENCODE_API_KEY", "sk-env")
        p = OpencodeProvider("opencode-go/x", read_store=False)
        assert p.api_key == "sk-env"

    def test_chat_api_returns_plain_text(self):
        import asyncio
        prov = OpencodeProvider("opencode-go/glm-5.2", api_key="sk-test", read_store=False)
        payload = {"choices": [{"message": {"role": "assistant", "content": "Direkte svar"}}]}
        with patch("urllib.request.urlopen", return_value=_FakeHttp(payload)):
            out = asyncio.run(prov.chat([{"role": "user", "content": "hi"}]))
        assert out == "Direkte svar"

    def test_chat_api_returns_tool_calls(self):
        import asyncio
        prov = OpencodeProvider("opencode-go/glm-5.2", api_key="sk-test", read_store=False)
        tc = {"id": "c1", "type": "function",
              "function": {"name": "search", "arguments": "{\"query\": \"x\"}"}}
        payload = {"choices": [{"message": {"role": "assistant", "content": "",
                                            "tool_calls": [tc]}}]}
        with patch("urllib.request.urlopen", return_value=_FakeHttp(payload)):
            out = asyncio.run(prov.chat([{"role": "user", "content": "søg"}],
                                        tools=[{"type": "function"}]))
        parsed = json.loads(out)
        assert parsed["tool_calls"][0]["function"]["name"] == "search"

    def test_chat_api_error(self):
        import asyncio
        prov = OpencodeProvider("opencode-go/x", api_key="sk-test", read_store=False)
        with patch("urllib.request.urlopen", side_effect=OSError("http 401")):
            out = asyncio.run(prov.chat([{"role": "user", "content": "hi"}]))
        assert out.startswith("[Error")

    def test_chat_api_error_surfaces_gateway_body(self):
        """A gateway 400 must show its explanation (e.g. orphan tool message)
        instead of the useless bare '400: Bad Request'."""
        import asyncio
        import urllib.error
        from io import BytesIO
        prov = OpencodeProvider("opencode-go/glm-5.2", api_key="sk-test", read_store=False)

        def boom(req, timeout=None):
            body = b'{"error": {"message": "Messages with role \'tool\' must be a response to a preceding message with \'tool_calls\'"}}'
            raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, BytesIO(body))

        with patch("urllib.request.urlopen", side_effect=boom):
            out = asyncio.run(prov.chat([{"role": "user", "content": "hi"}]))
        assert "400" in out
        assert "must be a response to a preceding message" in out

    def test_chat_api_sends_bearer_and_tools(self):
        import asyncio
        import urllib.request
        prov = OpencodeProvider("opencode-go/glm-5.2", api_key="sk-test", read_store=False)
        payload = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        seen = {}

        def fake_urlopen(req, timeout=None):
            seen["url"] = req.full_url
            seen["auth"] = req.get_header("Authorization")
            seen["body"] = json.loads(req.data)
            return _FakeHttp(payload)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            asyncio.run(prov.chat([{"role": "user", "content": "hi"}],
                                  tools=[{"type": "function", "function": {"name": "x"}}],
                                  max_tokens=123))
        assert seen["url"].endswith("/chat/completions")
        assert seen["auth"] == "Bearer sk-test"
        assert seen["body"]["tools"][0]["function"]["name"] == "x"
        assert seen["body"]["max_tokens"] == 123
        assert seen["body"]["model"] == "glm-5.2"  # hosted ids are unprefixed

    def test_list_models_via_api(self):
        prov = OpencodeProvider("opencode-go/x", api_key="sk-test", read_store=False)
        payload = {"data": [{"id": "a"}, {"id": "b"}]}
        with patch("urllib.request.urlopen", return_value=_FakeHttp(payload)):
            models = prov.list_models()
        assert models == ["opencode-go/a", "opencode-go/b"]

    def test_hosted_ids_prefixed_for_agent(self):
        prov = OpencodeProvider("opencode-go/x", api_key="sk-test", read_store=False)
        payload = {"data": [{"id": "deepseek-v4-flash"}, {"id": "glm-5.2"}]}
        with patch("urllib.request.urlopen", return_value=_FakeHttp(payload)):
            models = prov.list_models()
        assert models == ["opencode-go/deepseek-v4-flash", "opencode-go/glm-5.2"]

    def test_analyze_code_uses_api(self):
        import asyncio
        prov = OpencodeProvider("opencode-go/x", api_key="sk-test", read_store=False)
        payload = {"choices": [{"message": {"role": "assistant", "content": "lgtm"}}]}
        with patch("urllib.request.urlopen", return_value=_FakeHttp(payload)):
            out = asyncio.run(prov.analyze_code("def f(): pass"))
        assert out == "lgtm"
