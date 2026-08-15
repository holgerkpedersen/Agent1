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

    def test_tool_map_covers_expected_names(self):
        for oc_name in ("bash", "read", "write", "edit", "list", "grep", "webfetch"):
            assert oc_name in _TOOL_MAP


class TestListModels:
    def test_lists_models_from_server(self):
        prov = OpencodeProvider("opencode-go/x", server_url="http://x:4096")
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
        prov = OpencodeProvider("opencode-go/x", server_url="http://x:4096")
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
