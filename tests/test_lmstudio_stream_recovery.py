"""Regression tests: streaming chat must survive another shell's model swap.

User scenario (2026-08-23): shell 1 streams a long answer while shell 2
switches models — LM Studio evicts shell 1's model from VRAM, and the NEXT
streamed request gets HTTP 400 "model is not loaded, load it first".

The non-streaming path already recovered (``_make_request`` auto-loaded the
pinned model and retried once); ``chat_stream`` used raw ``urllib.urlopen``
and simply failed.  Both paths now share ``LMStudioProvider._open_chat``,
which reloads OUR pinned model on that specific 400 and retries once.

Contract under test:
1. streaming 400 "not loaded" → load_model(self.model_name) called with the
   session's OWN model (never the other shell's) + retry succeeds;
2. auto-load failure surfaces BOTH errors ("auto-load failed: ...");
3. a 400 that is NOT a load hint does NOT trigger any load;
4. non-streaming path unchanged: same recovery through _open_chat.
"""
import asyncio
import urllib.error
from io import BytesIO
from unittest.mock import patch

from agent_core.llm.lmstudio import LMStudioProvider


class _FakeHTTPResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(req, body: bytes, code: int = 400):
    raise urllib.error.HTTPError(req.full_url, code, "Bad Request", {}, BytesIO(body))


_NOT_LOADED = b'{"error": "model is not loaded, load it first"}'


class TestStreamAutoReloadOnEvictedModel:
    def test_stream_400_not_loaded_autoloads_pinned_model_and_retries(self):
        """Shell 2 put qwen in VRAM; our pinned laguna request 400s — the
        reload must target laguna (self.model_name), not what is in VRAM."""
        prov = LMStudioProvider(model_name="laguna-s-2.1")
        calls = {"n": 0}

        def fake(req, timeout=None):
            if calls["n"] == 0:
                calls["n"] += 1
                _http_error(req, _NOT_LOADED)
            calls["n"] += 1
            return _FakeHTTPResponse(
                b'data: {"choices": [{"delta": {"content": "hej"}}]}\n\ndata: [DONE]\n\n'
            )

        with patch("urllib.request.urlopen", side_effect=fake), patch(
            "agent_core.llm.lmstudio.load_model", return_value=(True, "loaded")
        ) as lm:
            result = asyncio.run(prov.chat_stream([{"role": "user", "content": "hi"}]))

        assert result == "hej"
        assert lm.call_count == 1
        # The PINNED model is reloaded — never whatever the other shell loaded.
        lm.assert_called_once_with("laguna-s-2.1")

    def test_stream_autoload_failure_reports_both_errors(self):
        prov = LMStudioProvider(model_name="laguna-s-2.1")

        def boom(req, timeout=None):
            _http_error(req, _NOT_LOADED)

        with patch("urllib.request.urlopen", side_effect=boom), patch(
            "agent_core.llm.lmstudio.load_model",
            return_value=(False, "not enough VRAM"),
        ):
            result = asyncio.run(prov.chat_stream([{"role": "user", "content": "hi"}]))

        assert "auto-load failed" in result
        assert "not enough VRAM" in result

    def test_stream_other_400_does_not_trigger_load(self):
        prov = LMStudioProvider(model_name="laguna-s-2.1")

        def boom(req, timeout=None):
            _http_error(req, b'{"error": "invalid parameter: foo"}')

        with patch("urllib.request.urlopen", side_effect=boom), patch(
            "agent_core.llm.lmstudio.load_model"
        ) as lm:
            result = asyncio.run(prov.chat_stream([{"role": "user", "content": "hi"}]))

        assert "invalid parameter" in result
        lm.assert_not_called()


class TestNonStreamingRecoveryUnchanged:
    def test_make_request_still_autoloads_via_shared_opener(self):
        """The original fix keeps working — now routed through _open_chat."""
        prov = LMStudioProvider(model_name="meta/muse-glimmer")
        calls = {"n": 0}

        def fake(req, timeout=None):
            if calls["n"] == 0:
                calls["n"] += 1
                _http_error(req, _NOT_LOADED)
            calls["n"] += 1
            return _FakeHTTPResponse(b'{"choices": [{"message": {"content": "ok"}}]}')

        with patch("urllib.request.urlopen", side_effect=fake), patch(
            "agent_core.llm.lmstudio.load_model", return_value=(True, "loaded")
        ) as lm:
            result = prov._make_request({"model": "x"})

        assert result["choices"][0]["message"]["content"] == "ok"
        lm.assert_called_once_with("meta/muse-glimmer")

    def test_open_chat_reraises_non_load_http_errors(self):
        """A genuine 4xx/5xx must surface its body, not be swallowed."""
        prov = LMStudioProvider(model_name="meta/muse-glimmer")

        def boom(req, timeout=None):
            _http_error(req, b'{"error": "context length exceeded"}', code=500)

        req = urllib.request.Request("http://localhost:1234/v1/chat/completions")
        with patch("urllib.request.urlopen", side_effect=boom), patch(
            "agent_core.llm.lmstudio.load_model"
        ) as lm:
            try:
                prov._open_chat(req, timeout=5)
                assert False, "expected RuntimeError"
            except RuntimeError as e:
                assert "500" in str(e)
                assert "context length exceeded" in str(e)
        lm.assert_not_called()
