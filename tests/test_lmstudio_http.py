"""Tests for LM Studio HTTP error handling (body surfacing + auto-load)."""
import urllib.error
from io import BytesIO
from unittest.mock import patch

from agent_core.llm.lmstudio import LMStudioProvider, _model_load_hint


class _FakeHTTPResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(req, body: bytes, code: int = 400):
    raise urllib.error.HTTPError(req.full_url, code, "Bad Request", {}, BytesIO(body))


class TestMakeRequest:
    def _provider(self):
        return LMStudioProvider(model_name="meta/muse-glimmer")

    def test_model_load_hint(self):
        assert _model_load_hint("model is not loaded, load it first")
        assert _model_load_hint("error: model not loaded")
        assert not _model_load_hint("invalid parameter: foo")

    def test_400_surfaces_body(self):
        prov = self._provider()
        body = b'{"error": "model is not loaded, load it first"}'

        def boom(req, timeout=None):
            _http_error(req, body)

        with patch("urllib.request.urlopen", side_effect=boom), patch(
            "agent_core.llm.lmstudio.load_model", return_value=(False, "load refused")
        ) as lm:
            try:
                prov._make_request({"model": "x"})
                assert False, "forventede RuntimeError"
            except RuntimeError as e:
                assert "400" in str(e)
                assert "model is not loaded" in str(e)
        lm.assert_called_once()

    def test_400_autoloads_and_retries(self):
        prov = self._provider()
        calls = {"n": 0}

        def fake(req, timeout=None):
            if calls["n"] == 0:
                calls["n"] += 1
                _http_error(req, b'{"error": "model is not loaded, load it first"}')
            calls["n"] += 1
            return _FakeHTTPResponse(b'{"choices": [{"message": {"content": "hej"}}]}')

        with patch("urllib.request.urlopen", side_effect=fake), patch(
            "agent_core.llm.lmstudio.load_model", return_value=(True, "loaded")
        ) as lm:
            result = prov._make_request({"model": "x"})
        assert result["choices"][0]["message"]["content"] == "hej"
        lm.assert_called_once_with("meta/muse-glimmer")

    def test_400_autoload_failed_reports_both_errors(self):
        prov = self._provider()
        body = b'{"error": "model is not loaded, load it first"}'

        def boom(req, timeout=None):
            _http_error(req, body)

        with patch("urllib.request.urlopen", side_effect=boom), patch(
            "agent_core.llm.lmstudio.load_model", return_value=(False, "load refused")
        ):
            try:
                prov._make_request({"model": "x"})
                assert False, "forventede RuntimeError"
            except RuntimeError as e:
                assert "auto-load failed" in str(e)

    def test_400_other_body_no_autoload(self):
        prov = self._provider()
        body = b'{"error": "invalid parameter: foo"}'

        def boom(req, timeout=None):
            _http_error(req, body)

        with patch("urllib.request.urlopen", side_effect=boom), patch(
            "agent_core.llm.lmstudio.load_model"
        ) as lm:
            try:
                prov._make_request({"model": "x"})
                assert False, "forventede RuntimeError"
            except RuntimeError as e:
                assert "invalid parameter" in str(e)
        lm.assert_not_called()
