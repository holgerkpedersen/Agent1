"""Regression tests: transient-HTTP retries for LM Studio (plan item B-#7).

Before this fix ``LMStudioProvider`` retried only Timeout/ConnectionReset/
ConnectionRefused — an LM Studio HTTP 429/5xx surfaced as ``[Error: ...]``
with ZERO retries while the hosted opencode provider quietly backed off and
succeeded on the same blip (``opencode_provider._with_retry``).  Now:

- ``_open_chat`` raises :class:`TransientHTTPError` for 429/500/502/503/504;
- the provider's default ``RetryPolicy`` retries it with backoff;
- permanent statuses (400 invalid parameter, 404 ...) still fail fast.

Retry tests go through ``prov.chat(...)`` — the REAL code path, where
``execute_with_retry`` wraps ``_make_request`` — never around it.
"""
from __future__ import annotations

import asyncio
import json
import urllib.error
from io import BytesIO
from unittest.mock import patch

import pytest

from agent_core.llm.lmstudio import LMStudioProvider
from agent_core.llm.retry import TRANSIENT_HTTP_STATUSES, TransientHTTPError


class _FakeResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(req, code: int, body: bytes) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(req.full_url, code, "Err", {}, BytesIO(body))


def _provider() -> LMStudioProvider:
    prov = LMStudioProvider(model_name="meta/muse-glimmer")
    # Zero backoff so the test does not actually sleep.
    prov.retry_policy.base_delay = 0.0
    return prov


class TestTransientStatuses:
    def test_status_set_matches_opencode_provider(self) -> None:
        from agent_core.llm.opencode_provider import _TRANSIENT_HTTP_STATUSES

        assert TRANSIENT_HTTP_STATUSES == _TRANSIENT_HTTP_STATUSES

    @pytest.mark.parametrize("status", sorted(TRANSIENT_HTTP_STATUSES))
    def test_transient_status_raises_typed_error(self, status: int) -> None:
        prov = _provider()

        def boom(req, timeout=None):
            raise _http_error(req, status, b"server busy")

        with patch("urllib.request.urlopen", side_effect=boom), patch(
            "agent_core.llm.lmstudio.load_model", return_value=(False, "no")
        ):
            with pytest.raises(TransientHTTPError) as excinfo:
                prov._make_request({"model": "x"})
        assert excinfo.value.status == status


class TestRetryBehaviour:
    def test_503_retried_until_success(self) -> None:
        """THE regression: two 503s then success must succeed via retry."""
        prov = _provider()
        calls = {"n": 0}

        def flaky(req, timeout=None):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise _http_error(req, 503, b"server busy")
            return _FakeResponse(
                b'{"choices": [{"message": {"content": "ok"}}], '
                b'"usage": {"prompt_tokens": 1, "completion_tokens": 1}}'
            )

        # Go through the REAL code path: chat() wraps _make_request in the
        # RetryPolicy — calling _make_request directly would bypass it.
        with patch("urllib.request.urlopen", side_effect=flaky):
            result = asyncio.run(prov.chat([{"role": "user", "content": "hi"}]))
        assert calls["n"] == 3
        assert result == "ok"

    def test_permanent_400_fails_fast(self) -> None:
        """A non-transient status must NOT be retried."""
        prov = _provider()
        calls = {"n": 0}

        def bad(req, timeout=None):
            calls["n"] += 1
            raise _http_error(req, 400, b'{"error": "invalid parameter: foo"}')

        with patch("urllib.request.urlopen", side_effect=bad), patch(
            "agent_core.llm.lmstudio.load_model"
        ) as lm:
            result = asyncio.run(prov.chat([{"role": "user", "content": "hi"}]))
        assert calls["n"] == 1  # no retry on permanent errors
        assert result.startswith("[Error:")
        assert "invalid parameter" in result
        lm.assert_not_called()

    def test_transient_exhaustion_reports_error_string(self) -> None:
        """All attempts failing still degrades to an [Error: ...] string —
        providers never raise into the tool loop."""
        prov = _provider()
        calls = {"n": 0}

        def always_503(req, timeout=None):
            calls["n"] += 1
            raise _http_error(req, 503, b"server busy")

        with patch("urllib.request.urlopen", side_effect=always_503):
            result = asyncio.run(prov.chat([{"role": "user", "content": "hi"}]))
        assert calls["n"] == prov.retry_policy.max_retries
        assert result.startswith("[Error:")

    def test_retryable_errors_include_transient_http_error(self) -> None:
        from agent_core.llm.retry import RetryPolicy

        policy = RetryPolicy()
        assert any(
            issubclass(TransientHTTPError, t) for t in policy.retryable_errors
        ), "default policy must classify TransientHTTPError as retryable"


def _exhausted_response() -> bytes:
    return json.dumps({
        "choices": [{
            "message": {"content": "", "reasoning_content": "r" * 2000},
            "finish_reason": "length",
        }]
    }).encode()


class TestThinkingBudgetRecovery:
    """Regression: a reasoning model that burns its budget must be retried with
    thinking disabled, not surfaced as a task-ending [Error: ...]."""

    def test_retries_with_thinking_disabled(self) -> None:
        prov = _provider()
        payloads: list = []

        def fake(req, timeout=None):
            payloads.append(json.loads(req.data.decode()))
            if len(payloads) == 1:
                return _FakeResponse(_exhausted_response())
            return _FakeResponse(
                b'{"choices": [{"message": {"content": "ok"}}]}'
            )

        with patch("urllib.request.urlopen", side_effect=fake):
            out = asyncio.run(prov.chat(
                [{"role": "user", "content": "hi"}], max_tokens=200
            ))
        assert out == "ok"
        assert len(payloads) == 2
        # The retry carries the reasoning-off knob (fallback path for a model
        # with no catalog entry sends reasoning:"off" + enable_thinking:false)...
        assert payloads[1].get("reasoning") == "off"
        # ...and a larger output budget, because some reasoning models ignore
        # the reasoning-off knob and only need more room to finish thinking.
        assert payloads[0]["max_tokens"] == 200
        assert payloads[1]["max_tokens"] >= 4096

    def test_no_retry_when_thinking_already_disabled(self) -> None:
        prov = _provider()
        calls = {"n": 0}

        def fake(req, timeout=None):
            calls["n"] += 1
            return _FakeResponse(_exhausted_response())

        with patch("urllib.request.urlopen", side_effect=fake):
            out = asyncio.run(prov.chat(
                [{"role": "user", "content": "hi"}], disable_thinking=True
            ))
        assert calls["n"] == 1
        assert out.startswith("[Error:")
        assert "reasoning bytes" in out

    def test_retry_failure_still_returns_error(self) -> None:
        prov = _provider()
        calls = {"n": 0}

        def fake(req, timeout=None):
            calls["n"] += 1
            return _FakeResponse(_exhausted_response())

        with patch("urllib.request.urlopen", side_effect=fake):
            out = asyncio.run(prov.chat([{"role": "user", "content": "hi"}]))
        assert calls["n"] == 2  # one attempt + one disable-thinking retry
        assert out.startswith("[Error:")


#: The reported live failure: LM Studio wrapped an engine crash in HTTP 400.
#: The inner error is a JSON string, so its quotes arrive backslash-escaped.
_ENGINE_ERROR_BODY = (
    b'{"error":"Engine protocol predict stream returned an error: '
    b'{\\"code\\":500,\\"message\\":\\"The model produced output that does not '
    b'match the expected peg-native format\\",\\"type\\":\\"server_error\\"}"}'
)


class TestEngineServerErrors:
    def test_400_server_error_body_raises_typed_error(self) -> None:
        prov = _provider()

        def boom(req, timeout=None):
            raise _http_error(req, 400, _ENGINE_ERROR_BODY)

        with patch("urllib.request.urlopen", side_effect=boom), patch(
            "agent_core.llm.lmstudio.load_model", return_value=(False, "no")
        ):
            with pytest.raises(TransientHTTPError) as excinfo:
                prov._make_request({"model": "x"})
        assert excinfo.value.status == 400

    def test_engine_error_retried_until_success(self) -> None:
        """A peg-native engine error is re-sampled, not surfaced as final."""
        prov = _provider()
        calls = {"n": 0}

        def flaky(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _http_error(req, 400, _ENGINE_ERROR_BODY)
            return _FakeResponse(b'{"choices": [{"message": {"content": "ok"}}]}')

        with patch("urllib.request.urlopen", side_effect=flaky):
            result = asyncio.run(prov.chat([{"role": "user", "content": "hi"}]))
        assert calls["n"] == 2
        assert result == "ok"

    def test_predict_fetch_failure_is_retried(self) -> None:
        prov = _provider()
        calls = {"n": 0}
        body = b'{"error":"Engine protocol predict request failed: fetch failed"}'

        def flaky(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _http_error(req, 400, body)
            return _FakeResponse(b'{"choices": [{"message": {"content": "ok"}}]}')

        with patch("urllib.request.urlopen", side_effect=flaky):
            result = asyncio.run(prov.chat([{"role": "user", "content": "hi"}]))
        assert calls["n"] == 2
        assert result == "ok"
