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


#: An engine-side failure LM Studio wraps in HTTP 400.  The inner error is a
#: JSON string, so its quotes arrive backslash-escaped.  Transient: re-sampling
#: can succeed.
_ENGINE_ERROR_BODY = (
    b'{"error":"Engine protocol predict stream returned an error: '
    b'{\\"code\\":500,\\"message\\":\\"internal engine failure\\",'
    b'\\"type\\":\\"server_error\\"}"}'
)

#: The PEG tool-grammar failure (reported live with
#: ``llama-4-scout-17b-16e-instruct``).  DETERMINISTIC for a tool payload:
#: retrying verbatim is wasted, so the provider drops tools and answers as text.
_PEG_ERROR_BODY = (
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
        """A transient engine server_error is re-sampled, not surfaced."""
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


class TestToolGrammarRecovery:
    """Regression (2026-09-27, llama-4-scout-17b-16e-instruct): LM Studio's PEG
    tool-call grammar rejects the model's output; the provider must drop tools
    and answer as text instead of failing the turn.  The failure is
    deterministic, so it must NOT be retried verbatim as a transient error."""

    def test_peg_body_is_not_transient(self) -> None:
        from agent_core.llm.lmstudio import (
            _engine_server_error,
            _is_tool_grammar_error,
        )

        detail = _PEG_ERROR_BODY.decode()
        assert _is_tool_grammar_error(detail)
        assert not _engine_server_error(detail)

    def test_peg_error_retries_without_tools(self) -> None:
        prov = _provider()
        payloads: list = []

        def fake(req, timeout=None):
            payloads.append(json.loads(req.data.decode()))
            if len(payloads) == 1:
                raise _http_error(req, 400, _PEG_ERROR_BODY)
            return _FakeResponse(b'{"choices": [{"message": {"content": "ok"}}]}')

        tools = [{"type": "function", "function": {"name": "x", "parameters": {}}}]
        with patch("urllib.request.urlopen", side_effect=fake):
            out = asyncio.run(prov.chat(
                [{"role": "user", "content": "hi"}], tools=tools
            ))
        assert out == "ok"
        assert len(payloads) == 2  # no verbatim retries, just the no-tools retry
        assert "tools" in payloads[0]
        assert "tools" not in payloads[1]

    def test_peg_error_without_tools_surfaces(self) -> None:
        prov = _provider()
        calls = {"n": 0}

        def fake(req, timeout=None):
            calls["n"] += 1
            raise _http_error(req, 400, _PEG_ERROR_BODY)

        with patch("urllib.request.urlopen", side_effect=fake):
            out = asyncio.run(prov.chat([{"role": "user", "content": "hi"}]))
        assert calls["n"] == 1  # nothing to drop -> no retry
        assert out.startswith("[Error:")


#: The reported live failure (llama-4-scout-17b-16e-instruct, ~63 GB GGUF):
#: the LM Studio engine cannot allocate the model/context.
_OOM_BODY = (
    b'{"error":"Engine protocol predict stream returned an error: '
    b'{\\"code\\":500,\\"message\\":\\"got exception: bad allocation\\",'
    b'\\"type\\":\\"server_error\\"}"}'
)


class TestEngineOomRecovery:
    """Regression (2026-09-27): an out-of-memory engine error is deterministic,
    so it must fail fast (with a clear message) and let the failover chain
    move on — not be retried three times verbatim."""

    def test_oom_body_is_not_transient(self) -> None:
        from agent_core.llm.lmstudio import (
            _engine_server_error,
            _is_engine_oom_error,
        )

        detail = _OOM_BODY.decode()
        assert _is_engine_oom_error(detail)
        assert not _engine_server_error(detail)

    def test_oom_raises_runtime_error_with_guidance(self) -> None:
        prov = _provider()

        def boom(req, timeout=None):
            raise _http_error(req, 400, _OOM_BODY)

        with patch("urllib.request.urlopen", side_effect=boom):
            with pytest.raises(RuntimeError) as excinfo:
                prov._make_request({"model": "x"})
        assert not isinstance(excinfo.value, TransientHTTPError)
        assert "out of memory" in str(excinfo.value)

    def test_oom_fails_fast_and_is_failover_worthy(self) -> None:
        from agent_core.llm.provider import is_connection_failure

        prov = _provider()
        calls = {"n": 0}

        def fake(req, timeout=None):
            calls["n"] += 1
            raise _http_error(req, 400, _OOM_BODY)

        with patch("urllib.request.urlopen", side_effect=fake):
            out = asyncio.run(prov.chat([{"role": "user", "content": "hi"}]))
        assert calls["n"] == 1  # no transient retries for a deterministic OOM
        assert out.startswith("[Error:")
        assert "out of memory" in out
        assert is_connection_failure(out)  # the chain should fail over


def _channel_error_body() -> bytes:
    return (
        b'{"error":"Engine protocol predict stream returned an error: '
        b'{\\"code\\":500,\\"message\\":\\"Error: Channel Error\\",'
        b'\\"type\\":\\"server_error\\"}"}'
    )


class TestToolPathDropTools:
    """When a model's tool path crashes the LM Studio engine, the provider may
    drop tools and answer as text (LM Studio's own chat sends no tools).

    An out-of-memory is NOT a tool-path failure: dropping tools cannot make an
    oversized model fit, so it must fail fast instead of degrading a
    tool-calling caller into plain chat that cannot act."""

    def test_tool_path_error_markers(self) -> None:
        from agent_core.llm.lmstudio import _is_tool_path_engine_error

        assert _is_tool_path_engine_error(f"[Error: {_PEG_ERROR_BODY.decode()}]")
        assert _is_tool_path_engine_error(f"[Error: {_channel_error_body().decode()}]")
        assert _is_tool_path_engine_error(
            '[Error: Engine protocol predict request failed: fetch failed]'
        )
        assert _is_tool_path_engine_error(
            '[Error: HTTP Error 400: {"error":"terminated"}]'
        )
        assert not _is_tool_path_engine_error("[Error: HTTP Error 401: Unauthorized]")
        assert not _is_tool_path_engine_error("normal response")

    def test_oom_is_not_a_tool_path_error(self) -> None:
        """Regression: the OOM message quotes "bad allocation", so it used to
        match the tool-path classifier and silently dropped tools."""
        from agent_core.llm.lmstudio import (
            _is_engine_oom_error,
            _is_tool_path_engine_error,
        )

        text = f"[Error: {_OOM_BODY.decode()}]"
        assert _is_engine_oom_error(text)
        assert not _is_tool_path_engine_error(text)

    def test_oom_with_tools_fails_fast_instead_of_dropping_tools(self) -> None:
        """Regression (subagent hang): a tool-using caller must never be
        silently answered without tools when the engine is out of memory — the
        caller cannot tell the difference between a real answer and a degraded
        one, so the task looks done while nothing happened."""
        from agent_core.llm.provider import is_connection_failure

        prov = _provider()
        payloads: list = []

        def fake(req, timeout=None):
            payloads.append(json.loads(req.data.decode()))
            raise _http_error(req, 400, _OOM_BODY)

        tools = [{"type": "function", "function": {"name": "x", "parameters": {}}}]
        with patch("urllib.request.urlopen", side_effect=fake):
            out = asyncio.run(prov.chat(
                [{"role": "user", "content": "hi"}], tools=tools
            ))
        assert len(payloads) == 1, "a deterministic OOM must not be retried"
        assert out.startswith("[Error:")
        assert "out of memory" in out
        assert is_connection_failure(out), "the failover chain must move on"

    def test_tools_disabled_on_subsequent_calls(self) -> None:
        """A GENUINE tool-path failure is remembered so the doomed attempt is
        not repeated; the caller keeps getting the degraded text answer that
        makes sense for plain chat."""
        prov = _provider()
        payloads: list = []

        def fake(req, timeout=None):
            payloads.append(json.loads(req.data.decode()))
            if "tools" in payloads[-1]:
                # PEG is deterministic and non-transient: it fails fast, so the
                # "already known unsupported" memo can be observed in one turn.
                raise _http_error(req, 400, _PEG_ERROR_BODY)
            return _FakeResponse(b'{"choices": [{"message": {"content": "ok"}}]}')

        tools = [{"type": "function", "function": {"name": "x", "parameters": {}}}]
        with patch("urllib.request.urlopen", side_effect=fake):
            out1 = asyncio.run(prov.chat(
                [{"role": "user", "content": "a"}], tools=tools
            ))
            first_call_len = len(payloads)
            out2 = asyncio.run(prov.chat(
                [{"role": "user", "content": "b"}], tools=tools
            ))
        assert out1 == "ok" and out2 == "ok"
        # The 2nd chat must skip tools entirely (no doomed attempt).
        assert first_call_len == 2
        assert "tools" not in payloads[first_call_len]
