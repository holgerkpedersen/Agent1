"""Regression: a tool-path engine crash must not be re-sampled.

Observed live (``inkling-2-mini-abliterated-fit``): a tool-using turn sent the
full prompt, the engine crashed in its tool path, and the provider re-sent the
IDENTICAL request three times under ``RetryPolicy`` — ~13 minutes of prefill per
attempt — before failing fast anyway.  The re-sample cannot help: the crash is
caused by the tool payload, so it fails at the same point every time.

The suppression is deliberately NARROW, and these tests pin all three edges:

* a tool-using turn with a tool-path crash sends exactly ONE request;
* a plain engine ``server_error`` on a tool-using turn is still retried (a
  re-sample does recover those — suppressing it would give up early);
* a tool-less turn is never affected at all.

Tests go through ``prov.chat(...)`` — the REAL path, where ``execute_with_retry``
wraps ``_make_request`` — never around it.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
from io import BytesIO
from unittest.mock import patch

import pytest

from agent_core.llm.lmstudio import LMStudioProvider, _is_tool_path_retry_worthless

TOOLS = [{"type": "function", "function": {"name": "x", "parameters": {}}}]

#: The engine crashing on the tool path, as LM Studio reports it.
_CHANNEL_ERROR_BODY = (
    b'{"error":"Engine protocol predict stream returned an error: '
    b'{\\"code\\":500,\\"message\\":\\"Error: Channel Error\\",'
    b'\\"type\\":\\"server_error\\"}"}'
)

#: A recoverable engine blip: carries NO tool-path marker.
_PLAIN_ENGINE_BODY = (
    b'{"error":"Engine protocol predict stream returned an error: '
    b'{\\"code\\":500,\\"message\\":\\"internal engine failure\\",'
    b'\\"type\\":\\"server_error\\"}"}'
)

_OK = b'{"choices": [{"message": {"content": "ok"}}]}'


class _FakeResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(req, code: int, body: bytes) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(req.full_url, code, "Err", {}, BytesIO(body))


def _provider() -> LMStudioProvider:
    prov = LMStudioProvider(model_name="meta/muse-glimmer")
    prov.retry_policy.base_delay = 0.0  # no real sleeping
    return prov


class TestPredicate:
    """The suppression predicate itself — the narrowness lives here."""

    @pytest.mark.parametrize(
        "detail",
        [
            _CHANNEL_ERROR_BODY.decode(),
            '{"error":"terminated"}',
            '{"error":"Engine protocol predict request failed: fetch failed"}',
        ],
    )
    def test_tool_path_crash_is_worthless_to_retry(self, detail: str) -> None:
        assert _is_tool_path_retry_worthless(detail, use_tools=True)

    def test_plain_engine_blip_stays_retryable(self) -> None:
        """A generic predict error is NOT a tool-path crash: keep retrying."""
        assert not _is_tool_path_retry_worthless(
            _PLAIN_ENGINE_BODY.decode(), use_tools=True
        )

    def test_tool_less_call_is_never_affected(self) -> None:
        """Without tools the same body keeps its transient retry."""
        assert not _is_tool_path_retry_worthless(
            _CHANNEL_ERROR_BODY.decode(), use_tools=False
        )


class TestSingleRequestOnToolTurn:
    def test_tool_path_crash_is_not_resampled(self) -> None:
        """THE regression: one request, not three, then fail fast."""
        prov = _provider()
        payloads: list = []

        def boom(req, timeout=None):
            payloads.append(json.loads(req.data.decode()))
            raise _http_error(req, 400, _CHANNEL_ERROR_BODY)

        with patch("urllib.request.urlopen", side_effect=boom):
            out = asyncio.run(
                prov.chat([{"role": "user", "content": "hi"}], tools=TOOLS)
            )
        assert len(payloads) == 1, (
            f"a tool-path crash must not be re-sampled (sent {len(payloads)}x)"
        )
        assert "tools" in payloads[0]
        # Still an honest, failover-worthy failure.
        assert out.startswith("[Error:")
        assert "cannot use tools in this setup" in out

    def test_fail_fast_error_carries_the_engine_detail(self) -> None:
        """The operator must still see WHY it crashed, minus the retry noise."""
        prov = _provider()

        def boom(req, timeout=None):
            raise _http_error(req, 400, _CHANNEL_ERROR_BODY)

        with patch("urllib.request.urlopen", side_effect=boom):
            out = asyncio.run(
                prov.chat([{"role": "user", "content": "hi"}], tools=TOOLS)
            )
        assert "channel error" in out.lower()
        assert "not re-sampled" in out
        assert "retry" not in out.lower()

    def test_plain_engine_blip_on_a_tool_turn_still_retries(self) -> None:
        """Narrowness guard: an ordinary engine blip keeps its transient retry,
        so a tool-using turn that WOULD recover still recovers."""
        prov = _provider()
        calls = {"n": 0}

        def flaky(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _http_error(req, 400, _PLAIN_ENGINE_BODY)
            return _FakeResponse(_OK)

        with patch("urllib.request.urlopen", side_effect=flaky):
            out = asyncio.run(
                prov.chat([{"role": "user", "content": "hi"}], tools=TOOLS)
            )
        assert calls["n"] == 2, "a recoverable engine blip must still be retried"
        assert out == "ok"

    def test_plain_chat_keeps_its_transient_retry(self) -> None:
        """A tool-less call is unaffected: same body, still re-sampled."""
        prov = _provider()
        calls = {"n": 0}

        def flaky(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _http_error(req, 400, _CHANNEL_ERROR_BODY)
            return _FakeResponse(_OK)

        with patch("urllib.request.urlopen", side_effect=flaky):
            out = asyncio.run(prov.chat([{"role": "user", "content": "hi"}]))
        assert calls["n"] == 2
        assert out == "ok"


class TestFailoverStillWorks:
    def test_suppressed_error_is_still_failover_worthy(self) -> None:
        """The point of failing fast: the chain moves to a tool-capable model."""
        from agent_core.llm.provider import is_connection_failure

        prov = _provider()

        def boom(req, timeout=None):
            raise _http_error(req, 400, _CHANNEL_ERROR_BODY)

        with patch("urllib.request.urlopen", side_effect=boom):
            out = asyncio.run(
                prov.chat([{"role": "user", "content": "hi"}], tools=TOOLS)
            )
        assert is_connection_failure(out)
