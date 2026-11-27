"""Regression tests: opencode-zen free-tier quota exhaustion must fail FAST and
hand over to the next provider instead of burning retries and answering with an
error string.

Live evidence (2026-11-27): every ``-free`` model returns HTTP 429 with body
``{"type":"error","error":{"type":"FreeUsageLimitError","message":"Free usage
limit reached. Free models work in OpenCode (https://opencode.ai); subscribe to
OpenCode Go ..."}}``.  The quota is ACCOUNT-wide, so rotating between free models
cannot succeed.

Before the fix three classifiers disagreed:
  * ``_TRANSIENT_HTTP_STATUSES`` retried it (2+4+8s = 14 s wasted per turn),
  * ``_BACKEND_DOWN_MARKERS`` did not treat it as down,
  * ``_CONNECTION_FAILURE_RE`` did not treat it as failover-worthy,
so the turn stalled and then returned ``[Error: ...]`` as the answer.
"""

import io
import urllib.error

from agent_core.llm.opencode_provider import (
    _is_quota_exhausted,
    _is_transient_http_error,
)
from agent_core.llm.provider import is_connection_failure

#: The exact gateway body observed live for an exhausted free tier.
LIVE_QUOTA_BODY = (
    '{"type":"error","error":{"type":"FreeUsageLimitError","message":'
    '"Free usage limit reached. Free models work in OpenCode '
    '(https://opencode.ai); subscribe to OpenCode Go to keep going: '
    'https://opencode.ai/go"}}'
)

#: How OpencodeProvider.chat() surfaces that 429 (opencode_provider.py:705).
LIVE_QUOTA_ERROR_STRING = (
    "[Error: opencode API request failed: HTTP Error 429: " + LIVE_QUOTA_BODY + "]"
)


def _http_error(code: int, body: str) -> urllib.error.HTTPError:
    """Build a real HTTPError carrying *body* (``fp`` mirrors a live socket)."""
    return urllib.error.HTTPError(
        url="https://opencode.ai/zen/v1/chat/completions",
        code=code,
        msg="Too Many Requests",
        hdrs=None,
        fp=io.BytesIO(body.encode("utf-8")),
    )


class TestQuotaBodyDetection:
    """Task 1 — one predicate owns the quota classification."""

    def test_live_quota_body_is_detected(self):
        assert _is_quota_exhausted(LIVE_QUOTA_BODY) is True

    def test_detection_ignores_json_whitespace(self):
        spaced = '{"type": "error", "error": {"type": "FreeUsageLimitError"}}'
        assert _is_quota_exhausted(spaced) is True

    def test_detection_is_case_insensitive(self):
        assert _is_quota_exhausted("FREE USAGE LIMIT REACHED") is True

    def test_ordinary_error_bodies_are_not_quota(self):
        for body in (
            "",
            "400: Bad Request",
            '{"type":"error","error":{"type":"invalid_request_error"}}',
            '{"type":"error","error":{"type":"server_error"}}',
        ):
            assert _is_quota_exhausted(body) is False, body


class TestTransientClassification:
    """Task 2 — quota is a 429 that must NOT be retried."""

    def test_quota_429_is_not_transient(self):
        assert _is_transient_http_error(_http_error(429, LIVE_QUOTA_BODY)) is False

    def test_bare_429_stays_transient(self):
        # Genuine burst rate limiting still deserves exponential backoff.
        assert _is_transient_http_error(_http_error(429, "")) is True

    def test_5xx_stays_transient(self):
        assert _is_transient_http_error(_http_error(503, "")) is True
        assert _is_transient_http_error(_http_error(500, "")) is True

    def test_server_error_body_still_reclassified(self):
        body = '{"type":"error","error":{"type":"server_error"}}'
        assert _is_transient_http_error(_http_error(403, body)) is True

    def test_client_error_stays_permanent(self):
        body = '{"type":"error","error":{"type":"invalid_request_error"}}'
        assert _is_transient_http_error(_http_error(400, body)) is False


class TestFailoverGate:
    """Task 3 — the quota error must let FailoverProvider try the next provider."""

    def test_quota_error_triggers_failover(self):
        assert is_connection_failure(LIVE_QUOTA_ERROR_STRING) is True

    def test_quota_error_also_counts_as_provider_error(self):
        from agent_core.llm.provider import is_provider_error

        assert is_provider_error(LIVE_QUOTA_ERROR_STRING) is True

    def test_successful_tool_payload_mentioning_error_does_not_failover(self):
        # A model writing error-handling code must not be misread as an outage.
        payload = 'Use `[Error: free usage limit reached]` in your handler.'
        assert is_connection_failure(payload) is False

    def test_auth_error_still_does_not_failover(self):
        auth = "[Error: opencode API request failed: HTTP Error 401: unauthorized]"
        assert is_connection_failure(auth) is False


class TestNoRetryOnQuota:
    """End-to-end: the retry loop aborts immediately on an exhausted quota."""

    def test_quota_error_raises_without_sleeping(self):
        import asyncio

        from agent_core.llm.opencode_provider import OpencodeProvider

        prov = OpencodeProvider("opencode-zen/beam-free", read_store=False)
        calls = {"n": 0}

        def factory():
            calls["n"] += 1
            raise _http_error(429, LIVE_QUOTA_BODY)

        slept: list[float] = []

        async def fake_sleep(delay):
            slept.append(delay)

        async def run():
            import agent_core.llm.opencode_provider as mod

            orig = mod.asyncio.sleep
            mod.asyncio.sleep = fake_sleep
            try:
                await prov._with_retry(factory, label="test")
            except urllib.error.HTTPError:
                return "raised"
            finally:
                mod.asyncio.sleep = orig
            return "returned"

        outcome = asyncio.run(run())
        assert outcome == "raised"
        assert calls["n"] == 1, f"retried {calls['n'] - 1} extra time(s)"
        assert slept == [], f"backed off {slept} on a permanent quota error"
