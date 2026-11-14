"""LLM metrics contract tests (plan ARCH items 14, 16, 17): ResponseMetrics,
MetricsTracker token/cost accounting, and provider-side capture."""

import json
import types

import pytest

from agent_core.llm.llm_types import ProfileType, TaskType
from agent_core.llm.metrics_tracker import MetricsTracker
from agent_core.llm import provider as pmod
from agent_core.llm.provider import (
    LAST_METRICS_ATTR,
    ResponseMetrics,
    get_last_metrics,
)


class TestResponseMetrics:
    def test_total_tokens(self):
        m = ResponseMetrics(prompt_tokens=10, completion_tokens=20)
        assert m.total_tokens == 30

    def test_defaults_zero(self):
        m = ResponseMetrics()
        assert m.total_tokens == 0
        assert m.latency_ms == 0.0
        assert m.cost == 0.0

    def test_frozen(self):
        m = ResponseMetrics(prompt_tokens=1)
        with pytest.raises(Exception):
            m.prompt_tokens = 2  # type: ignore[misc]

    def test_get_last_metrics_helpers(self):
        provider = type("P", (), {LAST_METRICS_ATTR: None})()
        assert get_last_metrics(provider) is None
        assert get_last_metrics(None) is None
        m = ResponseMetrics(completion_tokens=5)
        provider = type("P", (), {LAST_METRICS_ATTR: m})()
        assert get_last_metrics(provider) is m


class TestDecodeThroughput:
    """TTFT-based decode rate: the metric the NPU/iGPU gate compares.

    ``decode_tps`` must divide by the *post-TTFT* window.  Dividing by the
    whole latency would fold prefill into the rate, so a fast decoder on a
    long prompt would look slow and any A/B between two engines would be
    measuring prompt processing as much as generation.
    """

    def test_excludes_prefill_window(self):
        # 50 tokens generated; 200ms to first token, 1200ms total -> 1.0s of
        # decode, so 50 tok/s (not 50/1.2 = 41.7).
        m = ResponseMetrics(completion_tokens=50, ttft_ms=200.0, latency_ms=1200.0)
        assert m.decode_tps == pytest.approx(50.0)

    def test_unmeasured_returns_zero(self):
        # No first-token boundary (non-streaming transport): not measured.
        m = ResponseMetrics(completion_tokens=50, latency_ms=1200.0)
        assert m.decode_tps == 0.0
        # No completion tokens reported.
        assert ResponseMetrics(ttft_ms=200.0, latency_ms=1200.0).decode_tps == 0.0

    def test_degenerate_timings_return_zero(self):
        # TTFT == latency: the stream ended as the first token arrived.
        m = ResponseMetrics(completion_tokens=5, ttft_ms=900.0, latency_ms=900.0)
        assert m.decode_tps == 0.0
        # Inconsistent timings must not produce a negative rate.
        m = ResponseMetrics(completion_tokens=5, ttft_ms=900.0, latency_ms=100.0)
        assert m.decode_tps == 0.0


class TestStreamClock:
    """The shared TTFT/elapsed helper every streaming transport uses."""

    @staticmethod
    def _fake_monotonic(monkeypatch: pytest.MonkeyPatch):
        """Install a settable fake ``time.monotonic``; returns an object with
        a ``now`` attribute the test moves explicitly.

        A generator-backed fake would be corrupted by any unrelated
        ``monotonic()`` call (pytest itself makes them), so the value is set
        rather than consumed.
        """
        fake = types.SimpleNamespace(now=0.0)
        monkeypatch.setattr(pmod.time, "monotonic", lambda: fake.now)
        return fake

    def test_measures_ttft_and_elapsed(self, monkeypatch):
        # ``time.monotonic()`` is in SECONDS; StreamClock reports ms.
        fake = self._fake_monotonic(monkeypatch)
        fake.now = 1.0
        clock = pmod.StreamClock()
        clock.start()
        assert clock.has_first_token is False
        assert clock.ttft_ms == 0.0  # no token yet
        fake.now = 1.25
        clock.mark_first_token()
        assert clock.has_first_token is True
        assert clock.ttft_ms == pytest.approx(250.0)
        fake.now = 1.40
        assert clock.elapsed_ms == pytest.approx(400.0)

    def test_mark_first_token_only_records_once(self, monkeypatch):
        # A provider calls this on every delta; later calls must not move TTFT.
        fake = self._fake_monotonic(monkeypatch)
        fake.now = 1.0
        clock = pmod.StreamClock()
        clock.start()
        fake.now = 1.2
        clock.mark_first_token()
        fake.now = 1.3
        clock.mark_first_token()
        fake.now = 1.5
        clock.mark_first_token()
        assert clock.ttft_ms == pytest.approx(200.0)

    def test_unstarted_clock_is_zero(self):
        clock = pmod.StreamClock()
        assert clock.ttft_ms == 0.0
        assert clock.elapsed_ms == 0.0
        assert clock.has_first_token is False

    def test_metrics_snapshot_carries_ttft(self, monkeypatch):
        fake = self._fake_monotonic(monkeypatch)
        fake.now = 0.0
        clock = pmod.StreamClock()
        clock.start()
        fake.now = 0.3
        clock.mark_first_token()
        fake.now = 2.3
        m = clock.metrics(prompt_tokens=10, completion_tokens=100, cost=0.5)
        assert isinstance(m, ResponseMetrics)
        assert m.ttft_ms == pytest.approx(300.0)
        assert m.latency_ms == pytest.approx(2300.0)
        assert m.total_tokens == 110
        assert m.cost == 0.5
        # 100 tokens over the 2.0s post-TTFT window.
        assert m.decode_tps == pytest.approx(50.0)


class TestEstimateTokensFromText:
    """Fallback for servers that ignore ``stream_options.include_usage``."""

    def test_empty_is_zero(self):
        assert pmod.estimate_tokens_from_text("") == 0

    def test_four_chars_per_token(self):
        assert pmod.estimate_tokens_from_text("abcdefgh") == 2

    def test_never_returns_zero_for_short_text(self):
        # A one-character reply is still one token, not a 0-token division.
        assert pmod.estimate_tokens_from_text("a") == 1


class TestMetricsTrackerAccounting:
    def test_record_turn_accumulates(self):
        tracker = MetricsTracker()
        tracker.record_turn(
            TaskType.IMPLEMENT, ProfileType.PRECISE, tokens=100, cost=0.01
        )
        tracker.record_turn(
            TaskType.IMPLEMENT, ProfileType.PRECISE, tokens=200, cost=0.03
        )
        metrics = tracker.get_metrics(TaskType.IMPLEMENT, ProfileType.PRECISE)
        assert metrics["total_tokens"] == 300
        assert metrics["avg_tokens"] == 150
        assert metrics["total_cost"] == pytest.approx(0.04)
        assert metrics["avg_cost"] == pytest.approx(0.02)
        assert metrics["success_count"] == 2

    def test_record_turn_accepts_response_metrics(self):
        tracker = MetricsTracker()
        tracker.record_turn(
            TaskType.IMPLEMENT,
            ProfileType.PRECISE,
            metrics=ResponseMetrics(
                prompt_tokens=10, completion_tokens=20, latency_ms=500
            ),
        )
        metrics = tracker.get_metrics(TaskType.IMPLEMENT, ProfileType.PRECISE)
        assert metrics["total_tokens"] == 30
        assert metrics["avg_latency"] == pytest.approx(0.5)

    def test_no_tokens_records_zero_averages(self):
        tracker = MetricsTracker()
        tracker.record_success(TaskType.IMPLEMENT, ProfileType.PRECISE, 0.3)
        metrics = tracker.get_metrics(TaskType.IMPLEMENT, ProfileType.PRECISE)
        assert metrics["total_tokens"] == 0
        assert metrics["avg_tokens"] == 0
        assert metrics["total_cost"] == 0.0

    def test_reset_clears_tokens(self):
        tracker = MetricsTracker()
        tracker.record_turn(TaskType.IMPLEMENT, ProfileType.PRECISE, tokens=50)
        tracker.reset()
        metrics = tracker.get_metrics(TaskType.IMPLEMENT, ProfileType.PRECISE)
        assert metrics["total_tokens"] == 0


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


class TestProviderCapture:
    def test_lmstudio_captures_usage(self):
        import asyncio
        from unittest.mock import patch
        from agent_core.llm.lmstudio import LMStudioProvider
        from agent_core.llm.provider import ResponseMetrics

        prov = LMStudioProvider(model_name="laguna-s-2.1", api_key="fake")

        def fake_urlopen(req, timeout=None):
            return _FakeHttp(
                {
                    "choices": [
                        {"message": {"content": "hi"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7},
                }
            )

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            out = asyncio.run(prov.chat([{"role": "user", "content": "x"}]))
        assert out == "hi"
        assert isinstance(prov.last_response_metrics, ResponseMetrics)
        assert prov.last_response_metrics.total_tokens == 18
        assert prov.last_response_metrics.latency_ms >= 0.0

    def test_opencode_captures_usage(self):
        import asyncio
        from unittest.mock import patch
        from agent_core.llm.opencode_provider import OpencodeProvider
        from agent_core.llm.provider import ResponseMetrics

        prov = OpencodeProvider(
            "opencode-go/glm-5.2", api_key="sk-test", read_store=False
        )

        def fake_urlopen(req, timeout=None):
            return _FakeHttp(
                {
                    "choices": [
                        {"message": {"content": "ok"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 2},
                }
            )

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            out = asyncio.run(prov.chat([{"role": "user", "content": "q"}]))
        assert out == "ok"
        assert isinstance(prov.last_response_metrics, ResponseMetrics)
        assert prov.last_response_metrics.total_tokens == 6
