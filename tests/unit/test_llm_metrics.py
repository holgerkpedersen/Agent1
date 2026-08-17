"""LLM metrics contract tests (plan ARCH items 14, 16, 17): ResponseMetrics,
MetricsTracker token/cost accounting, and provider-side capture."""
import pytest

from agent_core.llm.llm_types import ProfileType, TaskType
from agent_core.llm.metrics_tracker import MetricsTracker
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


class TestMetricsTrackerAccounting:
    def test_record_turn_accumulates(self):
        tracker = MetricsTracker()
        tracker.record_turn(TaskType.IMPLEMENT, ProfileType.PRECISE, tokens=100, cost=0.01)
        tracker.record_turn(TaskType.IMPLEMENT, ProfileType.PRECISE, tokens=200, cost=0.03)
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
            metrics=ResponseMetrics(prompt_tokens=10, completion_tokens=20, latency_ms=500),
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


class TestProviderCapture:
    def test_lmstudio_captures_usage(self):
        import asyncio
        import json
        from unittest.mock import patch
        from agent_core.llm.lmstudio import LMStudioProvider
        from agent_core.llm.provider import ResponseMetrics
        from tests.test_opencode_provider import _FakeHttp

        prov = LMStudioProvider(model_name="laguna-s-2.1", api_key="fake")

        def fake_urlopen(req, timeout=None):
            return _FakeHttp({
                "choices": [{"message": {"content": "hi"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            })

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
        from tests.test_opencode_provider import _FakeHttp

        prov = OpencodeProvider("opencode-go/glm-5.2", api_key="sk-test", read_store=False)

        def fake_urlopen(req, timeout=None):
            return _FakeHttp({
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            })

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            out = asyncio.run(prov.chat([{"role": "user", "content": "q"}]))
        assert out == "ok"
        assert isinstance(prov.last_response_metrics, ResponseMetrics)
        assert prov.last_response_metrics.total_tokens == 6
