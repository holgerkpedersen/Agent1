"""Mocked tests for the cheapest-per-token failover strategy & cost metrics.

No real network/endpoints are touched: providers are stubbed and the pricing
table is monkeypatched, so the suite is deterministic and offline.  Model
names come from ``_helpers`` (``_default_zen_free_model`` / ``_default_llm``)
which honor ``AGENT_ZEN_FREE_MODEL`` / ``AGENT_OPENCODE_MODEL`` when set.
"""
from __future__ import annotations

import asyncio

import pytest

from agent_core.llm.pricing import cost_per_token, estimate_cost
from agent_core.llm.provider import FailoverProvider
from tests._helpers import _default_llm, _default_zen_free_model


class _Stub:
    """Minimal LLMProvider stand-in for failover ordering tests."""

    def __init__(self, name: str, model: str, fail: bool = False) -> None:
        self.name = name
        self.model_name = model
        self.temperature = 0.7
        self.max_tokens = 100
        self._profile_name = None
        self.last_response_metrics = None
        self.call_count = 0
        self._fail = fail

    async def chat(self, messages, **kwargs):
        self.call_count += 1
        if self._fail:
            return (
                f"[Error: opencode-zen free model {self.model_name} is "
                "currently unavailable on the backend. Try another free model "
                f"with 'model opencode-zen/<id>-free' (e.g. {self.model_name}). "
                f"Checked: {self.model_name}.]"
            )
        return f"ok:{self.name}"

    async def chat_stream(self, messages):
        return "streamed"

    async def analyze_code(self, code):
        return "analyzed"

    def apply_profile(self, *a):
        pass


def _free_model() -> str:
    return _default_zen_free_model()


def _paid_model() -> str:
    return _default_llm()


def test_cost_per_token_free_tiers_zero() -> None:
    """Free/zen and local providers are always cost 0.0 (cheapest wins)."""
    assert cost_per_token(_free_model(), "opencode") == 0.0
    assert cost_per_token("opencode-go/deepseek-v4-flash:free", "opencode") == 0.0
    assert cost_per_token("anything", "lmstudio") == 0.0
    assert cost_per_token("anything", "llama") == 0.0


def test_cost_per_token_paid_nonzero(monkeypatch) -> None:
    """A known paid model yields a positive per-token cost."""
    monkeypatch.setattr(
        "agent_core.llm.pricing.BASELINE_PRICES",
        {_paid_model(): {"prompt_per_1m": 1.0, "completion_per_1m": 3.0, "verified": False, "source": "test"}},
    )
    # Blended per-1M = (1 + 3)/2 = 2.0 -> per token = 2e-6.
    assert cost_per_token(_paid_model(), "opencode") == pytest.approx(2.0 / 1_000_000.0)


def test_ordered_strategy_preserves_chain() -> None:
    """Default 'ordered' strategy keeps the configured provider order."""
    zen = _Stub("zen", _free_model())
    go = _Stub("go", _paid_model())
    fp = FailoverProvider([zen, go], model_name=_free_model(), strategy="ordered")
    assert [type(p).__name__ for p in fp.providers] == ["_Stub", "_Stub"]
    assert fp.providers[0] is zen
    assert fp.providers[1] is go


def test_cheapest_strategy_orders_by_cost(monkeypatch) -> None:
    """'cheapest' re-orders the chain by ascending per-token cost."""
    paid = _paid_model()
    monkeypatch.setattr(
        "agent_core.llm.pricing.BASELINE_PRICES",
        {paid: {"prompt_per_1m": 1.0, "completion_per_1m": 3.0, "verified": False, "source": "test"}},
    )
    # Order the inputs as paid-then-free; cheapest must try free first.
    go = _Stub("go", paid)
    zen = _Stub("zen", _free_model())
    fp = FailoverProvider([go, zen], model_name=_free_model(), strategy="cheapest")
    assert fp.providers[0] is zen
    assert fp.providers[1] is go


def test_cheapest_strategy_prefers_free_over_paid(monkeypatch) -> None:
    """With a paid go and free zen, cheapest tries zen (cost 0) first."""
    paid = _paid_model()
    monkeypatch.setattr(
        "agent_core.llm.pricing.BASELINE_PRICES",
        {paid: {"prompt_per_1m": 2.0, "completion_per_1m": 4.0, "verified": False, "source": "test"}},
    )
    go = _Stub("go", paid, fail=False)
    zen = _Stub("zen", _free_model(), fail=False)
    fp = FailoverProvider([go, zen], model_name=_free_model(), strategy="cheapest")
    out = asyncio.run(fp.chat([{"role": "user", "content": "hi"}]))
    assert out == f"ok:{zen.name}"
    assert zen.call_count == 1
    assert go.call_count == 0


def test_cheapest_falls_over_on_failure(monkeypatch) -> None:
    """If the cheapest provider fails, the next-cheapest answers."""
    paid = _paid_model()
    monkeypatch.setattr(
        "agent_core.llm.pricing.BASELINE_PRICES",
        {paid: {"prompt_per_1m": 2.0, "completion_per_1m": 4.0, "verified": False, "source": "test"}},
    )
    # zen is cheapest (cost 0) but failing; go is next and must answer.
    zen = _Stub("zen", _free_model(), fail=True)
    go = _Stub("go", paid, fail=False)
    fp = FailoverProvider([go, zen], model_name=_free_model(), strategy="cheapest")
    out = asyncio.run(fp.chat([{"role": "user", "content": "hi"}]))
    assert out == f"ok:{go.name}"
    assert zen.call_count == 1
    assert go.call_count == 1


def test_cheapest_strategy_invalid_value() -> None:
    with pytest.raises(ValueError):
        FailoverProvider([_Stub("a", _free_model())], model_name=_free_model(), strategy="bogus")


def test_estimate_cost_free_zero() -> None:
    assert estimate_cost(1000, 2000, _free_model(), "opencode") == 0.0
    assert estimate_cost(1000, 2000, "anything", "lmstudio") == 0.0


def test_estimate_cost_paid_nonzero(monkeypatch) -> None:
    paid = _paid_model()
    monkeypatch.setattr(
        "agent_core.llm.pricing.BASELINE_PRICES",
        {paid: {"prompt_per_1m": 1.0, "completion_per_1m": 3.0, "verified": False, "source": "test"}},
    )
    # (1000*1 + 2000*3)/1e6 = (1000 + 6000)/1e6 = 7e-3.
    assert estimate_cost(1000, 2000, paid, "opencode") == pytest.approx(7.0 / 1_000.0)
