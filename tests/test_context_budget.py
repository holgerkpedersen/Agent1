"""Tests for per-provider context budgeting (NPU small-window safety)."""
from __future__ import annotations

from agent_core.llm.context_budget import (
    CHARS_PER_TOKEN,
    TRIM_NOTE,
    char_budget_for,
    estimate_tokens,
    provider_context_limit,
    trim_messages_to_context,
)


class _P:
    def __init__(self, ctx):
        self.context_limit = ctx


class TestEstimates:
    def test_estimate_scales_with_content(self):
        small = estimate_tokens([{"role": "user", "content": "hi"}])
        big = estimate_tokens([{"role": "user", "content": "x" * 3500}])
        assert big > small
        assert big >= int(3500 / CHARS_PER_TOKEN)

    def test_provider_context_limit(self):
        assert provider_context_limit(_P(65536)) == 65536
        assert provider_context_limit(_P(None)) is None
        assert provider_context_limit(_P(0)) is None
        assert provider_context_limit(object()) is None

    def test_char_budget(self):
        budget = char_budget_for(_P(1000), reserve_tokens=0, safety=1.0)
        assert budget == int(1000 * CHARS_PER_TOKEN)
        assert char_budget_for(_P(None)) is None


class TestTrimming:
    def test_noop_when_it_fits(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert trim_messages_to_context(msgs, limit_tokens=100) == msgs

    def test_noop_when_limit_unknown(self):
        msgs = [{"role": "user", "content": "x" * 10000}]
        assert trim_messages_to_context(msgs, provider=_P(None)) == msgs

    def test_drops_oldest_and_keeps_system_and_newest(self):
        msgs = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "a" * 400},
            {"role": "assistant", "content": "b" * 400},
            {"role": "user", "content": "c" * 400},
        ]
        out = trim_messages_to_context(msgs, limit_tokens=200, reserve_tokens=0)
        assert out[0] == msgs[0]  # system kept
        assert any(m.get("content") == TRIM_NOTE for m in out)
        assert out[-1]["content"] == "c" * 400  # newest kept
        assert not any(m["content"] == "a" * 400 for m in out)

    def test_leading_orphan_tool_is_skipped(self):
        msgs = [
            {"role": "system", "content": "SYS"},
            {"role": "tool", "content": "t" * 400, "tool_call_id": "1"},
        ]
        out = trim_messages_to_context(msgs, limit_tokens=10, reserve_tokens=0)
        assert out[0] == msgs[0]
        assert all(m.get("role") != "tool" for m in out)
