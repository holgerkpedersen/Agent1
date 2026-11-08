"""Phase 1 tests: provider token facts land in traces and feed the scorer.

Acceptance criteria covered here:
- a tool loop run whose provider reports usage writes a ``token_usage``
  event into the trace (additive; absent when the provider omits usage);
- :func:`harnessfix.evolution_metrics.score_run_tokens` composes the
  Phase 0 token-efficiency contract with the authoritative outcome/latency
  score, is neutral when no token facts exist, and never lets tokens
  credit a failed run;
- :func:`harnessfix.plan_eval.build_plan_evaluation` builds a fact-based
  plan evaluation (planned/completed tasks, verifier/gates/tests flags,
  tokens as ``None`` when unknown) and writes it additively:
  ``.docs/<ts>/evaluation.json`` + an appended ledger line.

The token capture is provider-fact based: the fake provider below sets
``last_response_metrics`` exactly like the real providers do (ARCH item 14).
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_core.llm.provider import ResponseMetrics
from agent_core.llm.tool_loop import ToolLoopRunner
from harnessfix.evolution_metrics import score_run, score_run_tokens
from harnessfix.plan_eval import (
    EVALUATION_FILE,
    build_plan_evaluation,
    write_plan_evaluation,
)
from harnessfix.tracing import KIND_LLM_RESPONSE, TraceWriter


class _UsageProvider:
    """Provider twin: sets ``last_response_metrics`` after every chat call."""

    def __init__(self, metrics: ResponseMetrics | None) -> None:
        self.last_response_metrics = metrics


def _answer_llm(provider: _UsageProvider):
    async def llm_chat_fn(messages, tools):
        text = "final answer " * 20
        if provider.last_response_metrics is not None:
            # Simulate the provider refreshing its per-call metrics on chat.
            provider.last_response_metrics = ResponseMetrics(
                prompt_tokens=100, completion_tokens=40, latency_ms=5.0
            )
        return text, messages + [{"role": "assistant", "content": text}]

    return llm_chat_fn


def _run_loop(tmp_path, provider: _UsageProvider):
    trace = TraceWriter(task_id="p1-tokens", directory=tmp_path)
    loop = ToolLoopRunner(
        max_iterations=3,
        display_mode="quiet",
        trace=trace,
        usage_provider=provider,
    )
    asyncio.run(
        loop.run(
            messages=[{"role": "user", "content": "do it"}],
            llm_chat_fn=_answer_llm(provider),
            execute_tool_fn=None,
            tools=[],
        )
    )
    events = [json.loads(line) for line in trace.path.read_text("utf-8").splitlines()]
    return events


def test_loop_writes_token_usage_event_when_provider_reports_usage(tmp_path):
    provider = _UsageProvider(ResponseMetrics(prompt_tokens=100, completion_tokens=40))
    events = _run_loop(tmp_path, provider)
    usage_events = [e for e in events if e.get("kind") == "token_usage"]
    assert len(usage_events) == 1
    ev = usage_events[0]
    assert ev["prompt_tokens"] == 100
    assert ev["completion_tokens"] == 40
    assert ev["total_tokens"] == 140


def test_loop_emits_no_token_usage_event_when_provider_omits_usage(tmp_path):
    provider = _UsageProvider(None)
    events = _run_loop(tmp_path, provider)
    assert not [e for e in events if e.get("kind") == "token_usage"]


def test_run_tokens_returns_none_when_trace_has_no_token_facts():
    events = [
        {"kind": "loop_end", "outcome": "completed"},
        {"kind": "tool_result", "duration_s": 0.5},
    ]
    assert score_run_tokens(events) == (score_run(events), None)


def test_run_tokens_reads_token_usage_event():
    events = [
        {
            "kind": "token_usage",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
        {"kind": "loop_end", "outcome": "completed"},
    ]
    assert score_run_tokens(events)[1] == 15


def test_efficient_completed_run_scores_at_least_as_well_as_neutral():
    base_events = [
        {"kind": "loop_end", "outcome": "completed"},
        {"kind": "tool_result", "duration_s": 0.1},
    ]
    cheap_events = base_events + [
        {
            "kind": "token_usage",
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70,
        },
    ]
    neutral = score_run(base_events)
    _, cheap_total = score_run_tokens(cheap_events)
    from harnessfix.scoring import TokenBudgetConfig, efficiency_score

    assert cheap_total == 70
    cfg = TokenBudgetConfig(token_budget=4096)
    assert efficiency_score(neutral, 70, cfg) >= neutral


def test_failed_run_token_score_is_zero_even_with_tiny_tokens():
    events = [
        {"kind": "token_usage", "prompt_tokens": 1, "completion_tokens": 1,
         "total_tokens": 2},
        {"kind": "loop_end", "outcome": "error"},
    ]
    outcome_score, total = score_run_tokens(events)
    assert outcome_score == 0.0
    assert total == 2
    from harnessfix.scoring import efficiency_score

    assert efficiency_score(outcome_score, total) == 0.0


def test_llm_response_events_carry_iteration_token_fields(tmp_path):
    provider = _UsageProvider(ResponseMetrics(prompt_tokens=7, completion_tokens=3))
    events = _run_loop(tmp_path, provider)
    llm_events = [e for e in events if e.get("kind") == KIND_LLM_RESPONSE]
    assert llm_events
    # The fake provider refreshes its metrics to 100/40 on every chat (see
    # _answer_llm), so each iteration's response event must carry THAT call's
    # per-iteration facts — not the constructor-seeded 7/3, and not a total.
    for ev in llm_events:
        assert ev["prompt_tokens"] == 100
        assert ev["completion_tokens"] == 40
        assert ev["total_tokens"] == 140


def _plan_facts() -> dict:
    return {
        "plan_id": "2026-11-08_12-07-35",
        "model": "test-model",
        "profile": "dev",
        "tasks_planned": 4,
        "tasks_completed": 3,
        "verifier_ok": True,
        "gates_ok": True,
        "tests_ok": True,
        "tokens": {
            "prompt_tokens": 900,
            "completion_tokens": 100,
            "total_tokens": 1000,
        },
        "latency_s": 12.5,
    }


def test_build_plan_evaluation_is_fact_based():
    ev = build_plan_evaluation(_plan_facts())
    assert ev["schema_version"] == 1
    assert ev["tasks_planned"] == 4
    assert ev["tasks_completed"] == 3
    assert ev["tasks_pass_ratio"] == pytest.approx(0.75)
    assert ev["success"] is True
    assert ev["tokens"]["total_tokens"] == 1000
    assert ev["latency_s"] == 12.5


def test_build_plan_evaluation_missing_tokens_stay_none():
    facts = _plan_facts()
    facts.pop("tokens")
    ev = build_plan_evaluation(facts)
    assert ev["tokens"] is None
    # Missing facts must not penalize: score equals the neutral outcome score.
    from harnessfix.scoring import plan_outcome

    assert ev["score"] == pytest.approx(
        plan_outcome(
            verifier_ok=True, gates_ok=True, tests_ok=True,
            tasks_completed=3, tasks_planned=4,
        )["score"]
    )


def test_write_plan_evaluation_writes_artifact_and_ledger(tmp_path):
    facts = _plan_facts()
    result = write_plan_evaluation(tmp_path, "2026-11-08_13-00-00", facts)
    eval_path = (
        tmp_path / ".docs" / "2026-11-08_13-00-00" / EVALUATION_FILE
    )
    doc = json.loads(eval_path.read_text("utf-8"))
    assert doc["plan_id"] == "2026-11-08_12-07-35"
    ledger = tmp_path / "reports" / "history" / "executions.jsonl"
    lines = [json.loads(x) for x in ledger.read_text("utf-8").splitlines()]
    assert lines[-1]["command"] == "plan_eval"
    assert lines[-1]["note"].startswith("plan 2026-11-08_12-07-35")
    assert result["path"].endswith(EVALUATION_FILE)


def test_write_plan_evaluation_never_raises(tmp_path):
    # A non-writable workspace must degrade to None, never crash the caller.
    bogus = tmp_path / "does" / "not" / "exist"
    assert write_plan_evaluation(bogus, "ts", _plan_facts()) is None
