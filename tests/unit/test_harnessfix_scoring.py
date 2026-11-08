"""Unit tests for harnessfix.scoring (pure, fact-based).

Phase 0 contract: deterministic formulas, no I/O, no LLM. Tokens are
provider-reported facts only — missing stays None, never fabricated.
"""

from __future__ import annotations

import pytest

from harnessfix.scoring import (
    SCORING_SCHEMA_VERSION,
    TokenBudgetConfig,
    apply_latency_penalty,
    efficiency_score,
    has_substantive_answer,
    is_regression,
    make_score_record,
    normalize_token_usage,
    outcome_base,
    plan_outcome,
    token_efficiency_component,
    total_tokens_of,
    windowed_summary,
)


class TestNormalizeTokenUsage:
    def test_missing_returns_none(self):
        assert normalize_token_usage(None) is None
        assert normalize_token_usage({}) is None

    def test_provider_reported_fields(self):
        t = normalize_token_usage(
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
        assert t == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    def test_derives_total_from_prompt_and_completion(self):
        t = normalize_token_usage({"prompt_tokens": 10, "completion_tokens": 5})
        assert t["total_tokens"] == 15

    def test_does_not_fabricate_missing_side(self):
        t = normalize_token_usage({"prompt_tokens": 10})
        assert t == {
            "prompt_tokens": 10,
            "completion_tokens": None,
            "total_tokens": None,
        }
        assert total_tokens_of(t) is None

    def test_rejects_invalid_and_bool(self):
        assert normalize_token_usage({"total_tokens": -1}) is None
        assert normalize_token_usage({"total_tokens": "12"}) is None
        assert normalize_token_usage({"total_tokens": True}) is None

    def test_explicit_kwargs(self):
        t = normalize_token_usage(None, total_tokens=7)
        assert t["total_tokens"] == 7


class TestOutcomeBase:
    def test_completed(self):
        assert outcome_base("completed", delivered=True) == 1.0

    def test_guard_delivered(self):
        for o in ("stuck", "budget_exhausted", "no_progress"):
            assert outcome_base(o, delivered=True) == pytest.approx(0.9)

    def test_guard_not_delivered(self):
        for o in ("stuck", "budget_exhausted", "no_progress"):
            assert outcome_base(o, delivered=False) == 0.0

    def test_error_even_if_delivered_is_failure(self):
        assert outcome_base("error", delivered=True) == 0.0

    def test_interrupted_no_outcome_is_failure(self):
        assert outcome_base(None, delivered=True) == 0.0

    def test_latency_penalty(self):
        assert apply_latency_penalty(
            1.0, 5.0, latency_budget_s=30.0, penalty_weight=0.3
        ) == pytest.approx(0.95)
        assert apply_latency_penalty(0.9, 5.0) == pytest.approx(0.85)


class TestTokenEfficiency:
    def test_unknown_tokens_returns_none_component(self):
        assert token_efficiency_component(None, 4096) is None

    def test_efficiency_score_neutral_when_tokens_unknown(self):
        # Missing facts never penalize and never reward.
        assert efficiency_score(0.95, None) == pytest.approx(0.95)
        assert efficiency_score(1.0, None) == pytest.approx(1.0)

    def test_failure_gated_never_credited_by_tokens(self):
        assert efficiency_score(0.0, 1) == 0.0
        assert efficiency_score(0.0, 999999) == 0.0
        assert efficiency_score(0.0, None) == 0.0

    def test_known_tokens_blend_but_respect_outcome(self):
        cfg = TokenBudgetConfig(token_budget=4000, efficiency_weight=0.3)
        cheap = efficiency_score(1.0, 100, cfg)
        pricey = efficiency_score(1.0, 12000, cfg)
        assert 0.0 <= pricey <= 1.0
        assert cheap >= pricey
        # delivered-guard base (0.9) stays <= 0.9
        assert efficiency_score(0.9, 100, cfg) <= 0.9

    def test_component_boundaries(self):
        b = 4000
        assert token_efficiency_component(b // 2, b) == 1.0
        assert token_efficiency_component(b * 2, b) == 0.0


class TestPlanOutcome:
    def test_green_plan(self):
        r = plan_outcome(
            verifier_ok=True, gates_ok=True, tests_ok=True,
            tasks_completed=3, tasks_planned=3,
        )
        assert r["success"] is True and r["score"] == 1.0
        assert r["tasks_pass_ratio"] == 1.0

    def test_partial(self):
        r = plan_outcome(
            verifier_ok=True, gates_ok=True, tests_ok=True,
            tasks_completed=1, tasks_planned=2,
        )
        assert r["success"] is True and r["score"] == pytest.approx(0.5)

    def test_regression_or_gate_fail(self):
        failed_gates = plan_outcome(verifier_ok=True, gates_ok=False, tests_ok=True)
        regressed = plan_outcome(
            verifier_ok=True, gates_ok=True, tests_ok=True, regressed=True
        )
        assert failed_gates["success"] is False
        assert regressed["success"] is False


class TestRegression:
    def test_detects_drop(self):
        assert is_regression(0.6, 0.8, delta=0.05) is True
        assert is_regression(0.78, 0.8, delta=0.05) is False  # within delta
        assert is_regression(0.8, 0.8, delta=0.05) is False

    def test_unknown_is_none(self):
        assert is_regression(None, 0.8) is None
        assert is_regression(0.7, None) is None

    def test_windowed_summary(self):
        s = windowed_summary([0.8, 0.6, 1.0])
        assert s["count"] == 3 and s["min"] == 0.6 and s["max"] == 1.0
        assert windowed_summary([])["avg"] is None


class TestScoreRecord:
    def test_record_is_fact_based_and_versioned(self):
        rec = make_score_record(
            run_id="r1", task_id="t1", model="m", profile="p", source="trace",
            success=True, score=0.95, outcome="completed",
            tokens={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            latency_s=1.2, regression_flags=[],
        )
        d = rec.to_dict()
        assert d["schema_version"] == SCORING_SCHEMA_VERSION
        assert d["success"] is True and d["tokens"]["total_tokens"] == 15

    def test_missing_tokens_stay_null(self):
        rec = make_score_record(run_id="r2", success=False, score=0.0, tokens=None)
        assert rec.tokens is None

    def test_substantive_answer(self):
        assert has_substantive_answer("x" * 80) is True
        assert has_substantive_answer("short") is False
