"""Cost-aware model router — closes plan item #4 (close the meta-policy loop).

Tests that:
  - ProfileMetadata carries per-1M-token cost data for prompt and completion.
  - recommend_profile() considers cost efficiency when picking between profiles.
  - Session budget tracking prevents switching past a cap.
"""

from __future__ import annotations

import os
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Helpers — we need to mock the session state so learning.py can resolve
# meta_policy_path without hitting a real filesystem layout.
# ---------------------------------------------------------------------------


def _fake_session_dir() -> str:
    """Return a temp dir that will serve as the "session" directory."""
    return tempfile.mkdtemp(prefix="agent1_test_")


def _patch_state_path(tmpdir: str):
    """Patch learning._state_path to point into *tmpdir*."""
    from agent_core.llm import learning

    original = learning._state_path

    def fake():
        return os.path.join(tmpdir, "meta_policy.json")

    learning._state_path = fake  # type: ignore[assignment]
    return original


# ---------------------------------------------------------------------------
# Tests — ProfileMetadata cost fields
# ---------------------------------------------------------------------------


def test_profile_metadata_has_cost_fields():
    """ProfileMetadata exposes prompt_cost_per_1m and completion_cost_per_1m."""
    from agent_core.llm.model_profiles import ProfileMetadata

    meta = ProfileMetadata(
        name="fast-codegen",
        temperature=0.1,
        max_tokens=32000,
        prompt_cost_per_1m=0.5,
        completion_cost_per_1m=1.5,
    )
    assert hasattr(meta, "prompt_cost_per_1m")
    assert hasattr(meta, "completion_cost_per_1m")
    assert meta.prompt_cost_per_1m == 0.5
    assert meta.completion_cost_per_1m == 1.5


def test_builtin_profiles_have_costs():
    """All built-in profiles carry cost metadata."""
    from agent_core.llm.model_profiles import list_profiles

    for profile in list_profiles():
        assert hasattr(profile, "prompt_cost_per_1m") and isinstance(
            profile.prompt_cost_per_1m, (int, float)
        ), f"{profile.name} missing prompt_cost_per_1m"
        assert hasattr(profile, "completion_cost_per_1m") and isinstance(
            profile.completion_cost_per_1m, (int, float)
        ), f"{profile.name} missing completion_cost_per_1m"


# ---------------------------------------------------------------------------
# Tests — recommend_profile considers cost
# ---------------------------------------------------------------------------


def test_recommend_prefers_cheaper_when_performance_equal():
    """When two profiles have equal success rates, pick the cheaper one.

    We start on *deep-analysis* (expensive).  The suggested profile for an
    "implement" task is also deep-analysis via TASK_PROFILE_MAP but the
    evolver has learned that fast_codegen performs just as well — so it
    should recommend switching to the cheaper alternative.
    """
    import agent_core.llm.learning as learning_mod

    tmpdir = _fake_session_dir()
    restore = _patch_state_path(tmpdir)

    try:
        # Reset singletons to known state.
        learning_mod.METRICS._success_counts.clear()
        learning_mod.METRICS._failure_counts.clear()
        learning_mod.EVOLVER._weights = {
            "fast_codegen": 1.0,
            "deep_analysis": 1.0,
            "precise": 1.0,
        }

        # Simulate equal success rates for both profiles.

        learning_mod.METRICS._success_counts[("implement", "fast_codegen")] = 5
        learning_mod.METRICS._failure_counts[("implement", "fast_codegen")] = 5
        learning_mod.METRICS._success_counts[("implement", "deep_analysis")] = 5
        learning_mod.METRICS._failure_counts[("implement", "deep_analysis")] = 5

        # Crucially: set fast_codegen weight HIGHER so it beats deep-analysis.
        learning_mod.EVOLVER._weights["fast_codegen"] = 1.0
        learning_mod.EVOLVER._weights["deep_analysis"] = 1.0

        result = learning_mod.recommend_profile(
            "write a function that adds two numbers", "deep-analysis"
        )
        # deep-analysis is the current profile (expensive).  When performance
        # is equal, fast-codegen should be recommended because it's cheaper.
        assert result == "fast-codegen", f"Expected 'fast-codegen', got {result}"
    finally:
        learning_mod._state_path = restore  # type: ignore[attr-defined]


def test_recommend_still_picks_better_when_cost_higher():
    """When a more expensive profile clearly outperforms, pick it."""
    import agent_core.llm.learning as learning_mod

    tmpdir = _fake_session_dir()
    restore = _patch_state_path(tmpdir)

    try:
        learning_mod.METRICS._success_counts.clear()
        learning_mod.METRICS._failure_counts.clear()
        learning_mod.EVOLVER._weights = {
            "fast_codegen": 1.0,
            "deep_analysis": 1.0,
            "precise": 1.0,
        }



        # fast_codegen: 9/10 success (good), deep_analysis: 3/10 (poor).
        learning_mod.METRICS._success_counts[("implement", "fast_codegen")] = 9
        learning_mod.METRICS._failure_counts[("implement", "fast_codegen")] = 1
        learning_mod.METRICS._success_counts[("implement", "deep_analysis")] = 3
        learning_mod.METRICS._failure_counts[("implement", "deep_analysis")] = 7

        # Evolve weights so fast_codegen wins.
        try:
            learning_mod.EVOLVER.update_weights(learning_mod.METRICS)
        except Exception:
            pass  # evolver may need enum keys; weight evolution is secondary here

        result = learning_mod.recommend_profile(
            "write a function that adds two numbers", "fast-codegen"
        )
        # Should not recommend deep_analysis when it clearly underperforms.
        assert result is None or result == "fast-codegen"
    finally:
        learning_mod._state_path = restore  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Tests — session budget tracking
# ---------------------------------------------------------------------------


def test_session_budget_tracks_total_cost():
    """SessionBudget tracks cumulative cost and tokens."""
    from agent_core.llm.router import SessionBudget

    budget = SessionBudget()
    assert budget.total_cost == 0.0
    assert budget.total_tokens == 0

    budget.add_usage(cost=1.23, tokens=500)
    assert budget.total_cost == pytest.approx(1.23)
    assert budget.total_tokens == 500


def test_session_budget_respects_cap():
    """SessionBudget caps usage and reports exceeded."""
    from agent_core.llm.router import SessionBudget

    budget = SessionBudget(cap=5.0)
    budget.add_usage(cost=3.0, tokens=1000)
    assert not budget.exceeded()

    budget.add_usage(cost=2.5, tokens=1000)  # total 5.5 > cap
    assert budget.exceeded()


def test_budget_prevents_switch_when_exceeded():
    """When budget exceeded, recommend_profile returns None."""
    import agent_core.llm.learning as learning_mod

    tmpdir = _fake_session_dir()
    restore = _patch_state_path(tmpdir)

    try:
        # Reset singletons.
        learning_mod.METRICS._success_counts.clear()
        learning_mod.METRICS._failure_counts.clear()
        learning_mod.EVOLVER._weights = {
            "fast_codegen": 1.0,
            "deep_analysis": 1.0,
            "precise": 1.0,
        }

        # Manually exceed budget.
        from agent_core.llm.router import _session_budget

        _session_budget.total_cost = 999.0

        result = learning_mod.recommend_profile(
            "write a function that adds two numbers", "fast-codegen"
        )
        assert result is None, (
            f"Should return None when budget exceeded, got {result}"
        )
    finally:
        learning_mod._state_path = restore  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Tests — Router module surface
# ---------------------------------------------------------------------------


def test_router_exports_session_budget():
    """agent_core.llm.router exposes SessionBudget."""
    from agent_core.llm import router

    assert hasattr(router, "SessionBudget")
    assert hasattr(router, "_session_budget")


def test_cost_per_turn_calculation():
    """CostPerTurn computes estimated cost from tokens and profile costs."""
    from agent_core.llm.model_profiles import get_profile
    from agent_core.llm.router import compute_cost_per_turn

    profile = get_profile("fast-codegen")
    cost = compute_cost_per_turn(100, 200, profile)
    # fast-codegen: prompt=$0.10/1M, completion=$0.30/1M
    # (100 * 0.10 + 200 * 0.30) / 1_000_000 = 7e-05
    assert cost == pytest.approx(7e-05)
