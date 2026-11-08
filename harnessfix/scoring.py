"""Fact-based scoring contract for Agent1 (token efficiency + plan outcomes).

Phase 0 of the token-efficiency / self-evaluation / dashboard plan
(`2026-11-08-token-efficiency-self-eval-dashboard`).

This module is PURE and DETERMINISTIC: no I/O, no LLM, no clock, no network.
It defines the scoring formulas and a versioned score record that later phases
instrument and surface on the dashboard.

Principles (fact-based, never fabricated):
- Success/outcome follows trace semantics already used by
  ``harnessfix.evolution_metrics`` (completed / guard-delivered / failed).
- Token metrics use ONLY provider-reported counts; missing -> ``None`` (never
  guessed). A ``None`` token count must never be treated as 0.
- Efficiency is computed so a failed/undelivered run cannot score well on
  tokens alone: the composite score is gated by outcome success.
- Regression is a numeric comparison (windowed delta or baseline delta).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

SCORING_SCHEMA_VERSION = 1

# Outcome bases mirror harnessfix.evolution_metrics (kept as defaults so this
# module stays leaf-pure; Phase 1 composes with the authoritative constants).
DELIVERED_OUTCOMES: frozenset[str] = frozenset(
    {"stuck", "budget_exhausted", "no_progress"}
)
DEFAULT_COMPLETED_BASE = 1.0
DEFAULT_DELIVERED_BASE = 0.9

# Efficiency defaults (configurable per model|profile via TokenBudgetConfig).
DEFAULT_TOKEN_BUDGET = 4096  # total tokens considered "within budget"
DEFAULT_EFFICIENCY_WEIGHT = 0.3  # share of composite score from token efficiency
DEFAULT_REGRESSION_DELTA = 0.05
DEFAULT_MIN_ANSWER_CHARS = 80  # mirrors htir.MIN_FINAL_ANSWER_CHARS


def _clamp01(value: float) -> float:
    if not isfinite(value):
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _nonneg_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and isfinite(value) and value.is_integer():
        iv = int(value)
        return iv if iv >= 0 else None
    return None


def normalize_token_usage(
    usage: Mapping[str, Any] | None,
    *,
    prompt_tokens: Any = None,
    completion_tokens: Any = None,
    total_tokens: Any = None,
) -> dict[str, int | None] | None:
    """Return a fact-based token dict or ``None`` when no usable counts exist.

    Accepts either a usage mapping (``prompt_tokens``/``completion_tokens``/
    ``total_tokens``) or explicit keyword values. Counts must be provider-
    reported non-negative integers (or whole-number floats). Missing/invalid
    values stay ``None``; ``total_tokens`` is derived from prompt+completion
    only when both are known and total is absent. Returns ``None`` if every
    field is unknown — callers must not fabricate a zero.
    """
    src: Mapping[str, Any] = usage or {}
    p = _nonneg_int(
        prompt_tokens if prompt_tokens is not None else src.get("prompt_tokens")
    )
    c = _nonneg_int(
        completion_tokens
        if completion_tokens is not None
        else src.get("completion_tokens")
    )
    t = _nonneg_int(
        total_tokens if total_tokens is not None else src.get("total_tokens")
    )
    if t is None and p is not None and c is not None:
        t = p + c
    if p is None and c is None and t is None:
        return None
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": t}


def total_tokens_of(tokens: Mapping[str, Any] | None) -> int | None:
    """Best-effort total token count from a normalized token dict, or ``None``."""
    if not tokens:
        return None
    t = _nonneg_int(tokens.get("total_tokens"))
    if t is not None:
        return t
    p = _nonneg_int(tokens.get("prompt_tokens"))
    c = _nonneg_int(tokens.get("completion_tokens"))
    if p is not None and c is not None:
        return p + c
    return None


def has_substantive_answer(
    text: str | None, min_chars: int = DEFAULT_MIN_ANSWER_CHARS
) -> bool:
    return len((text or "").strip()) >= min_chars


def outcome_base(
    outcome: str | None,
    *,
    delivered: bool,
    completed_base: float = DEFAULT_COMPLETED_BASE,
    delivered_base: float = DEFAULT_DELIVERED_BASE,
    delivered_outcomes: frozenset[str] = DELIVERED_OUTCOMES,
) -> float:
    """Base score from loop outcome, fact-based (no latency/tokens).

    - ``completed`` -> ``completed_base`` (default 1.0)
    - guard-terminated (stuck/budget_exhausted/no_progress) AND delivered
      -> ``delivered_base`` (0.9)
    - anything else (error, interrupted/no loop_end, undelivered guard stop) -> 0.0
    """
    if outcome == "completed":
        return _clamp01(completed_base)
    if outcome in delivered_outcomes and delivered:
        return _clamp01(delivered_base)
    return 0.0


def apply_latency_penalty(
    base: float,
    latency_s: float,
    latency_budget_s: float = 30.0,
    penalty_weight: float = 0.3,
) -> float:
    if latency_budget_s <= 0:
        penalty = 0.0
    else:
        lat = 0.0
        if isinstance(latency_s, (int, float)) and isfinite(latency_s):
            if latency_s > 0:
                lat = float(latency_s)
        penalty = min(lat / latency_budget_s, 1.0) * penalty_weight
    return _clamp01(base - penalty)


@dataclass(frozen=True)
class TokenBudgetConfig:
    """Efficiency budget/weights, configurable per ``model|profile`` (Open/Closed)."""

    token_budget: int = DEFAULT_TOKEN_BUDGET
    efficiency_weight: float = DEFAULT_EFFICIENCY_WEIGHT

    @property
    def success_weight(self) -> float:
        return _clamp01(1.0 - _clamp01(self.efficiency_weight))


def token_efficiency_component(total_tokens: int | None, budget: int) -> float | None:
    """Return [0,1] efficiency for a known token count, else ``None``.

    1.0 at/below half-budget, linear down to 0.0 at 2x budget (clamped).
    ``None`` when tokens are unknown — callers must not fabricate efficiency.
    """
    if total_tokens is None:
        return None
    if budget <= 0:
        return 1.0
    # scale: <= budget/2 -> 1, >= 2*budget -> 0
    lo = budget / 2.0
    hi = budget * 2.0
    if total_tokens <= lo:
        return 1.0
    if total_tokens >= hi:
        return 0.0
    return _clamp01(1.0 - (total_tokens - lo) / (hi - lo))


def efficiency_score(
    outcome_score: float,
    total_tokens: int | None,
    config: TokenBudgetConfig | None = None,
) -> float:
    """Composite efficiency-aware score in [0,1], gated by outcome.

    When tokens are unknown (``None``), the outcome score passes through
    unchanged (neutral) — missing facts never penalize and never reward.
    When tokens are known, efficiency modulates within the outcome band:
    ``outcome * (success_weight + efficiency_weight * eff)`` so the result
    never exceeds ``outcome_score`` and a failed/undelivered run (outcome 0)
    always yields 0 — tokens cannot credit failure.
    """
    cfg = config or TokenBudgetConfig()
    base = _clamp01(outcome_score)
    if base <= 0.0:
        return 0.0
    eff = token_efficiency_component(total_tokens, cfg.token_budget)
    if eff is None:
        return base
    return _clamp01(base * (cfg.success_weight + _clamp01(cfg.efficiency_weight) * eff))


def plan_outcome(
    *,
    verifier_ok: bool,
    gates_ok: bool,
    tests_ok: bool,
    tasks_completed: int | None = None,
    tasks_planned: int | None = None,
    regressed: bool = False,
) -> dict[str, Any]:
    """Fact-based plan/workflow outcome summary.

    ``tasks_pass_ratio`` is ``completed/planned`` when both known and planned>0,
    else ``None``. Success requires verifier+gates+tests ok and no regression.
    """
    ratio: float | None = None
    if tasks_completed is not None and tasks_planned is not None and tasks_planned > 0:
        ratio = _clamp01(tasks_completed / tasks_planned)
    success = bool(verifier_ok and gates_ok and tests_ok and not regressed)
    # score favors verified passing runs; ratio (if known) refines completeness
    if not success:
        score = 0.0 if regressed or not (verifier_ok and gates_ok and tests_ok) else 0.0
    else:
        score = ratio if ratio is not None else 1.0
    return {
        "success": success,
        "score": _clamp01(score),
        "tasks_pass_ratio": ratio,
        "verifier_ok": bool(verifier_ok),
        "gates_ok": bool(gates_ok),
        "tests_ok": bool(tests_ok),
        "regressed": bool(regressed),
    }


def is_regression(
    current: float | None,
    baseline: float | None,
    delta: float = DEFAULT_REGRESSION_DELTA,
    *,
    higher_is_better: bool = True,
) -> bool | None:
    """True if ``current`` regressed vs ``baseline`` by more than ``delta``.

    Returns ``None`` when either value is missing/non-finite (fact unknown).
    """
    if current is None or baseline is None:
        return None
    if not (isfinite(current) and isfinite(baseline) and isfinite(delta)):
        return None
    drop = (baseline - current) if higher_is_better else (current - baseline)
    return drop > delta


#: Canonical token-metric names for the monitoring collector. Keeping the map
#: here (not in the collector call sites) means the dashboard histogram picker
#: and every recorder agree on the names by construction.
TOKEN_HISTOGRAM_METRICS: dict[str, str] = {
    "prompt_tokens": "tokens.run.prompt",
    "completion_tokens": "tokens.run.completion",
    "total_tokens": "tokens.run.total",
}


def token_histogram_samples(
    tokens: Mapping[str, Any] | None,
) -> dict[str, float]:
    """Map a usage mapping to ``{metric_name: value}`` for histograms (pure).

    Only finite non-negative numbers become samples; a missing key, ``None``
    (the provider did not report it) or a non-numeric value yields no entry,
    so absent facts never reach the distribution. An empty dict therefore
    means "nothing to record" - never invent a zero.
    """
    usage = normalize_token_usage(tokens)
    out: dict[str, float] = {}
    for key, metric in TOKEN_HISTOGRAM_METRICS.items():
        value = usage.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and isfinite(value) and value >= 0:
            out[metric] = float(value)
    return out


def record_token_histograms(
    tokens: Mapping[str, Any] | None,
    *,
    collector: Any = None,
) -> dict[str, float]:
    """Record per-run token samples into the shared metrics collector.

    Returns the samples it recorded (empty when there are no facts). Never
    raises: a missing/broken collector degrades to a no-op, because metrics
    bookkeeping must not break the run that produced the facts.
    """
    samples = token_histogram_samples(tokens)
    if not samples or collector is None:
        return {}
    try:
        from agent_core.monitoring.metrics_collector import (
            MetricsCollector,
            append_event,
        )

        target = collector if collector is not None else MetricsCollector.get_instance()
        for metric, value in samples.items():
            target.record_histogram(metric, value)
            append_event("histogram", metric, value)
    except Exception:  # pragma: no cover - metrics must never break a run
        return {}
    return samples


def windowed_summary(scores: Sequence[float]) -> dict[str, float | int | None]:
    """Deterministic summary over a score window (pure)."""
    vals = [float(s) for s in scores if isinstance(s, (int, float)) and isfinite(s)]
    if not vals:
        return {"count": 0, "avg": None, "median": None, "min": None, "max": None}
    avg = sum(vals) / len(vals)
    return {
        "count": len(vals),
        "avg": avg,
        "median": float(median(vals)),
        "min": min(vals),
        "max": max(vals),
    }


@dataclass
class ScoreRecord:
    """Versioned, fact-based score record (dashboard/ledger payload)."""

    schema_version: int = SCORING_SCHEMA_VERSION
    run_id: str | None = None
    task_id: str | None = None
    plan_id: str | None = None
    model: str | None = None
    profile: str | None = None
    source: str = "unknown"  # trace|plan|workflow|chat|benchmark
    success: bool = False
    score: float = 0.0
    outcome: str | None = None
    tokens: dict[str, int | None] | None = None
    latency_s: float | None = None
    tasks_pass_ratio: float | None = None
    regression_flags: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["score"] = _clamp01(float(self.score))
        return d


def make_score_record(
    *,
    run_id: str | None = None,
    task_id: str | None = None,
    plan_id: str | None = None,
    model: str | None = None,
    profile: str | None = None,
    source: str = "trace",
    success: bool,
    score: float,
    outcome: str | None = None,
    tokens: Mapping[str, Any] | None = None,
    latency_s: Any = None,
    tasks_pass_ratio: Any = None,
    regression_flags: Iterable[str] | None = None,
    facts: Mapping[str, Any] | None = None,
) -> ScoreRecord:
    lat: float | None = None
    if isinstance(latency_s, (int, float)) and isfinite(latency_s) and latency_s >= 0:
        lat = float(latency_s)
    ratio: float | None = None
    if isinstance(tasks_pass_ratio, (int, float)) and isfinite(tasks_pass_ratio):
        ratio = _clamp01(float(tasks_pass_ratio))
    return ScoreRecord(
        run_id=run_id,
        task_id=task_id,
        plan_id=plan_id,
        model=model,
        profile=profile,
        source=source,
        success=bool(success),
        score=_clamp01(float(score) if isfinite(score) else 0.0),
        outcome=outcome,
        tokens=normalize_token_usage(tokens) if tokens else None,
        latency_s=lat,
        tasks_pass_ratio=ratio,
        regression_flags=list(regression_flags or []),
        facts=dict(facts or {}),
    )
