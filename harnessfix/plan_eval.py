"""Fact-based plan evaluation artifact (Phase 1 of the self-evaluation plan).

Builds a per-plan evaluation record from FACTS (planned/completed task counts,
verifier/gates/tests flags, provider token facts, latency) using the Phase 0
scoring contract (:mod:`harnessfix.scoring`), and writes it ADDITIVELY:

* ``.docs/<ts>/evaluation.json`` — the per-plan artifact;
* one appended line in ``reports/history/executions.jsonl`` — the execution
  ledger read by :mod:`harnessfix.history`, so plan outcomes join the same
  history corpus as run/trace events.

Principles (identical to :mod:`harnessfix.scoring`):
- missing facts stay ``None`` and are never guessed, never treated as zero;
- a failed plan scores 0 regardless of how cheap its tokens were;
- writing is best-effort: an unwritable workspace degrades to ``None`` and
  must never crash the caller.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

from .scoring import (
    SCORING_SCHEMA_VERSION,
    TokenBudgetConfig,
    efficiency_score,
    normalize_token_usage,
    plan_outcome,
)

#: Name of the per-plan evaluation artifact written under ``.docs/<ts>/``.
EVALUATION_FILE = "evaluation.json"
#: Ledger location (mirrors harnessfix.history: reports/history/executions.jsonl).
HISTORY_SUBDIR = "history"
EXECUTIONS_FILE = "executions.jsonl"
#: Command tag for this record kind in the ledger.
LEDGER_COMMAND = "plan_eval"

#: Aggregated plan-evaluation report file (gap 2). Append-only: one line per
#: finished plan, so all evaluations live in a single aggregated report file
#: alongside the per-plan ``.docs/<ts>/evaluation.json`` artifacts.
PLAN_EVALUATIONS_FILE = "plan_evaluations.jsonl"
#: Cumulative prompt/completion token-split summary across runs (gap 3).
PLAN_TOKEN_AGGREGATE_FILE = "plan_token_aggregate.json"


def build_plan_evaluation(facts: Mapping[str, Any]) -> dict[str, Any]:
    """Return a schema-versioned evaluation record from plan *facts*.

    Recognised facts (all optional): ``plan_id``, ``model``, ``profile``,
    ``tasks_planned``, ``tasks_completed``, ``verifier_ok``, ``gates_ok``,
    ``tests_ok``, ``regressed``, ``tokens`` (a usage mapping with
    ``prompt_tokens``/``completion_tokens``/``total_tokens``), ``latency_s``.

    The record is fact-based: unknown values stay ``None``; tokens are
    normalised through :func:`harnessfix.scoring.normalize_token_usage` so a
    missing usage mapping yields ``None`` (never 0).  ``score`` is the plan
    outcome score modulated by token efficiency — neutral when tokens are
    unknown, and gated so a failed plan can never be credited by tokens.
    """
    src: Mapping[str, Any] = facts or {}
    verifier_ok = bool(src.get("verifier_ok", False))
    gates_ok = bool(src.get("gates_ok", False))
    tests_ok = bool(src.get("tests_ok", False))
    regressed = bool(src.get("regressed", False))
    tasks_planned = _int_or_none(src.get("tasks_planned"))
    tasks_completed = _int_or_none(src.get("tasks_completed"))

    outcome = plan_outcome(
        verifier_ok=verifier_ok,
        gates_ok=gates_ok,
        tests_ok=tests_ok,
        tasks_completed=tasks_completed,
        tasks_planned=tasks_planned,
        regressed=regressed,
    )
    tokens = normalize_token_usage(src.get("tokens"))
    from .scoring import total_tokens_of

    total = total_tokens_of(tokens)
    score = efficiency_score(
        outcome["score"],
        total,
        TokenBudgetConfig(),
    )
    latency = src.get("latency_s")
    if isinstance(latency, (int, float)) and not isinstance(latency, bool):
        latency = float(latency)
    else:
        latency = None

    return {
        "schema_version": SCORING_SCHEMA_VERSION,
        "kind": "plan_evaluation",
        "plan_id": str(src.get("plan_id") or ""),
        "model": src.get("model"),
        "profile": src.get("profile"),
        "tasks_planned": tasks_planned,
        "tasks_completed": tasks_completed,
        "tasks_pass_ratio": outcome["tasks_pass_ratio"],
        "success": outcome["success"],
        "score": score,
        "outcome_score": outcome["score"],
        "verifier_ok": verifier_ok,
        "gates_ok": gates_ok,
        "tests_ok": tests_ok,
        "regressed": regressed,
        "tokens": tokens,
        "latency_s": latency,
    }


def write_plan_evaluation(
    workspace: str | Path,
    ts: str,
    facts: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Write ``.docs/<ts>/evaluation.json`` and append a ledger line.

    Best-effort and additive: creates the artifact directories inside an
    EXISTING workspace, appends (never rewrites) the ledger, and returns the
    record with its artifact ``path``.  Returns ``None`` when *workspace* is
    not a real directory, or the filesystem refuses the write — evaluation
    bookkeeping must never crash the caller, and it must never invent a
    workspace tree that does not exist.
    """
    try:
        record = build_plan_evaluation(facts)
        root = Path(workspace)
        if not root.is_dir():
            return None
        doc_dir = root / ".docs" / str(ts)
        doc_dir.mkdir(parents=True, exist_ok=True)
        eval_path = doc_dir / EVALUATION_FILE
        eval_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        ledger_dir = root / "reports" / HISTORY_SUBDIR
        ledger_dir.mkdir(parents=True, exist_ok=True)
        ledger = ledger_dir / EXECUTIONS_FILE
        line = {
            "ts": time.time(),
            "command": LEDGER_COMMAND,
            "outcome": "success" if record["success"] else "failed",
            "note": (
                f"plan {record['plan_id']}: score={record['score']:.3f} "
                f"tasks={record['tasks_completed']}/{record['tasks_planned']} "
                f"tokens={_fmt_tokens(record['tokens'])}"
            ),
        }
        with ledger.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")

        record["path"] = str(eval_path)
        return record
    except OSError:
        return None


def append_plan_evaluation_aggregate(
    workspace: str | Path,
    record: Mapping[str, Any],
) -> bool:
    """Append one aggregated plan-evaluation line to ``reports/plan_evaluations.jsonl``.

    Best-effort and append-only: creates ``reports/`` if needed, appends (never
    rewrites) one JSON line summarising the evaluation, and returns True on
    success. Never crashes the caller and never writes outside an existing
    workspace — evaluation bookkeeping must never break plan finishing.
    """
    try:
        root = Path(workspace)
        if not root.is_dir():
            return False
        report_path = root / "reports" / PLAN_EVALUATIONS_FILE
        report_path.parent.mkdir(parents=True, exist_ok=True)
        line = {
            "ts": time.time(),
            "kind": "plan_evaluation_aggregate",
            "plan_id": record.get("plan_id"),
            "tasks_planned": record.get("tasks_planned"),
            "tasks_completed": record.get("tasks_completed"),
            "success": record.get("success"),
            "score": record.get("score"),
            "tokens": record.get("tokens"),
            "tokens_total": _tokens_total(record.get("tokens")),
            "path": record.get("path"),
        }
        with report_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False


def _tokens_total(tokens: Mapping[str, Any] | None) -> int | None:
    from .scoring import total_tokens_of

    if not tokens:
        return None
    return total_tokens_of(tokens)


def write_plan_token_aggregate(workspace: str | Path) -> dict[str, Any] | None:
    """Recompute the cumulative prompt/completion token-split across runs.

    Reads every line of ``reports/plan_evaluations.jsonl`` (gap 2's aggregated
    file) and sums prompt/completion/total token facts, None-safe (missing facts
    stay None, never 0), writing the cumulative summary to
    ``reports/plan_token_aggregate.json``. Best-effort and additive: returns the
    summary on success, ``None`` when the workspace is missing/unwritable.

    Prompt/completion split is per-run (``token_usage`` events + per-run
    histograms); this aggregates the totals across runs into a single
    fact-based summary, so the split is no longer "totals only".
    """
    try:
        root = Path(workspace)
        if not root.is_dir():
            return None
        report_path = root / "reports" / PLAN_EVALUATIONS_FILE
        if not report_path.is_file():
            summary = _empty_token_aggregate()
            _write_json(root, PLAN_TOKEN_AGGREGATE_FILE, summary)
            return summary
        total_prompt: float | None = None
        total_completion: float | None = None
        total_tokens: float | None = None
        count = 0
        with report_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue  # corrupt line skipped, never guessed
                toks = ev.get("tokens")
                if not isinstance(toks, Mapping):
                    continue
                prompt = _finite(toks.get("prompt_tokens"))
                completion = _finite(toks.get("completion_tokens"))
                total_t = _finite(toks.get("total_tokens"))
                if prompt is None and completion is None and total_t is None:
                    continue  # line without usable token facts: not a plan eval
                if prompt is not None:
                    total_prompt = _sum(total_prompt, prompt)
                if completion is not None:
                    total_completion = _sum(total_completion, completion)
                if total_t is not None:
                    total_tokens = _sum(total_tokens, total_t)
                count += 1
        summary = _token_aggregate(
            prompt=total_prompt,
            completion=total_completion,
            total=total_tokens,
            count=count,
        )
        _write_json(root, PLAN_TOKEN_AGGREGATE_FILE, summary)
        return summary
    except OSError:
        return None
    except Exception:
        return None


def _empty_token_aggregate() -> dict[str, Any]:
    return _token_aggregate(
        prompt=None, completion=None, total=None, count=0
    )


def _token_aggregate(
    prompt: float | None,
    completion: float | None,
    total: float | None,
    count: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCORING_SCHEMA_VERSION,
        "kind": "plan_token_aggregate",
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "plan_evaluations": count,
    }


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0:
        return None
    return value


def _sum(current: float | None, value: float) -> float | None:
    """Add *value* to *current*; ``None`` accumulates to ``None`` (no facts
    seen yet — never guessed as zero)."""
    if current is None:
        return value
    return current + value


def _write_json(root: Path, filename: str, obj: dict[str, Any]) -> None:
    path = root / "reports" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    return None


def _fmt_tokens(tokens: Mapping[str, Any] | None) -> str:
    if not tokens:
        return "None"
    from .scoring import total_tokens_of

    total = total_tokens_of(tokens)
    return "None" if total is None else str(total)
