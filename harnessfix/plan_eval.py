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
