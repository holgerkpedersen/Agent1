"""Best-effort plan-finish evaluation (Phase 2 of the self-evaluation plan).

Phase 1 built :mod:`harnessfix.plan_eval` (fact-based evaluation records +
the ``reports/history/executions.jsonl`` ledger) but nothing called it.  This
module closes that gap: when a plan finishes, gather FACTS from the plan
directory and the run traces and hand them to
:func:`harnessfix.plan_eval.write_plan_evaluation`.

Facts are never guessed (the Phase 0 contract):

* task counts come from the ``## Tasks`` block of the plan file;
* completed/failed ids come from ``plan_execution_report.md`` lines written
  by ``plan_step`` (``- [id] completed`` / ``- [id] failed``);
* token totals come from provider FACTS in ``reports/traces/*.jsonl``
  (``token_usage`` events) via :func:`harnessfix.evolution_metrics._run_total_tokens`;
* no trace, unreadable file or missing harnessfix degrades to ``None`` —
  the evaluation is still written with tokens left null.

All functions here are additive and best-effort: plan finishing must never
crash because bookkeeping failed, and it must touch no Kanban surface
(see tests/unit/test_plan_finish_kanban_isolation.py).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

#: Lines written by ``plan_step``: ``- [t1] completed (role: implementer)``.
_REPORT_LINE_RE = re.compile(
    r"^\s*-\s*\[(?P<id>[^\]]+)\]\s+(?P<status>\w+)", re.MULTILINE
)


def collect_task_counts(plan_dir: Path) -> tuple[int | None, int | None]:
    """Return ``(tasks_planned, tasks_completed)`` facts for a plan directory.

    Planned count comes from the plan markdown's ``## Tasks`` block; completed
    count from lines marked ``completed`` in ``plan_execution_report.md``.
    A missing/corrupt source yields ``None`` (fact absent, never 0).
    """
    planned: int | None = None
    plan_file: Path | None = None
    for name in ("plan_executing.md", "plan_proposed.md"):
        candidate = plan_dir / name
        if candidate.is_file():
            plan_file = candidate
            break
    if plan_file is not None:
        try:
            from agent_core.plan_execution.parser import parse_plan_tasks

            tasks = parse_plan_tasks(plan_file.read_text(encoding="utf-8"), fmt="md")
            planned = len(tasks)
        except Exception:
            planned = None

    completed: int | None = None
    report_path = plan_dir / "plan_execution_report.md"
    if report_path.is_file():
        try:
            text = report_path.read_text(encoding="utf-8")
            completed = sum(
                1
                for m in _REPORT_LINE_RE.finditer(text)
                if m.group("status").lower() == "completed"
            )
        except Exception:
            completed = None

    return planned, completed


def collect_trace_token_facts(workspace: Path) -> dict[str, Any] | None:
    """Sum provider token facts across this workspace's run traces.

    Returns a usage mapping ``{"prompt_tokens", "completion_tokens",
    "total_tokens"}`` built from the per-run ``token_usage`` events (and
    per-iteration ``llm_response`` fallbacks) that
    :func:`harnessfix.evolution_metrics._run_total_tokens` reads, or ``None``
    when no trace in ``reports/traces/`` carries token facts.  Missing stays
    missing — never guessed, never zero.
    """
    try:
        from harnessfix.evolution_metrics import _run_total_tokens
    except Exception:
        return None

    trace_dir = Path(workspace) / "reports" / "traces"
    if not trace_dir.is_dir():
        return None

    total: int | None = None
    for path in sorted(trace_dir.glob("*.jsonl")):
        try:
            events = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except Exception:
            continue
        run_total = _run_total_tokens(events)
        if run_total is not None:
            total = (total or 0) + run_total

    if total is None:
        return None
    # Provider facts are totals; prompt/completion split is not aggregated
    # across runs, so it stays None and normalize_token_usage keeps the total.
    return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": total}


def _fmt_summary(record: dict[str, Any] | None) -> str:
    """One-line summary for the plan-finish report (never raises)."""
    if not record:
        return ""
    try:
        from harnessfix.plan_eval import _fmt_tokens

        score = record.get("score")
        score_s = f"{score:.3f}" if isinstance(score, (int, float)) else "?"
        return (
            f"Evaluation: score={score_s} "
            f"tasks={record.get('tasks_completed')}/{record.get('tasks_planned')} "
            f"tokens={_fmt_tokens(record.get('tokens'))} -> {record.get('path', '?')}"
        )
    except Exception:
        return ""


def record_plan_token_metrics(
    plan_id: str, tokens: dict[str, Any] | None, score: float | None
) -> None:
    """Mirror the token FACTS of one finished plan into the shared collector.

    Emitted at the point Phase 1/2 gather token facts (plan finish), so the
    dashboard's Token efficiency panel sees live activity without re-reading
    traces.  Names follow the metrics_bridge convention — one counter per
    metric, a histogram for distributions, a gauge for the latest value —
    and every write is mirrored into the shared event file so a standalone
    ``--serve`` dashboard replays it (same plumbing as command/tool metrics).

    Missing facts are never invented: no total tokens -> no token writes;
    no score -> no score writes.  The whole function is best-effort —
    monitoring must never crash plan finishing.
    """
    try:
        from agent_core.monitoring.metrics_bridge import get_metrics_collector
        from agent_core.monitoring.metrics_file import append_event
        from harnessfix.scoring import total_tokens_of

        collector = get_metrics_collector()
        total = total_tokens_of(tokens)
        if total is not None:
            # The gate is the TOTAL fact: with a known total, record the
            # per-component distributions (prompt / completion / total) too,
            # via the shared helper so metric names agree with every other
            # recorder by construction. Components the provider did not
            # report are skipped, never zero-filled.
            from harnessfix.scoring import record_token_histograms

            record_token_histograms(tokens, collector=collector)
            tasks = collector.get_counter_value("plans.finished.count")
            per_task = total / tasks if tasks and tasks > 0 else None
            collector.increment_counter("tokens.plan.total", float(total))
            collector.record_histogram("tokens.plan.total", float(total))
            collector.set_gauge("last.tokens.plan.total", float(total))
            append_event("counter", "tokens.plan.total", float(total))
            append_event("histogram", "tokens.plan.total", float(total))
            append_event("gauge", "last.tokens.plan.total", float(total))
            if per_task is not None:
                collector.record_histogram("tokens.plan.per_task", per_task)
                append_event("histogram", "tokens.plan.per_task", per_task)
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            collector.record_histogram("plans.efficiency.score", float(score))
            collector.set_gauge("last.plans.efficiency.score", float(score))
            append_event("histogram", "plans.efficiency.score", float(score))
            append_event("gauge", "last.plans.efficiency.score", float(score))
        collector.increment_counter("plans.finished.count")
        append_event("counter", "plans.finished.count", 1.0)
    except Exception:
        pass


def write_evaluation_for_plan(plan_dir: Path, workspace: Path, success: bool) -> str:
    """Write the plan evaluation artifact + ledger line; return a report line.

    Called from ``_nlp_plan_finish`` after the lifecycle transition.  Returns
    an empty string when nothing could be written (harnessfix missing,
    unwritable workspace) — the caller appends it verbatim, so degradation is
    silent and safe.
    """
    try:
        from harnessfix.plan_eval import write_plan_evaluation

        planned, completed = collect_task_counts(plan_dir)
        tokens = collect_trace_token_facts(workspace)
        facts: dict[str, Any] = {
            "plan_id": Path(plan_dir).name,
            "tasks_planned": planned,
            # Failed plans get 0 by the scoring contract (gates_ok/tests_ok
            # are plan-level gates we do not track per-plan; success is the
            # authoritative fact and gates a failed plan to score 0).
            "tasks_completed": completed,
            "verifier_ok": bool(success),
            "gates_ok": bool(success),
            "tests_ok": bool(success),
            "regressed": False,
            "tokens": tokens,
        }
        record = write_plan_evaluation(workspace, Path(plan_dir).name, facts)
        # Dashboard token metrics: emitted right where the token facts live.
        # Counters are emitted BEFORE the efficiency score so the per-task
        # histogram divides by this plan's own plans.finished.count.
        record_plan_token_metrics(
            Path(plan_dir).name, tokens, record.get("score") if record else None
        )
        return _fmt_summary(record)
    except Exception:
        return ""


__all__ = [
    "collect_task_counts",
    "collect_trace_token_facts",
    "record_plan_token_metrics",
    "write_evaluation_for_plan",
]
