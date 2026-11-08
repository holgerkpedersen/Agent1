# Token Self-Evaluation Plan (Phases 0-2)

Status: **shipped** (commits `cb0c5fc`, `0783faa`, `0323cec`, pushed to `origin/master`).
Written retrospectively — this feature shipped across three commits without a plan doc;
this closes that gap.

## Goal

Give Agent1 fact-based self-evaluation of token efficiency: score finished plans and
runs from provider-reported token facts, persist the evaluations, and show them on the
dashboard. The governing rule (the "Phase 0 contract") is **facts are never guessed**:
a missing fact stays `None`, it is never fabricated as `0`.

## Phase 0 — scoring contract (`cb0c5fc`)

Module: `harnessfix/scoring.py` — pure, deterministic, no I/O.

- `normalize_token_usage(...)` — provider token normalization; missing stays `None`.
- `total_tokens_of(tokens)` — total or `None`.
- `outcome_base(...)` — outcome score compatible with `evolution_metrics` semantics;
  `apply_latency_penalty(...)` for latency.
- `token_efficiency_component(total_tokens, budget)` + `efficiency_score(outcome, tokens, config)`
  — efficiency gated by outcome, budget via `TokenBudgetConfig`.
- `plan_outcome(...)` / `is_regression(...)` / `windowed_summary(...)` — plan outcome and
  regression helpers.
- `ScoreRecord` / `make_score_record(...)` — versioned record with token facts.

Tests: `tests/unit/test_harnessfix_scoring.py` (26).

## Phase 1 — capture + score token facts (`0783faa`)

- `agent_core/llm/tool_loop.py` (+ `agent_core/llm/parallel.py`, `harnessfix/tracing.py`):
  the tool loop emits additive `token_usage` events and per-iteration token fields when
  the provider reports usage.
- `harnessfix/evolution_metrics.py`: `score_run_tokens(...)` composes the outcome score
  with provider token totals (`None` when unknown).
- `harnessfix/plan_eval.py`: `build_plan_evaluation(facts)` / `write_plan_evaluation(...)`
  build fact-based plan evaluations, write `.docs/<ts>/evaluation.json` and one
  `plan_evaluation` line in the `reports/history/executions.jsonl` ledger
  (same ledger as `harnessfix.history`).

Tests: `tests/unit/test_harnessfix_phase1_tokens.py` (11).

## Phase 2 — plan-finish evaluation + dashboard panel (`0323cec`)

Module: `agent_core/plan_execution/plan_finish_eval.py`. Phase 1 wrote the evaluator
but nothing called it; Phase 2 closes that gap.

- `collect_task_counts(plan_dir)` — planned count from the plan markdown `## Tasks`
  block (via `plan_execution.parser.parse_plan_tasks`); completed count from
  `plan_execution_report.md` lines written by `plan_step` (`- [id] completed`).
  Missing/corrupt source yields `None`, never `0`.
- `collect_trace_token_facts(workspace)` — sums provider token facts across
  `reports/traces/*.jsonl` via `evolution_metrics._run_total_tokens`; prompt/completion
  split stays `None` (not aggregated across runs).
- `record_plan_token_metrics(plan_id, tokens, score)` — mirrors token facts into the
  shared `MetricsCollector` at plan finish. Metric names follow the metrics_bridge
  convention and every write is mirrored into the shared event file so a standalone
  `--serve` dashboard replays it:
  - counters: `tokens.plan.total`, `plans.finished.count`
  - histograms: `tokens.plan.total`, `tokens.plan.per_task`, `plans.efficiency.score`
  - gauges: `last.tokens.plan.total`, `last.plans.efficiency.score`
- `write_evaluation_for_plan(plan_dir, workspace, success)` — assembles the facts
  (`verifier_ok`/`gates_ok`/`tests_ok` = plan success; a failed plan scores 0 by the
  Phase 0 contract), calls `harnessfix.plan_eval.write_plan_evaluation`, then records
  the dashboard metrics. Returns a one-line summary for the finish report.

Call site: `agent_core/handlers/nlp_tools.py` `_nlp_plan_finish` invokes
`write_evaluation_for_plan(plan_dir, workspace, success)` after the lifecycle transition.

Dashboard API: `agent_core/monitoring/dashboard_api.py` serves **`GET /api/tokens`**
(handler `_tokens`). Response fields: `plans_finished`, `tokens_total`,
`tokens_per_plan` (None when 0 plans), `last_tokens`, `last_score`, histogram summaries
(`total_summary`, `per_task_summary`, `score_summary`), and `history` — the newest 20
plan evaluations parsed from the ledger note
`plan <id>: score=<s> tasks=<c>/<p> tokens=<t|None>` (`_token_history`). The endpoint is
strictly read-only: it never writes files.

Dashboard UI: `static/index.html` Token Efficiency panel renders the endpoint with a
dedicated `tok-stats` chip class — deliberately separate from `.hgram-stats` so the
histogram chart tests keep seeing exactly 6 chips.

All Phase 2 bookkeeping is best-effort and additive: plan finishing must never crash
because evaluation/monitoring failed, and missing facts degrade silently.

Tests: `tests/unit/test_plan_finish_eval.py` (11), `tests/test_dashboard_tokens.py` (6),
`tests/unit/test_plan_finish_kanban_isolation.py` (5 — asserts the Kanban bridge is
untouched by the plan-finish path).

## Data locations

- Per-plan artifact: `.docs/<plan_id>/evaluation.json` (workspace `.docs/` directory).
- Ledger: `reports/history/executions.jsonl` (`plan_evaluation` lines).
- Traces with token facts: `reports/traces/*.jsonl`.
- Metrics event file: shared collector events mirrored by `metrics_file.append_event`.

## Verification

Full suite on `0323cec`: **4740 passed, 22 skipped**. CI (`.github/workflows/ci.yml`)
triggers on push to master: package job + ubuntu/windows test matrix.

## Known gaps / follow-ups (not started)

- No live-refresh of the panel in the dashboard poll loop (values update on page reload).
- `reports/plan_evaluations.jsonl` does not exist — evaluations land in
  `.docs/<ts>/evaluation.json` and the executions ledger; if a single aggregated
  report file is wanted, it must be added.
- Prompt/completion split is not aggregated across runs (totals only).
