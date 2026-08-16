# Project Specification: HarnessFix — Trace-Grounded Harness Repair for Agent1

## 1. Goal

Implement a **HarnessFix-style framework** (after arXiv:2606.06324, "From Failed
Trajectories to Reliable LLM Agents: Diagnosing and Repairing Harness Flaws") for
this repository's agent ("Agent1"). The framework must:

1. Capture per-task execution traces from the agent's tool loop.
2. Compile traces into a Harness-aware Trace Intermediate Representation (HTIR)
   with provenance links, control-flow links, and layer responsibility facets.
3. Diagnose failed traces and map each failure to a specific harness layer and
   mechanism (heuristic first; LLM-agent tier optional later).
4. Apply **scoped, code-level repairs** to the harness (never prompt tweaks as
   the primary lever).
5. Close the loop: re-run the benchmark, keep only repairs that improve the
   pass rate without regressing unit tests or security benchmarks.

## 2. Scope

This repo already contains most harness layers (HarnessFix taxonomy):

- **Lifecycle orchestration**: `agent_core/llm/tool_loop.py` — `ToolLoopRunner`
  (budget caps, deadline window, stuck detection, no-mutation guards).
- **Tool interfaces**: `tool_router.py`, `agent_core/tool_dispatcher.py`,
  `agent_core/tool_executor.py`, `agent_core/tool_schemas.py`.
- **Context**: `agent_core/context_management.py`, `chat_history.json`.
- **Governance/security**: `agent_core/security/` (allowlist, sanitizer, path_utils).
- **Verification**: `benchmark.py` (~125-question bank), `benchmarks/security_benchmarks.py`,
  `tests/` (pytest suite).
- **Observability**: `logging_config.py` (JSON logging + correlation IDs) — the
  foundation for trace capture.

**In scope**: new `harnessfix/` package with tracing, HTIR, diagnosis, repair
catalog, and loop orchestration; instrument `ToolLoopRunner` to emit traces;
benchmark integration.

**Out of scope**: changing the base model, tuning system prompts as the primary
lever, modifying the security policy without trace evidence, auto-merging repairs
without a human review gate.

## 3. Requirements

### 3.1 Trace capture (`harnessfix/tracing.py`)
- Instrument `ToolLoopRunner.run()` to emit one JSONL trace per task to
  `reports/traces/{task_id}.jsonl`.
- Event types: `step_start`, `llm_response`, `tool_call`, `tool_result`,
  `tool_error`, `guard_triggered` (deadline/stuck/no-mutation/budget-exhausted
  incl. the injected note text), `loop_end` (outcome: completed |
  budget_exhausted | stuck | no_progress | error).
- Each event tagged with a harness layer facet; reuse `CorrelationIdContext`
  from `agent_core/context_management.py`.
- Trace capture must be non-invasive: zero change to the LLM request payloads,
  and a `--no-trace` / env toggle so existing behavior and tests are unaffected.

### 3.2 HTIR (`harnessfix/htir.py`, `harnessfix/links.py`)
- Pydantic models: `HTIRStep` (index, kind, layer_facet, payload),
  `TraceGraph` (steps + links).
- **Provenance links**: for each tool call / final answer, search backward
  through messages for explicit or semantic reuse of an earlier tool result.
- **Control-flow links**: infer triggering logic from step context and harness
  code (e.g., a step caused by an injected guard note).
- Every failed step maps to exactly one of the 7 layer facets (execution
  environment, tool interface, context, lifecycle, observability, verification,
  governance).

### 3.3 Diagnosis (`harnessfix/diagnose.py`)
- Heuristic tier (must ship first), signature-based:
  - `ValidationError` in dispatcher → tool interface
  - `_DEADLINE_NOTE` / `_STUCK_SYNTHESIS_NOTE` present → lifecycle
  - repeated identical tool-call hash (≥3) → lifecycle (stuck)
  - sanitizer/allowlist rejection or path escape → governance
  - benchmark checker failed with completed loop → verification
  - truncation/token warnings → context
- Output `reports/harnessfix/diagnoses/{task_id}.json`:
  `{root_layer, mechanism, evidence: [htir link ids], confidence, repair_proposal}`.
- Optional LLM tier (later): trace-abstraction agent → diagnosis agent → repair
  agent → reviewer; must not run unless the heuristic tier's precision < 70%.

### 3.4 Scoped repair catalog (`harnessfix/repairs/`)
One module per layer; each repair is a small, testable code change. Seed catalog:
- tool interface: richer tool error messages fed back to the model; auto-retry
  with corrected args on validation failure.
- lifecycle: guard tuning (deadline window, no-mutation limits) driven by trace
  evidence.
- context: history compaction/summarization before the deadline.
- verification: stricter/normalized checkers in `benchmark.py`.
- governance: tighten allowlist only when trace evidence shows a rejected-but-
  needed command.

### 3.5 Closed loop (`harnessfix/loop.py`)
1. Run `benchmark.py` baseline; split pass/fail traces.
2. Diagnose failing traces; group by root layer.
3. Apply repair for the highest-frequency layer.
4. Re-run benchmark + full pytest suite + `benchmarks/security_benchmarks.py`.
5. Accept iff pass-rate improves AND no regressions (test + security gates).
6. Repeat; stop when the layer yields <1pt improvement (diminishing returns).
7. Write `reports/harnessfix/summary.json` with per-layer deltas.

### 3.6 Tests
- `tests/test_harnessfix_tracing.py` — a fake ToolLoopRunner run emits a valid,
  complete trace.
- `tests/test_harnessfix_htir.py` — a synthetic failed trace compiles to the
  expected HTIR nodes/links.
- `tests/test_harnessfix_diagnose.py` — signature classification unit tests.
- `tests/test_harnessfix_loop.py` — the loop rejects a repair that fails tests.

## 4. Constraints

- New files MUST use the `harnessfix/` sub-package prefix; no bare filenames.
- New files max 150 lines each (SRP); prefer modifying existing modules over
  near-duplicates. Do NOT create modules duplicating `security/allowlist.py`,
  `logging_config.py`, or `context_management.py`.
- All code must pass mypy strict; no unbound TypeVars; no circular imports.
- Never shadow Python stdlib module names (`logging`, `json`, `os`, ...).
- Repairs must be backed by ≥1 trace diagnosis; no speculative changes.

## 5. Acceptance criteria

- Running `python -m harnessfix.loop` produces `reports/harnessfix/summary.json`
  with a baseline and at least one accepted repair or a documented "no repair
  met the gates" verdict.
- Full pytest suite still passes after trace instrumentation is enabled.
- Existing `benchmark.py` behavior is unchanged when traces are disabled.
- Trace JSONL files are parseable by `harnessfix/reader.py` back into HTIR.
