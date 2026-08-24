# HarnessFix Implementation Plan — Applying Trace-Grounded Harness Repair to Agent1

Based on: *"From Failed Trajectories to Reliable LLM Agents: Diagnosing and Repairing Harness Flaws"* (arXiv:2606.06324, Chen/Wang et al., CAS & Tianjin University). Reported gains: +6.3% to +18.4% over initial harnesses.

## 0. What HarnessFix does (the loop)

```
failed trajectories → HTIR (trace IR w/ links) → diagnosis (which harness layer?)
→ scoped code-level repair → re-run benchmark → keep if improved (no regression)
```

Core principle: **repairs are grounded in trace evidence and scoped to one harness
mechanism** — NOT broad prompt tweaks. The paper uses four cooperating LLM agents
(trace-abstraction, diagnosis, repair), but a heuristic-first MVP works and costs less.

## 1. Harness layers in this repo (HarnessFix taxonomy mapping)

| Layer | Files | Example mechanism to repair |
|---|---|---|
| Execution environment | `agent_core/file_system.py`, `subprocess_utils.py` | sandbox rules, path handling |
| Tool interfaces | `tool_router.py`, `tool_dispatcher.py`, `tool_schemas.py` | schemas, error messages, retry-on-validation |
| Context | `agent_core/context_management.py`, chat history | truncation, compaction, injection order |
| Lifecycle orchestration | `agent_core/llm/tool_loop.py` (`ToolLoopRunner`) | budget guards, stuck detection, deadline notes |
| Observability | `logging_config.py` (JSON + correlation IDs — already good) | trace emission (Phase 0) |
| Verification | `benchmark.py`, `benchmarks/security_benchmarks.py`, `tests/` | checkers, eval gates |
| Governance | `agent_core/security/` (allowlist, sanitizer) | policy tightening from path-escape evidence |

## 2. Phase 0 — Trace capture (trace grounding)

Instrument `ToolLoopRunner` to emit one JSONL trace per task to `reports/traces/{task_id}.jsonl`.
Reuse existing `CorrelationIdContext` / `JsonFormatter`. Event types, each tagged with a layer facet:

- `step_start` (iteration, budget remaining, correlation_id)
- `llm_response` (message text, tool_call requests, truncated)
- `tool_call` / `tool_result` (name, args-hash, duration, result truncated to a cap)
- `tool_error` (exception type, message, traceback)
- `guard_triggered` (deadline / stuck / no-mutation / budget-exhausted + the note text injected)
- `loop_end` (outcome: completed | budget_exhausted | stuck | error, final answer)

Deliverable: `harnessfix/tracing.py` (writer) + `harnessfix/reader.py`.
Test: `tests/test_harnessfix_tracing.py` — a fake `ToolLoopRunner` run emits a valid, complete trace.

## 3. Phase 1 — HTIR compiler (Harness-aware Trace Intermediate Representation)

`harnessfix/htir.py`: parse trace JSONL → HTIR graph. Pydantic models:

- `HTIRStep` (index, kind, layer_facet, payload)
- **Provenance links**: for each tool call / answer, search backward through messages
  for explicit or semantic reuse of an earlier tool result (which file content does
  this edit depend on?). Matches the paper's "search backward through messages".
- **Control-flow links**: infer triggering logic from step context and harness code
  (e.g., step N was caused by the `_DEADLINE_NOTE` injected by the loop guard).
- **Layer responsibility facets**: each failed step maps to exactly one of the 7 layers.

Deliverable: `harnessfix/htir.py` + `harnessfix/links.py` (provenance/control-flow inference).
Test: feed a synthetic failed trace, assert the expected HTIR nodes and links.

## 4. Phase 2 — Diagnosis

`harnessfix/diagnose.py`. Two tiers:

1. **Heuristic tier (MVP)**: classify failure by signature —
   - `ValidationError` in dispatcher → tool interface
   - `_DEADLINE_NOTE` / `_STUCK_SYNTHESIS_NOTE` present → lifecycle
   - repeated identical tool_call hash ×3 → lifecycle (stuck)
   - path escape / sanitizer rejection → governance
   - benchmark checker failed, loop completed → verification
   - truncation warnings / token cap → context
2. **LLM tier (matches the paper)**: 4 cooperating agents (can be 1 model with roles):
   trace-abstraction agent (builds HTIR) → diagnosis agent (root layer + mechanism +
   evidence links) → repair agent (proposal) → reviewer (scope check: one mechanism only).

Output: `reports/harnessfix/diagnoses/{task_id}.json`:
`{root_layer, mechanism, evidence: [htir link ids], confidence, repair_proposal}`.

Deliverable: `harnessfix/diagnose.py`, `harnessfix/agents.py`.

## 5. Phase 3 — Scoped repair catalog

`harnessfix/repairs/` — one module per layer, each repair is a small code change + a
regression test. Catalog to start with:

| Layer | Candidate repairs |
|---|---|
| tool interface | richer error messages fed back to model; better parameter descriptions; auto-retry with corrected args on validation failure |
| lifecycle | tune `deadline_window` / `no_mutation_limit`; add guard for "reads-only spiral"; earlier stuck detection on arg-hash repeats |
| context | history compaction/summarization before deadline; drop stale tool results from context |
| verification | stricter/normalized checkers in `benchmark.py` (e.g., number parsing for "0.10" vs "10 cents") |
| governance | tighten allowlist based on evidence of rejected-but-needed commands (evidence-driven, not vibes) |

Rule: a repair is accepted only if (a) it fixes a diagnosed failure in ≥1 trace,
(b) `tests/` still passes, (c) it doesn't regress other categories on the benchmark.

## 6. Phase 4 — The HarnessFix loop (closed loop)

`harnessfix/loop.py`:

1. Run `benchmark.py` baseline → split pass/fail traces.
2. Phase 1–2 on failing traces → diagnosis report per task.
3. Group by root_layer; pick the highest-frequency layer; apply catalog repair.
4. Re-run benchmark + `tests/` + `benchmarks/security_benchmarks.py`.
5. Accept iff pass-rate improves and no regression (test gate + security gate).
6. Repeat; stop when a layer yields <1pt improvement on a full run (diminishing returns).
7. Log per-layer deltas to `reports/harnessfix/summary.json`.

Guardrails:
- Human review gate before merging any auto-generated repair.
- Never change the base model / prompts as the primary lever — repairs target harness code.
- Keep traces as ground truth; HTIR must be reproducible from a trace alone.

## 7. Suggested build order (incremental)

1. **Phase 0 only** → run benchmark once → you now have a trace corpus (the missing asset today).
2. Phase 1 HTIR + Phase 2 heuristic diagnosis on the corpus → first `diagnoses.json`.
3. Phase 3 repair #1 in the highest-frequency layer (likely `tool interface` or `lifecycle`)
   + Phase 4 loop for that single layer.
4. Add the LLM-tier diagnosis agents (Phase 2 tier 2) only if heuristic precision < ~70%.

## 8. Risks / notes

- **Overfitting the benchmark**: keep `benchmarks/security_benchmarks.py` and unit tests
  as an orthogonal gate; spot-check repairs on real tasks (e.g., `USAGE.md` walkthroughs).
- **Cost**: LLM-tier agents cost tokens; use the small/fast provider profile for diagnosis,
  the strong one for repair proposals only.
- **Determinism**: LLM behavior varies; always report deltas over ≥2 runs, not 1.
- Expected outcome in the paper was +6.3–18.4%; treat anything > +5pt on your benchmark
  as a realistic success target for the first closed loop.
