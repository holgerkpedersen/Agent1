# Pre-#050 Trace Labels — Historical Assessment of the 12 Failed Traces

Scope: the 12 FAILED traces recorded before the metadata stamping (decision
#050). They carry no prompt/model/profile, so they are excluded from the
`review` ledger — this document is their review. Each verdict is grounded in
the raw trace (`reports/traces/<task>.jsonl`): event counts, tool-call
histograms, guard notes, `llm_response` texts, `loop_end` records.

Date of assessment: 2026-08-18 (all traces were re-read for this document).

## Summary

| Task (prefix) | Disposition | Root cause (per trace evidence) |
| --- | --- | --- |
| `0f7793…` | **bug** | 50 silent read-only calls circling through `agent.py` pages (offset 49500 → 49000); `no_mutation` guard fired; no final answer |
| `17131c…` | **noise** | Interrupted mid-exploration (no `loop_end`); 25 reads of `agent.py`; no mutation, no error, no guard |
| `536848…` | **bug** | Real work in progress (1 write + 8 edits in `workflow_cmd.py`), then 3× identical `search` call tripped the `stuck` guard; run ended without answer |
| `9acf2d…` | **bug** | Real work done (2 writes, 34 tests passing), then a 50-call read-only verification phase tripped `no_mutation` before any final answer |
| `9c89a2…` | **noise** | Interrupted mid-verification: last event is a `tests` call with no result and no `loop_end` (4 edits had been applied) |
| `b5b264…` | **bug** | Circled `interface.py` in 3-line read steps (offset 145 → 148); `no_mutation` guard fired; no mutation, no answer |
| `demo-http500` | **noise** | Spec fixture; delivered a final answer ("the tool failed twice"); the `ValueError: boom` is the fixture's scripted tool error |
| `demo-stuck` | **noise** | Spec fixture for the stuck guard; 3× identical `search`; no final answer |
| `e0b03b…` | **noise** | Provider/environment error ~30 ms after a tool result (`loop_end outcome=error`); exploration only, no mutation |
| `f00b3088…` | **noise** | Interrupted mid-scan after writing 2 helper scripts; no `loop_end`; no error/guard |
| `f1b4320b…` | **noise** | Provider/environment error ~4 s after a tool result (`loop_end outcome=error`); exploration only, no mutation |
| `fc5199…` | **noise** | Interrupted mid-exploration (no `loop_end`); 21 exploration calls, no mutation |

**Totals: 4 bugs, 8 noise, 0 regressions, 0 ok.**

## The bugs (what the improvement loop should learn from)

### `0f77934619d64e2e8fded087168169ec` — bug
- 204 events, 50 tool calls: `read`×42, `search`×6, `list_files`×2.
- All 50 `llm_response` events have **empty `text`** — the model emitted tool
  calls with no narration for the entire run.
- Reads paged through `agent.py` backwards (offset 49500 → 49000) — circling,
  no direction change.
- `no_mutation` guard at iteration 30 (warning) and 50 (stop); `loop_end`
  `outcome=no_progress`.
- No final answer. Diagnosis `lifecycle/no_mutation` is correct.
- Lesson: exploration loop without narration or mutation; guard worked as
  designed.

### `536848aebf354bf7b16b2e675e33d3e1` — bug
- 358 events, 89 tool calls: `read`×67, `edit`×8, `search`×8, `run`×2,
  `list_files`×3, `write`×1. Substantive implementation work on
  `workflow_cmd.py` (8 edits applied, no tool errors).
- After the edits the model tried to verify a renamed function and issued the
  **same `search` call 3× in a row** (`_analysis_flag_gate|_verify_or_repair…`).
- `stuck` guard fired at iteration 88 (twice — once per guard stage); the
  trace then ends **without `loop_end`**: the run was interrupted right after
  the guard.
- No final answer. Diagnosis `lifecycle/stuck` is correct.
- Lesson: verification-phase repetition after real edits; the stuck guard
  caught it, but the run never got to answer.

### `9acf2d867cfb427d89b3500baaebd25d` — bug
- 321 events, 87 tool calls: `read`×61, `tests`×6, `run`×4, `git`×3,
  `write`×2. Two scripts written (iterations 8, 30), `run`/`tests` show "All
  34 harnessfix tests pass", then git status + `CHANGES.md` research.
- The model then made **50+ read-only calls** (CHANGES.md page-by-page) —
  `no_mutation` guard fired at iteration 72 ("50 tool calls without modifying
  any file"); `loop_end outcome=no_progress`.
- No final answer despite completed work. Diagnosis is correct, but the guard
  message ("without modifying any file") reads oddly given 2 writes happened
  earlier — the guard counts calls since the last mutation.
- Lesson: the agent did the work but never stopped to answer; the 50-call
  post-write verification phase is the behavior to fix.

### `b5b264198553417d869ae2053ffb14f2` — bug
- 158 events, 39 tool calls, all read-only (`read`×35, `list_files`×3,
  `search`×1). Monopoly mortgage-button layout task.
- Reads stepped through `C:\Dev\Monopoly/src/ui/interface.py` in **3-line
  increments** (offset 145 → 148) — the circling signature.
- Only one `llm_response` with text in the entire run — the opening line,
  repeated verbatim as the "last" text.
- `no_mutation` guard at iteration 39 ("30 calls without discovering anything
  new"); trace ends without `loop_end` (interrupted after the guard).
- No mutation, no answer. Diagnosis correct.
- Lesson: same silent-circling pattern as `0f7793…` — worth a shared lesson
  entry.

## The noise (do not send to the improvement loop)

### Interrupted runs (no `loop_end`, no error, no guard) — 4× noise
`17131c…` (25 reads of `agent.py`, budget 135 remaining at cut-off),
`9c89a2…` (4 edits applied, last event an unanswered `tests` call),
`f00b3088…` (2 helper scripts written, mid-scan), `fc5199…` (21 exploration
calls, Danish user-manual task).
Pattern: trace stops immediately after a `tool_call` or `tool_result` with no
`loop_end` — consistent with external process death (LM Studio crash/kill,
machine sleep), not agent behavior. No damage: no file was left half-written
by the loop.

### Provider-error endings (`loop_end outcome=error`, reason=answer) — 2× noise
`e0b03b…` (loop_end 0.03 s after the last tool result; 7 calls, exploration
only) and `f1b4320b…` (loop_end ~4 s after the last tool result; 6 searches).
Pattern: no `tool_error` event, no guard, no final `llm_response` — the loop
died on the LLM call itself (provider/connection failure). No mutation, so
nothing to inspect.

### Spec fixtures — 2× noise
`demo-http500` (delivered a 21-char final answer; the scripted `ValueError:
boom` is fixture behavior) and `demo-stuck` (3× identical `search "sym"`,
stuck guard, no answer — the fixture's purpose).

## Evidence method (reproducible)

Each verdict comes from a full re-read of the raw trace. The quick checks:

1. `kinds` set — `loop_end` present? `tool_error` present? `guard_triggered`?
2. Tool histogram + longest identical-call run — circling vs. progress.
3. First and last `llm_response` with non-empty `text` — intent and final
   answer.
4. Last 8 events — how the run actually ended (or where it was cut off).
5. For traces with writes/edits: every `tool_call`/`tool_result` pair for
   `write`/`edit` (result `err=None` confirms the mutation applied).