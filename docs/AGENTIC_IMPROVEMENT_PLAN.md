# Agentic-AI improvement plan for Agent1

Recorded 2026-08-25 from a full code audit (every claim below was verified
against the current tree). Grouped by theme; [Q] = quick win (hours),
[S] = strategic. Status markers updated as items land. (Tags are plain text
per decision #079 — no emojis in files.)

## A. Wire up the dormant intelligence (highest leverage)

1. [Q] **Inject decisions into chat_nlp** — STATUS: **DONE 2026-08-25**.
   `decisions_as_system_prompt()` guarded `implement_cmd`/`fix_cmd`, but the
   conversational loop never saw `.decisions.json`. Now `chat_nlp` rebuilds
   the leading system message every turn with a constraints block matched on
   the files read this session (`Agent._decision_constraints_block`).2. [S] **Real semantic memory** — `Agent._semantic_index` / `_knowledge_graph`
   / `_working_memory` are loaded, saved, counted, never read into any
   prompt; `src/agent1/core/__init__.py::EmbeddingService.embed_text`
   returns `np.zeros((n, 384))` (stub). Options: real embeddings via LM
   Studio `/v1/embeddings`, or a zero-dep RAG block reusing
   `harnessfix/history.py` matchers to inject recent per-file history.
3. [S] **Make multillm consensus honest** — `ParallelRun.agree()` derives each
   vote from `_looks_negative()` (first-120-chars heuristic). Replace with a
   structured verdict schema requested in the role prompt, then let
   `quorum_reached()` gate something real (e.g. auto-approve an implement
   plan).
4. [S] **Close the meta-policy loop** — `MetaPolicyEvolver.update_weights`,
   `MetricsTracker.record_turn`, `PromptCache` are implemented + tested but
   nothing outside tests instantiates them. Feed `evolution_metrics.score_run()`
   outcomes back so profile selection adapts from trace data.

## B. Tool-loop capability upgrades

5. [Q] **Token-aware chat-history trimming** — `_trim_chat_history` keeps 60
   *messages* regardless of size; a few big reads can blow the context while
   staying under the count cap. Add char/token accounting and replace
   dropped middle turns with a rolling summary message.
6. [Q] **Self-review turn after mutations** — STATUS: **DONE 2026-08-25**.
   After a turn that wrote/edited files, `chat_nlp` prints a `[self-review]`
   nudge listing the changed files (relative paths) with an explicit
   "py_compile verified syntax only" pointer to `tests`/`git diff`
   (`_mutating_files_this_turn` + `_print_self_review_note`).
7. [Q] **Retry-taxonomy asymmetry** — STATUS: **DONE 2026-08-25**.
   `opencode_provider._with_retry` retried HTTP 429/5xx;
   `LMStudioProvider`'s `RetryPolicy` covered only Timeout /
   ConnectionReset / ConnectionRefused, so LM Studio transient HTTP errors
   surfaced as `[Error: ...]` with zero retries. Added
   `TransientHTTPError` raised from `_open_chat` for 429/500/502/503/504 and
   widened the provider's retryable set (incl. `OSError`).
8. [S] **Symbol-level tools** — STATUS: **DONE 2026-08-25**.
   New `agent_core/symbol_intel.py` + two NLP tools:
   `definitions(path)` — every class/function with compact signature and
   line span via pure AST (orient in a big file, then read only the needed
   window); `references(symbol, max_results)` — capped file:line list of
   uses across workspace `.py` files, whole-word/attribute-aware matching
   (`run` does not hit `run_interactive`), oversized files skipped.
   Both read-only → added to `PLAN_MODE_TOOLS`; system prompt steers the
   model to prefer them over grep+read paging.
9. **Stream final answers** — `chat_stream` exists (with model-eviction
   recovery) but `chat_nlp` blocks silently through long turns.

## C. Complete the self-improvement loop

10. [S] **Grow the repair catalog** — `harnessfix/repairs/` has exactly one
    repair (`tool_interface.py`). Each diagnosed mechanism in `diagnose.py`
    maps naturally to a catalog repair (truncation-pressure → auto-compaction
    note; stuck-repeat → tool-hint table; abandonment-after-mutation → resume
    protocol). Apply/revert/collision-guard scaffolding already built.
11. **LLM-tier diagnosis** — `diagnose.py` promises an LLM fallback tier when
    heuristic precision <70%; measure precision from labeled `review` data
    before building.
12. [S] **Agentic benchmark** — `benchmark.py` measures Q&A accuracy only;
    add SWE-task scenarios (seeded-broken fixture repos, score = tests green
    after headless agent work) wired into `gates.run_benchmark_gate`.

## D. Orchestration & UX

13. [S] **Planner→executor split** — `src/agent1/orchestration/task_scheduler.py::DependencyGraph`
    (get_ready_tasks/topological_order) is unused by the main agent; natural
    fit for plan-mode output → scheduled subtask execution.
14. [Q] **Plan-mode handoff** — plan-mode answers evaporate as terminal text;
    persist them to `.docs/<ts>/plan_proposed.md` and offer a one-command
    transition into build/implement.
15. [Q] **multillm synthesis flag** — results print side-by-side then stop.
    Add `--synthesize` (merge answers through one model) and print per-model
    `ResponseMetrics` token/latency in the REPL summary.
16. **Unify shell policy** — NLP `run` uses a destructive-*blocklist* +
    `shell=True`; `tool_router.py`/`tools/shell_ops.py` use the
    `security/allowlist.py` allowlist. Converge on allowlist-with-fallbacks.

## E. Code health

17. [Q] **Dead duplicate modules** — `agent_core/tool_executor.py`,
    `secure_file_retriever.py`, `tools/file_ops.py` are not imported by the
    live path (phantom-module bait, invariant #5). Wire or archive.
18. [Q] **Fix duplicated imports / enum drift** — `db_io.py` has
    `from typing import Any` seven times; `llm_types.ProfileType`
    ("fast_codegen") vs `config.ProfileType` ("fast-codegen") enums diverge.
19. **Doc-commanding conventions** — STATUS: **DONE 2026-08-25**.
    `_warn_uncommitted` runs on every REPL shutdown path (quit / stdin end /
    EOF): lists up to 5 uncommitted paths with the invariant-#4 reminder;
    silent on clean repos and outside git; never blocks exit.

## Sequencing

Weekend of quick wins: ~~#1~~ → ~~#14~~ → ~~#6~~ → ~~#7~~ — **all quick
wins from the original sequencing are DONE** (plus #5, #15, #18).
Next substantive feature: #8 or #10. Strategic horizon: #12 / #13 / #2.
Remaining quick wins: none — remaining items are [S]-scale.

## Progress log

- 2026-08-25 — **#8 DONE**: symbol-level tools. `agent_core/symbol_intel.py`
  (pure-AST, stdlib-only) + `definitions`/`references` NLP tools wired into
  the schema set, dispatch table and plan-mode read-only set; system prompt
  steers toward them for "where is X used" / large-file orientation.
  Tests: `tests/test_symbol_intel.py` (17).
- 2026-08-25 — **#6 DONE**: post-mutation self-review note. After a turn
  whose write/edit results carry py_compile verification lines,
  `chat_nlp` prints one `[self-review]` nudge listing the changed files
  (`_mutating_files_this_turn` + `_print_self_review_note`) — py_compile
  proves syntax only, so the user is pointed at `tests`/`git diff`.
- 2026-08-25 — **#19 DONE**: `_warn_uncommitted` runs on every REPL
  shutdown path (quit / stdin end / EOF): lists up to 5 uncommitted paths
  with an invariant-#4 reminder when `git status --porcelain` is non-empty;
  silent on clean repos and outside git; never blocks exit.
- 2026-08-25 — **#1 DONE**: decisions block injected into the chat_nlp
  system message (`Agent._decision_constraints_block`, rebuilt per turn).
- 2026-08-25 — **#7 DONE**: `TransientHTTPError` + widened `RetryPolicy`
  default retryable set; LM Studio now retries 429/5xx like opencode.
  Tests: `tests/test_llm_retry_policy.py` (10).
- 2026-08-25 — **#14 DONE**: plan-mode final answers persisted to
  `.docs/<ts>/plan_proposed.md` (`Agent._persist_plan_answer`, best-effort).
- 2026-08-25 — **#5 DONE**: `_trim_chat_history` gains a char budget
  (`_HISTORY_CHAR_BUDGET`, 75k chars) alongside the message-count cap;
  oldest-first contiguous trim + compaction note; assistant/tool pairs are
  never split.
- 2026-08-25 — **#15 DONE**: `multillm --synthesize` merges the parallel
  answers through one extra LLM call (skips <2 usable answers, swallows
  provider errors).
- 2026-08-25 — **#18 DONE**: `db_io.py` duplicate imports collapsed to one;
  `llm/config.py` now re-exports the canonical `llm_types.ProfileType`
  (the divergent "fast-codegen" enum is gone).
- Tests for all of the above: `tests/test_quickwins_2026_08_25.py` (14).
