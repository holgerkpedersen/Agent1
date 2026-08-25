## 2026-08-25 - feat: delegate/delegate_batch tools + issue router (phases 2b-4)

**Change**: agent_core/tool_schemas.py (`delegate(role, task)` + `delegate_batch(roles[], task)` schemas), agent.py (`_nlp_delegate` spawns a role subagent per call — isolated history, whitelist executor, 600s `asyncio.wait_for` timeout surfacing as an error report instead of holding the turn; concurrency cap `_MAX_ACTIVE_SUBAGENTS=3`; termination reason propagated into the report as "stopped early: <cap|stuck|no_progress>"; `_nlp_delegate_batch` fans one task out to ≤3 roles in parallel via `asyncio.gather`, dedupes roles, merges per-role reports; spawn failures caught so a broken child never kills the parent turn; `_delegating` flag makes child reads skip the PARENT read-loop streak counter — a child's steering note would otherwise leak into its own tool result), agent_core/modes.py (`delegate` added to `PLAN_MODE_TOOLS` — safe because the parent's mode caps every child read-only; plan-mode steering text updated), agent_core/subagent.py (`last_termination_reason` exposed from the child's ToolLoopRunner), agent_core/issue_router.py (new: pure keyword-scoring issue→role classifier, planner fallback on zero hits, `routing_table()` for help output), agent_core/commands/subagent_cmd.py (`subagent route <issue>` command; `--role auto` hint), USAGE.md (new Subagents section).

**Reason**: The main chat loop could not use the roster from 3d539a8 by itself. Delegation keeps the parent context small (the 2026-08-25 stall post-mortem: long contexts kill this local model) while guards bound every child: turn cap, loop-level stuck/no-progress synthesis, wall-clock timeout, concurrency cap.

**Files**: agent.py, agent_core/tool_schemas.py, agent_core/modes.py, agent_core/subagent.py, agent_core/issue_router.py (new), agent_core/commands/subagent_cmd.py, USAGE.md, tests/test_delegate_and_router.py (new, 24 tests: router classification/fallback/table; delegate validation/cap/plan-mode capping/timeout/spawn-failure/termination-reason/read-streak isolation; batch validation/dedupe-cap/isolated parallel children).

## 2026-08-25 - feat: role-based subagents (nine issue-class roles, tool whitelists, mode capping)

**Change**: agent_core/subagent_roles.py (new: frozen `SubagentRole` registry — planner/implementer/tester/debugger/reviewer/integrator/researcher/security/documenter — each with persona prompt, session mode, explicit NLP-tool whitelist and turn cap; `validate_roles()` proves every tool exists in `NLP_TOOL_NAMES`, read-only roles stay inside `PLAN_MODE_TOOLS` + plan mode, mutating roles run in build), agent_core/subagent.py (`SubAgent(parent, name, workspace, role)`: persona system message, per-role bounded `ToolLoopRunner` loop whose executor enforces the whitelist BEFORE the parent `_execute_tool_call` choke point; parent plan mode caps every child to read-only so delegation can never become a plan-mode escape hatch; turn cap refuses further work instead of burning tokens; roleless subagents behave exactly as before), agent_core/commands/subagent_cmd.py (rewritten as a real async `Command`: the old class was dead code AND returned an un-awaited coroutine from `run` — now registered in `_register_commands`, with `roles/create/run/list/summary/reset` handlers), agent.py (`spawn_subagent(role=...)`, command registration).

**Reason**: A coding-implementation project hits distinct issue classes (triage, feature work, red→green testing, root-cause debugging, pre-commit review, git discipline, external research, security audit, docs drift); one role per class keeps each child's context small and single-purpose. The 2026-08-25 stall post-mortem showed long contexts kill this local model — short-lived isolated children beat one bloated swarm. Also fixes: `SubAgentCommand.execute` never awaited `respond()` (returned a coroutine object), and nothing imported the class at all.

**Files**: agent_core/subagent_roles.py (new), agent_core/subagent.py, agent_core/commands/subagent_cmd.py, agent.py, tests/test_subagent_roles.py (new, 19 tests: registry consistency vs NLP_TOOL_NAMES/PLAN_MODE_TOOLS, unknown-role rejection, parent-mode capping, whitelist-before-choke-point ordering, scripted tool_calls executed through a real ToolLoopRunner against a tmp workspace, turn cap, REPL registration/await/list/reset roundtrip, roleless backward compat).

## 2026-08-25 - feat: real binned distribution chart in the Histogram view

**Change**: agent_core/monitoring/metrics_collector.py (`MetricsCollector.all_histogram_samples()` returns defensive copies of every metric's raw samples under lock), agent_core/monitoring/dashboard_api.py (`GET /api/histograms` now also returns `"samples": {metric: [values]}` alongside the legacy summary; without `?name=` the summary is computed for an empty name and is all zeros - documented contract), static/index.html (Histogram view upgraded from a numeric table to a real binned distribution chart: metric picker, CSS bar chart with 4-24 bins via the square-root rule, hover tooltips per bin (range + sample count), samples/mean/p50/p95/min/max stat chips; bars styled from ttheme variables so dark/light/monochrome themes follow automatically; pure DOM/CSS, no new vendored libraries).

**Reason**: The Histogram view showed numbers where a shape was needed - p50/p95/mean/max hide whether latency is tight or long-tailed, which is exactly what a distribution shows. Verification caught two real bugs before they shipped: bars collapsed to 0 height because percentage heights inside `align-items:flex-end` flex columns have no definite parent (fixed with `align-items:stretch` + explicit column height; a regression test now pins the stretch rule), and an initially wrong test expectation about p95 interpolation (n=10 -> position 8.55, linear interpolation between 0.31 and 1.40 gives 0.91s - the code was right, the expectation was not).

**Files**: agent_core/monitoring/dashboard_api.py, agent_core/monitoring/metrics_collector.py, static/index.html, tests/test_dashboard_histogram_chart.py (new, 4 tests: API contract incl. ?name= behavior, copy-safety of returned samples, chart scaffolding + align-items:stretch guard, percentile() extracted from the page and executed under node against reference values).

## 2026-08-25 - fix: Command Summary view rendered blank - view panels were nested inside a dashboard row

**Change**: static/index.html (moved the command-summary view panel out of a dashboard row so all view panels are siblings; one-line structural fix).

**Reason**: A panel nested inside another layout container broke the show/hide contract: opening the view toggled a class on panels expected to be siblings, leaving Command Summary invisible. Regression tests pin the structure: every view has exactly one panel, panels are siblings (not nested), and no panel is a descendant of another panel.

**Files**: static/index.html, tests/test_dashboard_view_panels.py (new, 3 tests).

## 2026-08-25 - feat: 3D rotating sidebar logo (pure CSS cube, mascot faces + amber sides)

**Change**: partials/_head.html (stylesheet include), partials/_sidebar.html (cube markup replacing the flat logo img), static/assets/logo3d.css (pure-CSS rotating cube: six faces built from transforms, mascot face images on front/back, amber side panels; animation pauses on hover).

**Reason**: Visual identity upgrade for the dashboard sidebar with zero JavaScript - the whole effect is CSS transforms and keyframes, so it works offline and adds no runtime logic to maintain.

**Files**: partials/_head.html, partials/_sidebar.html, static/assets/logo3d.css (new).

## 2026-08-25 - feat: shared metrics event file so --serve dashboard shows cross-process activity

**Change**: agent_core/monitoring/metrics_file.py (new: JSONL event file as the cross-process metrics bus - append-only records, reader folds them into a MetricsCollector snapshot), agent.py (REPL emits metric events for its activity), agent_core/monitoring/dashboard_api.py (snapshot endpoints merge in-file events so a --serve dashboard reflects REPL sessions running in other processes), .gitignore (runtime event file excluded).

**Reason**: The dashboard previously only saw metrics from its own process, so running it via --serve while working in a separate REPL shell showed an idle agent. The shared event file decouples producers (any number of REPL processes) from the consumer (dashboard server) without adding a network protocol.

**Files**: agent_core/monitoring/metrics_file.py (new), agent.py, agent_core/monitoring/dashboard_api.py, .gitignore, tests/test_demo_data.py (extended).

## 2026-08-25 - feat: feed tool executions + top-level git command into dashboard metrics

**Change**: agent_core/tool_dispatcher.py (tool executions now emit counter/histogram metric events), agent_core/commands/git_cmd.py (top-level git command records elapsed-time histograms and outcome counters), agent.py (wiring for NLP-path tool calls).

**Reason**: The dashboard's tool/git widgets had no data source on the live path - only demo data produced numbers. Instrumenting the dispatcher covers every tool call in one place; git gets its own instrumentation because users invoke it as a top-level command outside the generic dispatcher path.

**Files**: agent.py, agent_core/commands/git_cmd.py, agent_core/tool_dispatcher.py.

## 2026-08-25 - feat: second repair catalog entry + diagnosis FP fix; archive dead modules

**Change**: harnessfix/repairs/stuck_repeat.py (new repair `stuck-repeat-tool-hints`, lifecycle layer: the SECOND consecutive identical tool call now carries concrete alternatives - `_REPEAT_HINTS` for read/search/list_files/run plus a default - appended after the pinned prefix "NOTE: This exact call has now been executed"; `apply()` inserts the hint table above `_PATH_MISS_PREFIXES` and routes the old second-strike suffix through `_REPEAT_HINTS.get(tool_name, _REPEAT_HINT_DEFAULT)`, `revert()` restores byte-identical source, a distinctive sentinel line turns half-applied trees into loud errors instead of silent no-ops), harnessfix/repairs/__init__.py (CATALOG now holds two repairs across two layers; lifecycle facet exported), harnessfix/diagnose.py (`_find_signature` now excludes `llm_response` free-text for ALL signatures - same root cause as the existing `tool_result.text` exclusion - and the context-layer truncation signature matches system fields (`guard_triggered.note`) instead of `llm_response.text`), agent_core/tool_executor.py + agent_core/secure_file_retriever.py deleted (zero references anywhere; the secure retriever was superseded by the live `agent_core/file_context_retriever.py` that agent.py imports; `tools/file_ops.py` + `tools/shell_ops.py` stay - they ARE consumed by benchmarks/security_benchmarks.py and tests/test_security_hardening.py), docs/AGENTIC_IMPROVEMENT_PLAN.md (#10 marked in progress, #17 partially done, sequencing + progress log updated), tests/test_harnessfix_diagnose.py (`test_truncation_warning_maps_to_context` replaced: it pinned exactly the false-positive behavior; new pair of tests pins the corrected contract).

**Reason**: Improvement-plan #10 ("grow the repair catalog"), chosen from evidence rather than guesswork: diagnosing the full trace corpus (254 traces -> 27 failed) exposed that ALL five "history truncation / token limit pressure" diagnoses were false positives - the signature matched `llm_response.text`, i.e. the model QUOTING the tracer's storage marker "...[truncated N chars]" (`tracing.truncate`) inside its own chat output, never a real context event. Fixing the FP re-classification surfaced 2 additional real stuck-cycles (2 -> 4), so 4 of the 27 remaining failures map directly to the new repair. Design choice: prevention beats decoration - at strike TWO the model still has budget, so concrete alternatives can avoid the fatal third strike entirely, while the third-strike stop guarantee stays intact (verified through the real ToolLoopRunner: forced synthesis still fires on the third repeat with only 4 LLM calls under a cap of 10). #17 scope correction recorded: the plan item wrongly listed tools/file_ops.py as dead.

**Files**: harnessfix/repairs/stuck_repeat.py (new), harnessfix/repairs/__init__.py, harnessfix/diagnose.py, agent_core/tool_executor.py (deleted), agent_core/secure_file_retriever.py (deleted), docs/AGENTIC_IMPROVEMENT_PLAN.md, tests/test_repairs_stuck_repeat.py (new, 12 tests: apply/revert roundtrip is byte-identical, double-apply no-op, anchor-missing raises, collision-guard surface clean, runtime prevention AND third-strike guarantee verified in a fresh interpreter against the applied source, closed-loop e2e proposes/accepts the new repair), tests/test_harnessfix_diagnose.py (17 tests).

## 2026-08-25 - refactor: simplify for comprehension (chat_nlp decomposed into 4 phases, stale backup copies purged)

**Change**: agent.py (Agent.chat_nlp reduced from a 285-line monolith to a ~30-line delegation over four single-purpose helpers: _refresh_system_message rebuilds history[0] from _SYSTEM_PROMPT + dynamic blocks; _append_user_turn handles multimodal input, plan-mode steering note and the _turn_start_index boundary; _run_chained_tool_loop owns llm_chat_fn, tracing, ToolLoopRunner execution, auto-continue chaining, repeated-answer detection and tagged-note stripping - returning (final_text, final_messages, llm_error, loop); _finish_turn trims/persists history and prints the outcome. Behaviour is unchanged - every guard, message string and decision-id comment moved with its code), tests/test_agent_dry_refactor.py (test_chat_nlp_uses_system_prompt_constant now asserts the constant is used by _refresh_system_message AND that chat_nlp delegates to it - same invariant, new code path), scripts/audit_invariants.py (invariant 5 message clarified: backups/ must exist but stay EMPTY between runs), backups/ (38 stale .py copies of live files deleted - implement_cmd writes fresh pre-run copies anyway and git history preserves old versions; the directory itself is kept because the auditor requires it).

**Reason**: User asked to simplify the codebase for better understanding. Verified first that src/agent1/ is NOT dead (agent.py imports MetricsCollector/AlertSystem/DashboardAPIServer from it; 10+ test files use it) despite packaging excluding it - an earlier plan to delete it was dropped after checking imports. The genuinely confusing parts were the chat_nlp monolith (6 interleaved concerns in one closure-heavy method) and 38 committed backup copies shadowing live files in every search/read. agent.py now passes mypy with ZERO errors (was 5 at baseline); the documented repo baseline drops to 22 errors in 6 untouched files.

**Files**: agent.py, tests/test_agent_dry_refactor.py, scripts/audit_invariants.py, backups/ (cleaned).

## 2026-08-25 - fix: agent.py robustness batch (atomic persistence, argument caps, tagged continue-notes, provider profile contract)

**Change**: agent_core/constants.py (CHAT_HISTORY_TMP_PATH / AGENT_MEMORY_TMP_PATH sidecar paths; LOOP_NOTE_TAG_KEY marker for loop-injected user notes), agent.py (_save_chat_history/_save_memory now write tmp + os.replace instead of truncating in place; _load_chat_history/_load_memory route through _read_json_quarantining, which renames a corrupt file to <path>.bad-<timestamp> before falling back to empty so the next save cannot clobber the evidence; _nlp_run caps model-supplied timeout at _MAX_RUN_TIMEOUT_S=600 and _nlp_read caps limit at _MAX_READ_LINES=500; the auto-continue note is appended with the LOOP_NOTE_TAG_KEY tag and stripped by TAG not by content match, so a user whose prompt is byte-identical to the note text keeps their message; the plan-mode banner respects QUIET display mode like every other status print), agent_core/llm/provider.py (LLMProvider protocol gains apply_profile(name, temperature, max_tokens) and corrects _profile_name to str | None), agent_core/llm/lmstudio.py + opencode_provider.py (apply_profile implemented; sanitize_message_roles strips the internal tag at the payload boundary because it copies extra message keys into the HTTP request), agent_core/commands/model_cmd.py (profile use goes through apply_profile), AGENTS.md (mypy baseline note 36 -> 22: agent.py 5 errors and model_cmd.py 4 errors are gone as a side effect of removing the _profile_name type mismatch).

**Reason**: Brainstorm follow-up on agent.py effectiveness. Four real defects fixed: (1) a crash mid-save corrupted chat_history.json/agent_memory.json AND the loader silently returned [] on JSONDecodeError, destroying the conversation without a trace - atomic writes prevent corruption and quarantine preserves it if it happens anyway; (2) the tool loop accepted unbounded model-supplied timeout (could stall a turn for hours) and read limit (one call could pull an entire large file into context); (3) continue-note stripping matched message CONTENT, so a genuine user prompt equal to the note text was silently dropped from history - observed live when this very note arrived as a user-side prompt; (4) LLMClient poked _provider._profile_name/temperature/max_tokens directly, which was also the root cause of 9 of the documented mypy baseline errors. The payload-boundary strip matters because sanitize_message_roles spreads unknown keys into provider requests where strict gateways can reject them.

**Files**: agent.py, agent_core/constants.py, agent_core/llm/provider.py, agent_core/llm/lmstudio.py, agent_core/llm/opencode_provider.py, agent_core/commands/model_cmd.py, tests/test_agent_robustness.py (new, 17 tests: atomic saves leave no tmp sidecar, corrupt files quarantined with original bytes preserved, timeout/read caps, tagged-note survives persist while injected notes do not, QUIET suppresses plan banner, apply_profile contract on both providers plus LLMClient restore path, tag stripped at payload boundary).

## 2026-08-25 - feat: agentic quick-win batch (decisions in chat, transient-HTTP retries, plan handoff, char-budget trimming, multillm --synthesize)

**Change**: agent.py (Agent._decision_constraints_block + per-turn system-message rebuild with _strip_dynamic_system_blocks; _persist_plan_answer writes .docs/<ts>/plan_proposed.md in plan mode; _trim_chat_history gains a 75k-char budget alongside the 60-message cap, oldest-first contiguous trim + compaction note, orphan tool messages dropped before AND after the cut), agent_core/llm/retry.py (TRANSIENT_HTTP_STATUSES, TransientHTTPError, default retryable set widened with OSError), agent_core/llm/lmstudio.py (_open_chat raises TransientHTTPError for 429/500/502/503/504 so the RetryPolicy backs off - previously zero retries on LM Studio HTTP errors while opencode retried the same blips), agent_core/llm/config.py (re-exports canonical llm_types.ProfileType; divergent fast-codegen enum removed), agent_core/llm/db_io.py (7 duplicate imports collapsed to 1), agent_core/commands/multillm_cmd.py (--synthesize merges parallel answers through one extra LLM call; skips <2 usable answers; swallows provider errors), docs/AGENTIC_IMPROVEMENT_PLAN.md (new improvement-plan doc with progress log).

**Reason**: Improvement-plan quick wins #1/#5/#7/#14/#15/#18 from a full code audit. The chat loop never saw .decisions.json (implement/fix did) so it could contradict recorded decisions mid-conversation; plan-mode answers evaporated as terminal text; a few huge read/write messages could blow the context window while staying under the message-count cap; LM Studio surfaced rate-limit/gateway errors as [Error: ...] without retrying. Tests: tests/test_llm_retry_policy.py (10 - retry tests go through prov.chat(), the real execute_with_retry path), tests/test_quickwins_2026_08_25.py (14). Full suite: 1470 passed, 2 skipped.

## 2026-08-25 - feat: agentic quick-win batch 2 - self-review note after mutations, uncommitted-changes reminder at shutdown

**Change**: agent.py (Agent._mutating_files_this_turn extracts written/edited paths from tool results carrying the [verify] py_compile marker; Agent._print_self_review_note prints a relative-path [self-review] nudge after mutating chat turns; module-level _warn_uncommitted wired into all three REPL shutdown paths - quit, stdin-end, EOF), tests/test_quickwins_batch2.py (new, 12 tests), docs/AGENTIC_IMPROVEMENT_PLAN.md + AGENTS.md (status updates).

**Reason**: Improvement-plan quick wins #6 and #19. py_compile proves a file PARSES, not that the change does what was asked - a turn that mutated files now ends with an explicit pointer to tests/git diff. Invariant #4 (commit after every session; uncommitted work is unrecoverable - decision #058) existed only as documentation; now the REPL itself reminds at exit. Verified: extraction driven through the REAL write/edit handlers; warn paths covered for dirty repo (lists up to 5 + overflow), clean repo (silent), non-repo (silent, never raises); e2e test mirrors the real loop message shape (assistant tool_calls + matching tool result) so orphan-dropping is not bypassed. Full suite: 1482 passed, 2 skipped.

## 2026-08-25 - feat: no-emoji policy with audit enforcement (decision #079); fixes stale self-review listing

**Change**: agent_core/text_policy.py (new: stdlib-only emoji/pictograph detector - Unicode So/Sk categories plus dedicated emoji ranges U+2600-27BF/U+2B00-BFFF/U+1F000-1FAFF/U+FE0F - with ALLOWED_MONO_CHARS exempting the repo's monochrome CLI glyphs (check/cross/warning, box drawing, arrows); scan_tree skips runtime-state chat_history.json/agent_memory.json and the .docs/backups/reports trees), scripts/audit_invariants.py (check 6: scan_tree over the workspace, findings reported as audit ERRORS referencing decision #079), AGENTS.md (10 badge markers -> plain [DONE]-style bold headers; conventions section documents the policy), docs/AGENTIC_IMPROVEMENT_PLAN.md (legend and inline lightning/building tags -> [Q]/[S]), CHANGES.md (one U+FFFD mojibake byte repaired to an em dash, matching analysis_verifier's real output format; git history shows the byte was committed already corrupted, so nothing was lost). Latent bug found by the new tests: Agent._mutating_files_this_turn scanned the ENTIRE persisted history although its contract says "this turn", so after a session restart the [self-review] note listed files written by previous sessions (observed live: a fresh Agent reported 7 stale tmp_* writes from an earlier session). Fixed with Agent._turn_start_index, set in __init__ and reset at the top of every chat_nlp turn before the user message is appended; the scan now starts there.

**Reason**: User directive: avoid emojis in files. Emoji render inconsistently across Windows codepages, CI logs and terminals and can be silently mangled into unrecoverable replacement characters (the CHANGES.md byte proves it happened here). Plain-text markers survive any encoding. The monochrome glyph vocabulary stays exempt because colors.py constants and verifier outputs use them and existing tests assert on them - they are load-bearing output, not decoration. The turn-boundary fix restores the documented behavior of the self-review note.

**Files**: agent_core/text_policy.py (new), scripts/audit_invariants.py (check 6 + None-safe docs-dir name), agent.py (_turn_start_index init/reset, scoped _mutating_files_this_turn), tests/test_text_policy.py (new, 26 tests incl. audit wiring: planted emoji fails, clean tree passes, real repo is clean), tests/test_quickwins_batch2.py (TestTurnBoundaryAfterRestart: restored-history ignored, current-turn still detected, end-to-end restart-then-write prints only the new file).

## 2026-08-25 - feat: symbol-level NLP tools - definitions and references (pure AST)

**Change**: agent_core/symbol_intel.py (new: collect_definitions renders every class/function with compact unparse'd signature and [lines a-b] span, nested classes indented; collect_references walks workspace .py files with whole-word attribute-aware regex matching, capped at MAX_REFERENCES=60 hits, oversized files >1.5MB skipped, cache/backup/docs dirs excluded), agent_core/tool_schemas.py (definitions + references OpenAI schemas), agent.py (_nlp_definitions/_nlp_references handlers in the dispatch table; references scan runs in a worker thread via asyncio.to_thread; system-prompt rule steering the model to prefer these over search+read paging), agent_core/modes.py (both tools added to PLAN_MODE_TOOLS - they are verified read-only), tests/test_symbol_intel.py (new, 17 tests).

**Reason**: Improvement-plan item #8. The toolset was text-grep + line-paged read only, so "where is X used?" cost a grep plus several reads and orienting inside a big file meant paging blindly through 5000-char windows. definitions gives the map first (jump straight to the right line window); references answers usage questions in ONE call without substring false positives (run never matches run_interactive). Verified live against this repo (14 tools = 14 schemas parity; _looks_incomplete lookup returned definition + call sites + tests). Full suite: 1501 passed, 2 skipped.

## 2026-08-23 - improve: implement command — line-anchor parsing, retry sentinels, --status

**Change**: agent_core/commands/implement_cmd.py (structured `_parse_line_number` + module regexes `_PATH_LINE_RE`/`_FILE_LINE_RE`/`_WORD_LINE_RE`; generation retry loop treats `[LM Studio ...]` as failure and records per-file outcomes; new read-only `--status` mode via `ImplementCommand._status_report`; help/usage), tests/test_implement_improvements.py (new, 14 tests)

**Reason**: (1) `_parse_line_number` scanned the FIRST digit in the error string, so `agent_core/llm/v2/client.py:88:` yielded 2 (from "v2") and Windows paths yielded 1 (from "C:") — the LLM fix window was centred on wrong lines. Now anchors on mypy's `path.py:LINE:`, tracebacks' `File "...", line N`, then plain `line N`. (2) The generation retry loop only re-tried `[Error:` responses, so an `[LM Studio stream error]` sentinel broke out as if valid and the batch was silently dropped by the block parser; both sentinels now retry, and exhausted batches record `generation failed` per file instead of vanishing. (3) No way to inspect plan progress without burning LLM calls — `implement <taskplan> --status` prints ready / needs-generation / stdlib-shadowing per file (shadow check first, since a not-yet-existing shadowing directory is precisely the dangerous case) and exits before any LLM call. Verified: 14 new hermetic tests, incl. regression proof that the old digit-scan returns 2/1 on real error strings; 85/85 implement-related tests green; mypy delta vs stash baseline = zero new errors (15 identical errors, shifted line numbers); full suite run interrupted by ANOTHER session's concurrent uncommitted edits to agent.py/tool_router.py/benchmark.py (`import agent` currently broken by their missing `agent_core.metrics` module — 54 failures all traced to that, none to this change).

## 2026-08-23 - fix: streaming chat died when another shell evicted the pinned model

**Change**: agent_core/llm/lmstudio.py (new `LMStudioProvider._open_chat` shared opener — auto-reload of the pinned model on LM Studio's 400 "model is not loaded" now covers BOTH paths; `chat_stream` switched from raw `urllib.request.urlopen` to `_open_chat`), tests/test_lmstudio_stream_recovery.py (new, 5 tests)

**Reason**: The multi-shell fix taught sessions to keep their own model, and `_make_request` recovered by reloading it on demand — but only the NON-streaming path. `chat_stream` used raw `urlopen`: after shell 2 loaded a different model, shell 1's next streamed chat got an opaque `[LM Studio stream error: HTTP Error 400: Bad Request]` with no recovery. Both paths now share `_open_chat`, which on that specific 400 reloads OUR pinned model (`self.model_name` — never whatever the other shell put in VRAM) and retries once; genuine HTTP errors still surface their body. Verified: 5 new hermetic tests (stream 400 → load_model called once with the PINNED model + retry succeeds; auto-load failure reports both errors; non-load 400 triggers no load; non-streaming recovery unchanged through the shared opener; non-load errors re-raised); confirmed the new tests FAIL against a temporarily reverted `chat_stream`; mypy baseline unchanged for lmstudio.py (2 pre-existing unused-ignore in secrets.py); full suite 1146 passed + 2 skipped.

## 2026-08-23 - fix: concurrent agent.py shells hijacked each other's model

**Change**: agent_core/commands/model_cmd.py (`_list_models` auto-sync → read-only advisory), agent_core/constants.py (`resolve_model` priority: persisted model.json now outranks the live LM Studio poll; live poll demoted to first-run fallback), tests/test_multi_shell_model_isolation.py (new, 7 tests)

**Reason**: With two `python agent.py` shells open, switching the model in shell 2 broke shell 1 through three leak paths: (1) `model list` in shell 1 SILENTLY switched the session to whatever shell 2 had loaded and persisted it to model.json/.env; (2) `resolve_model()` ranked the live LM Studio poll ABOVE the persisted choice, so any new Agent/LLMClient adopted the other shell's VRAM contents instead of its own persisted model; (3) nothing pinned a running session to its chosen model at all. Fix contract: a session keeps ITS model — listing is read-only (prints an advisory with an explicit `model <name>` / `model reload` adoption hint), the persisted choice wins over the live poll, and `LMStudioProvider._make_request`'s on-demand auto-reload (unchanged) recovers the pinned model when a request finds it missing from VRAM. Verified: 7 new regression tests cover all three paths end-to-end through the real LLMClient constructor (hermetic — no real server/model.json); 71 model-related + 103 tool-loop/parallel/component tests green; mypy back to baseline for both touched files (4 pre-existing errors each).

## 2026-08-21 - feat: multillm simultaneous-call regression test

**Change**: tests/test_parallel_llm.py (new `TestMultiLlmCommandRoles.test_simultaneous_review_command_end_to_end`)

**Reason**: prove the EXACT command ``multillm "review agent.py, and synthesize the two models' responses" --models laguna-s-2.1,opencode-go/deepseek-v4-flash --role "laguna-s-2.1:You are a security auditor..." --role "opencode-go/deepseek-v4-flash:You are a performance engineer..."`` fires SIMULTANEOUSLY end-to-end. The test drives the real REPL path (`shlex.split(posix=False)` → `MultiLlmCommand.execute`) with two 0.2s-latency fake providers and asserts wall time < 0.35s (parallel) — verified it FAILS (0.469s) against a temporarily serialized `run_parallel`, then passes. Also asserts each model got its own role system prompt and the shared quoted question survived intact. Full suite 1280 passed.

## 2026-08-21 - fix: multillm parallel calls were actually serial (blocking HTTP)

**Change**: agent_core/llm/lmstudio.py (`_do_request` → `asyncio.to_thread(self._make_request, payload)`), agent_core/llm/opencode_provider.py (`_with_retry` → `await asyncio.to_thread(factory)`; `_ensure_session` → `await asyncio.to_thread(self._request, ...)`), tests/test_parallel_llm.py (new `test_blocking_providers_do_not_serialize`)

**Reason**: multillm fired both models via `asyncio.gather`, but BOTH providers made their HTTP round-trip with SYNCHRONOUS blocking `urllib.request.urlopen` inside `async def chat()`. The first coroutine's sync call stalled the event loop, so the second model's `chat()` could not even start until the first finished — the two "parallel" calls were sequential. The fix dispatches every blocking HTTP call to a worker thread (`asyncio.to_thread`), so the event loop stays free and `gather` gets real concurrency. Verified: the new test mocks `urllib.request.urlopen` to block 0.2s per call and asserts two calls finish in <0.35s (was ~0.42s serialized — confirmed by temporarily reverting the fix); live smoke test of both real providers: 0.203s for two 0.2s blocking calls; full suite 1279 passed.

## 2026-08-21 - feat: tools available to multillm models

**Change**: agent_core/llm/parallel.py (tools/execute_tool_fn/max_tool_iterations params + tool loop), agent_core/commands/multillm_cmd.py (passes NLP_TOOL_SCHEMAS + agent._execute_tool_call), tests/test_parallel_llm.py (3 new TestParallelTools)

**Reason**: multillm fired the same prompt at multiple LLMs, but the models had NO tools — they could only answer from the prompt. The 2026-08-21 session showed both laguna-s-2.1 and opencode-go/deepseek-v4-flash asking "please paste the file content" instead of reading it. Now each model runs through the SAME ToolLoopRunner the agent uses: it can read/search/list files, run tests, etc. Each model gets its OWN loop instance (no cross-model tool-state contamination), all running concurrently via asyncio.gather. The command passes the agent's real `_execute_tool_call` executor + `NLP_TOOL_SCHEMAS`, degrading gracefully (no tools) when the host doesn't expose the executor. Verified: 3 new tests (tool call → execute → answer with result; tools-without-executor rejected; no-tools keeps single chat); live smoke test of the full tool loop; mypy-clean on both files; 76 related tests green.

## 2026-08-16 - feat: quoted inline --role values in multillm

**Change**: agent_core/commands/multillm_cmd.py (strip quotes from --role value), tests/test_parallel_llm.py (3 new TestMultiLlmCommandRoles)

**Reason**: the REPL splits input with `shlex.split(posix=False)`, which KEEPS the literal quotes on a quoted `--role "model:multi-word prompt"` value — so inline roles with multi-word prompts were broken (the model name would include a leading quote). The command now strips the quotes (repo convention: analyze_cmd/fix_cmd/implement_cmd all use `.strip('"')`). Inline `--role` is the primary way to set roles; `--role-file` remains for reusable setups. Verified: live smoke test through the real REPL shlex path; 3 new command-level tests (quoted multi-word, unquoted single-word, malformed rejected); 19 parallel tests green.

## 2026-08-16 - feat: per-model roles in multillm

**Change**: agent_core/llm/parallel.py (roles param), agent_core/commands/multillm_cmd.py (--role / --role-file flags), tests/test_parallel_llm.py (3 new TestRoles)

**Reason**: `multillm` fired the same prompt at multiple LLMs — but all models got the identical question, so they could only differ by their own weights, never by assigned expertise. Now each model can play a DIFFERENT expert role: `run_parallel(..., roles={model: system_prompt})` prepends that model's own `system` message to the shared question (both providers already handle a leading system block — LM Studio preserves it, opencode splits it into the system prompt). The command gains `--role model:prompt` (repeatable) and `--role-file path.json` (`{"model": "system prompt"}`) and prints the assigned role under each model's header. Roles are keyed by exact model name and never leak across models; unknown-model roles are ignored. Verified: 3 new tests (role prepended only to its model, no-role unchanged, unknown-model ignored); live smoke test of both flags + malformed-role rejection; mypy-clean on both files; 70 related tests green.

## 2026-08-16 - feat: parallel multi-LLM dispatch (multillm command)

**Change**: agent_core/llm/parallel.py (new), agent_core/commands/multillm_cmd.py (new), agent.py (register MultiLlmCommand + help entry), tests/test_parallel_llm.py (13 new)

**Reason**: The agent could route per-model to different providers (LM Studio local vs hosted opencode-go via `build_provider`) but only SEQUENTIALLY — nothing fired simultaneous calls to different LLMs. New `agent_core/llm/parallel.py::run_parallel` builds one provider instance per model and fires all `chat` calls concurrently with `asyncio.gather` (one provider per model = separate opencode sessions, no shared state). `return_exceptions=True` + provider error strings isolate a dead server so one failure never aborts the other models' answers. Each result carries provider metrics and the dormant `ConsensusVoter`/`RefinementVoter` machinery finally gets a real producer: every model's verdict is recorded under a template id and the quorum gate is exposed via `ParallelRun.quorum_reached()` (approval-ratio semantics matching `ConsensusVoter.tally_votes` — the pre-existing `RefinementVoter.decide()` path never fed its own consensus ledger and always returned False, so the quorum is computed directly from the vote ledger). New `multillm` REPL command: `multillm "question" --models laguna-s-2.1,opencode-go/deepseek-v4-flash` prints each model's answer with a consensus summary; defaults to the current agent model + configured opencode model. Verified: 13 new tests prove wall-clock parallelism (two 0.2s fake providers finish in ~0.2s, concurrency=1 serializes to ~0.4s), per-model build_provider routing (LMStudioProvider vs OpencodeProvider), error isolation, and consensus; mypy-clean on both new files; full suite 1269 passed / 2 skipped.

## 2026-08-16 - feat: paste an image into the agent for vision-capable LLMs

**Change**: agent_core/commands/paste_image_cmd.py (new), agent.py (chat_nlp images arg + multimodal user message + _strip_image_blocks + _save/_load_chat_history stripping), tests/test_paste_image.py (8 new)

**Reason**: There was no way to feed an image to a multimodal model (screenshots, diagrams, photos, UI errors). New `paste_image` REPL command reads an image from the clipboard (Pillow `ImageGrab.grabclipboard()`) or a file path, base64-encodes it as a data URL, and sends it to `chat_nlp` alongside an optional `--prompt`. `chat_nlp` now builds an OpenAI-format content array (`text` + one `image_url` block per image) so vision models can see the image. Image blocks are stripped from `chat_history.json` on save/load (decision: never persist multi-MB base64 blobs) — a pure-image turn is dropped from the persisted window while a mixed text+image turn keeps its text. Verified: provider `sanitize_message_roles` preserves the multimodal content array; 8 new tests green; existing 67 tool_loop_nlp + 3 registry tests still green.

## 2026-08-16 - fix: model jumps away from opencode + 400 root cause (orphan tool messages)

**Change**: agent_core/commands/model_cmd.py, agent.py, agent_core/llm/lmstudio.py, agent_core/llm/opencode_provider.py, tests (model_helpers, lmstudio_payload, opencode_provider, tool_loop_nlp)

**Reason**: (1) model list auto-synced the agent to the LM Studio-loaded model whenever the current model was 'not loaded' — REGARDLESS of provider — so opencode-go/deepseek-v4-flash was silently replaced by qwen and provider=lmstudio persisted, across sessions. Auto-sync (list + reload) is now gated on the active provider being lmstudio; sync also rebuilds the provider (stale-provider fix). (2) The repeated 400 'Bad Request' from the opencode API was traced to ORPHAN tool messages: _trim_chat_history can cut between an assistant tool_calls message and its tool result; strict gateways reject the surviving orphan with HTTP 400 (verified live: 'Messages with role tool must be a response to a preceding message with tool_calls'). Fixes: orphan tool messages are dropped in _trim_chat_history AND in the provider-level sanitizer (defense in depth, also cleans old persisted histories); opencode provider errors now surface the gateway body (the lmstudio provider already did). Verified live: model list keeps opencode-go/deepseek-v4-flash; orphan payload succeeds (no 400). 876 tests green (7 new).

## 2026-08-16 - fix: read-only tasks (audits) stopped mid-work by the mutation-based progress guard

**Change**: agent_core/llm/tool_loop.py, agent.py, tests/test_tool_loop_nlp.py, tests/test_loop_synthesis.py

**Reason**: The fan-platform security audit (trace b705348b...) was cut off at 51 tool calls (iteration 29 of 150, ~80% budget left) by the no-mutation guard even though the model was still discovering NEW files every call — read-only tasks never mutate, so mutation was a wrong progress proxy. Fix: the progress guard is now CONVERGENCE-based — progress = mutation OR discovering something new (new read window, search query, directory, or command); only repeats of known calls increment the stuck counter (the genuine stuck signal, e.g. the earlier agent.py paging spiral). Nudge/force texts updated accordingly. Safety net: a forced synthesis whose answer still signals unfinished work ('not covered', 'follow-up' markers added) now chains one continuation run even after no_progress. Verified live (qwen): read-only task runs to completion with a concrete answer. 869 tests green (new audit-runs-to-completion test).

## 2026-08-16 - fix: qwen 400 'System message must be at the beginning'

**Change**: agent_core/llm/tool_loop.py, agent.py, agent_core/llm/lmstudio.py, agent_core/llm/opencode_provider.py, tests/test_lmstudio_payload.py, tests/test_tool_loop_nlp.py, tests/test_loop_synthesis.py

**Reason**: The tool loop injected steering notes (no-mutation nudge, budget warning, forced-synthesis note, continuation note) as role=system MID-conversation; qwen's Jinja chat template rejects any system message outside the leading block ('System message must be at the beginning') -> LM Studio predict 500 wrapped as HTTP 400, killing long sessions. Fix, three layers: (1) all injected steering notes now travel as role=user (leading system block untouched) and the history-strip filters match on content; (2) NEW provider-level sanitize_message_roles() converts any system message after the leading block to user role ('[System note] ...') in both LM Studio and opencode payloads - a safety net that also neutralizes old leaked notes in persisted chat_history.json; (3) verified live against qwen/qwen3.8-27b: raw mid-list system payload reproduces the exact error, sanitized payload returns 200. 866 tests green (4 new sanitizer tests).

## 2026-08-16 - fix: cryptic 'no answer' in quiet mode -> forced-synthesis retry + concrete fallback

**Change**: agent.py, agent_core/llm/tool_loop.py, tests/test_loop_synthesis.py (ny), tests/test_tool_loop_nlp.py

**Reason**: After a long tool-loop run (72 iterations, 87 calls) the no_mutation guard fired with 'Give your final answer now' but the forced synthesis returned nothing (large context) and the user got only the cryptic '(The assistant did not produce a response...)' — especially bad in QUIET mode. Three fixes: (1) the forced-synthesis LLM call is now traced (decision #034: every loop event, incl. the final tool-less call); (2) an empty forced response triggers ONE explicit retry (_FORCED_SYNTHESIS_RETRY) so the loop keeps its never-ends-without-an-answer guarantee; (3) chat_nlp now prints a CONCRETE fallback with loop statistics (calls made, tools used, last action, termination reason) instead of the generic message, and providers' '(no output)' is normalized to empty so the fallback/retry triggers. Verified live: quiet mode prints the concrete answer only. 862 tests green (5 new).

## 2026-08-16 - fix: run timeout killed only the shell, orphaning children

**Change**: agent.py (run-tool branch + _kill_process_tree), agent_core/commands/run_cmd.py, tests/test_run_cmd.py, tests/test_run_timeout.py (ny)

**Reason**: 
un python -m harnessfix.loop --traces ... timed out after 120s but kept working: subprocess.run on Windows kills only cmd.exe on timeout, leaving the harnessfix.loop child orphaned and holding the captured pipes — the REPL then waited until the orphan finished (165s) while the message said 'timed out'. Fixes: (1) the run tool now starts the shell in its own process group (CREATE_NEW_PROCESS_GROUP) and kills the WHOLE tree on timeout via taskkill /T /F (verified live: returns at 8s, grandchild gone from tasklist); (2) the run REPL command now defaults to 600s and accepts --timeout <sec>; (3) the timeout message says the tree was killed. 845 tests green (6 new). agent.py also carries the graceful harnessfix tracing wiring (try/except degrades to no-op when harnessfix/ is absent).

## 2026-08-16 - feat: LLM-free 
un REPL command + LM Studio 400 diagnostics

**Change**: agent_core/commands/run_cmd.py (ny), agent.py (REPL-wiring, ucommitted pga. harnessfix-afhaengighed), agent_core/llm/lmstudio.py, tests/test_run_cmd.py (ny), tests/test_lmstudio_http.py (ny)

**Reason**: (1) 
un <shell command> in the REPL executes shell commands DIRECTLY without the LLM - deterministic byte-exact output (harnessfix dashboard dumps), zero tokens, works when the provider is down. Reuses the guarded run-tool path (blocked-command allowlist + truncation); verified live (python -c print works, rm -rf / blocked). (2) The 400 the user hit surfaced as a useless 'HTTP Error 400: Bad Request' - LM Studio's response body (e.g. 'model is not loaded, load it first') was swallowed, and the 'after 3 retries' wording was misleading because HTTPError is not in RetryPolicy.retryable_errors so it fails on the first attempt. Fix: _make_request now surfaces the response body, auto-loads the model once when a 400 says it is not in VRAM, and the error message is reworded honestly. 839 tests green (9 new).

## 2026-08-16 - fix: LM Studio models routed to opencode API (401)

**Change**: agent_core/llm/provider.py, agent_core/constants.py, agent_core/commands/model_cmd.py, tests/test_opencode_provider.py

**Reason**: After switching to an LM Studio model (e.g. reactagent-1.5b), chat still hit the hosted opencode API and 401'd. Two bugs: (1) the model command updated agent.llm.model_name but never rebuilt the provider, so a previously selected opencode provider kept receiving LM Studio models; (2) provider_for() ignored the provider persisted in model.json and fell back to AGENT_LLM_PROVIDER for unrecognized names. Fix: persisted provider (model.json) now wins over the setting for unprefixed names; all LM Studio switch paths (switch/known-fallback/sync) rebuild the provider via _rebuild_provider; provider display calls pass the persisted provider. Also fixed resolve_model defaulting an empty model.json to lmstudio, which defeated the opencode setting. Verified live: reactagent-1.5b -> LMStudioProvider, opencode-go/* -> OpencodeProvider. 830 tests green.

## 2026-08-16 - feat: HarnessFix phases 0-4 (trace-grounded harness repair)

**Change**: new `harnessfix/` package (tracing, reader, htir, links, diagnose,
corpus, gates, loop, repairs/), instrumented `agent_core/llm/tool_loop.py`,
wired `agent.py` chat_nlp trace capture, root `conftest.py`, .gitignore.

**Reason**: Implement docs/HARNESSFIX_SPEC.md — the closed loop that captures
per-task tool-loop traces, compiles them to a layer-faceted HTIR graph with
provenance/control-flow links, diagnoses failures to one of the 7 harness
layers, applies a scoped revertible code-level repair, and keeps it only if
the pytest + security + benchmark gates pass with no regression.

- **Phase 0 tracing**: `ToolLoopRunner.run()` accepts an optional trace sink;
  emits `step_start|llm_response|tool_call|tool_result|tool_error|
  guard_triggered|loop_end` JSONL events to `reports/traces/{task_id}.jsonl`,
  each tagged with a layer facet + correlation id. Non-invasive: no sink ==
  byte-identical behaviour (verified by test); AGENT_NO_TRACE=1 opt-out.
  `agent.py` attaches a `TraceWriter` per run() invocation (decision #029).
- **Phase 1 HTIR**: `htir.py` compiles a trace to a `TraceGraph`
  (reproducible from the trace alone); `links.py` infers provenance links
  (tool args reusing earlier result tokens) and control-flow links (steps
  caused by injected guard notes, stuck cycles).
- **Phase 2 diagnosis**: `diagnose.py` heuristic tier maps each failed trace
  to exactly one layer (tool_interface / execution_environment / governance /
  verification / context / lifecycle) with evidence link ids + confidence;
  writes `reports/harnessfix/diagnoses/{task_id}.json`.
- **Phase 3 repair catalog**: `repairs/tool_interface.py` — seed repair
  (include exception type in fed-back tool errors); apply/revert are exact
  source transforms verified by py_compile.
- **Phase 4 loop**: `python -m harnessfix.loop` collects traces, diagnoses
  failed ones, proposes the highest-frequency layer's repair, applies it
  behind a human review gate (`--approve`, fail-closed headless), runs
  pytest + security (+ optional benchmark pass-rate) gates, and accepts only
  if tests/security pass and the benchmark did not regress; verdict and
  per-layer deltas land in `reports/harnessfix/summary.json`.

Verified: `python -m harnessfix.loop` on a synthetic corpus diagnosed 2
failed traces (tool_interface + lifecycle), proposed the tool-interface
repair, and — on the real pytest gate — correctly REJECTED and reverted it
because the existing test `test_tool_error_is_fed_back_not_crashing` asserts
the old error string (gate-driven acceptance working). 22 new harnessfix
tests green; full suite 831 passed, 1 skipped; mypy strict clean on the new
package.

## 2026-08-16 - fix: opencode API read timeout killed workflow plan step

**Change**: agent_core/llm/opencode_provider.py, .env.example

**Reason**: Workflow plan step failed with 'opencode API request failed: The read operation timed out' after 694s — the direct-API default timeout was 120s (OPENCODE_TIMEOUT) while large plan prompts take minutes to generate. Raised the default to 600s (same as server mode) and documented OPENCODE_TIMEOUT in .env.example.

## 2026-08-15 - opencode-go direct API mode

**Change**: agent_core/llm/opencode_provider.py, agent_core/config.py, agent_core/llm/provider.py, agent_core/commands/model_cmd.py, tests/test_opencode_provider.py

**Reason**: Fully integrated opencode-go as an LLM provider: the provider now has two modes sharing the same chat contract - (1) DIRECT hosted API (https://opencode.ai/zen/go/v1, OpenAI-compatible, verified live) with NATIVE tool calling, activated automatically when an API key is available (OPENCODE_API_KEY env or opencode's auth.json store, opencode-go entry); (2) the existing local opencode serve fallback. Key resolution env-first, never logged. model list shows the hosted catalog (26 opencode-go models) in API mode, server catalog otherwise. Hosted model ids are unprefixed in payloads (verified: prefixed ids get 403 Model not supported), re-prefixed (opencode-go/...) for agent consistency. Sends a browser-like User-Agent (Cloudflare blocks Python-urllib, error 1010). Verified end-to-end live (chat + list). 24 provider tests green, full suite 802 passed.


## 2026-07-26 17:35 — fix --desc

**Change**: Modified `entities.py`
**Reason**: Fix all issues from project_tasks.md: 1) Merge duplicate exceptions from entities.py into agent_core/entities.py. 2) Rename _validate_path to normalize_path in path_utils.py.

## 2026-07-26 17:35 — fix --desc

**Change**: Modified `entities.py`
**Reason**: Fix all issues from project_tasks.md: 1) Merge duplicate exceptions from entities.py into agent_core/entities.py. 2) Rename _validate_path to normalize_path in path_utils.py.

## 2026-07-26 17:35 — fix --desc

**Change**: Modified `path_utils.py`
**Reason**: Fix all issues from project_tasks.md: 1) Merge duplicate exceptions from entities.py into agent_core/entities.py. 2) Rename _validate_path to normalize_path in path_utils.py.

## 2026-07-26 17:35 — fix --desc

**Change**: Modified `path_utils.py`
**Reason**: Fix all issues from project_tasks.md: 1) Merge duplicate exceptions from entities.py into agent_core/entities.py. 2) Rename _validate_path to normalize_path in path_utils.py.

## 2026-07-26 22:58 — fix --desc

**Change**: Modified `__init__.py`
**Reason**: Fix all issues from project_tasks.md: 3) Update __init__.py exports. 4) Fix read_file dedup in agent.py. 5) Fix analyze_handler AST traversal. 6) Register ShellCommandHandler in tool_router.py. 7) Con

## 2026-07-26 22:58 — fix --desc

**Change**: Modified `agent.py`
**Reason**: Fix all issues from project_tasks.md: 3) Update __init__.py exports. 4) Fix read_file dedup in agent.py. 5) Fix analyze_handler AST traversal. 6) Register ShellCommandHandler in tool_router.py. 7) Con

## 2026-07-26 22:58 — fix --desc

**Change**: Modified `benchmark.py`
**Reason**: Fix all issues from project_tasks.md: 3) Update __init__.py exports. 4) Fix read_file dedup in agent.py. 5) Fix analyze_handler AST traversal. 6) Register ShellCommandHandler in tool_router.py. 7) Con

## 2026-07-26 22:58 — fix --desc

**Change**: Modified `entities.py`
**Reason**: Fix all issues from project_tasks.md: 3) Update __init__.py exports. 4) Fix read_file dedup in agent.py. 5) Fix analyze_handler AST traversal. 6) Register ShellCommandHandler in tool_router.py. 7) Con

## 2026-07-26 22:58 — fix --desc

**Change**: Modified `exceptions.py`
**Reason**: Fix all issues from project_tasks.md: 3) Update __init__.py exports. 4) Fix read_file dedup in agent.py. 5) Fix analyze_handler AST traversal. 6) Register ShellCommandHandler in tool_router.py. 7) Con

## 2026-07-26 22:58 — fix --desc

**Change**: Modified `logging_config.py`
**Reason**: Fix all issues from project_tasks.md: 3) Update __init__.py exports. 4) Fix read_file dedup in agent.py. 5) Fix analyze_handler AST traversal. 6) Register ShellCommandHandler in tool_router.py. 7) Con

## 2026-07-26 22:58 — fix --desc

**Change**: Modified `path_utils.py`
**Reason**: Fix all issues from project_tasks.md: 3) Update __init__.py exports. 4) Fix read_file dedup in agent.py. 5) Fix analyze_handler AST traversal. 6) Register ShellCommandHandler in tool_router.py. 7) Con

## 2026-07-26 23:59 — fix --desc

**Change**: Modified `__init__.py`
**Reason**: Fix all issues from project_tasks.md: 3) Update __init__.py exports. 4) Fix read_file dedup in agent.py. 5) Fix analyze_handler AST traversal. 6) Register ShellCommandHandler in tool_router.py. 7) Con

## 2026-07-26 23:59 — fix --desc

**Change**: Modified `agent.py`
**Reason**: Fix all issues from project_tasks.md: 3) Update __init__.py exports. 4) Fix read_file dedup in agent.py. 5) Fix analyze_handler AST traversal. 6) Register ShellCommandHandler in tool_router.py. 7) Con

## 2026-07-26 23:59 — fix --desc

**Change**: Modified `analyze_handler.py`
**Reason**: Fix all issues from project_tasks.md: 3) Update __init__.py exports. 4) Fix read_file dedup in agent.py. 5) Fix analyze_handler AST traversal. 6) Register ShellCommandHandler in tool_router.py. 7) Con

## 2026-07-26 23:59 — fix --desc

**Change**: Modified `tool_router.py`
**Reason**: Fix all issues from project_tasks.md: 3) Update __init__.py exports. 4) Fix read_file dedup in agent.py. 5) Fix analyze_handler AST traversal. 6) Register ShellCommandHandler in tool_router.py. 7) Con

## 2026-07-26 23:59 — fix --desc

**Change**: Modified `benchmark.py`
**Reason**: Fix all issues from project_tasks.md: 3) Update __init__.py exports. 4) Fix read_file dedup in agent.py. 5) Fix analyze_handler AST traversal. 6) Register ShellCommandHandler in tool_router.py. 7) Con


## 2026-08-11 22:19 — fix --mypy

**Change**: Modified `agent.py`
**Reason**: mypy error fixes

## 2026-08-11 22:24 — fix --mypy

**Change**: Modified `agent.py`
**Reason**: mypy error fixes

## 2026-08-11 22:46 — fix --mypy

**Change**: Modified `agent.py`
**Reason**: mypy error fixes

## 2026-08-11 22:52 — fix --mypy

**Change**: Modified `agent.py`
**Reason**: mypy error fixes

## 2026-08-11 22:56 — fix --mypy

**Change**: Modified `agent.py`
**Reason**: mypy error fixes

## 2026-08-12 00:42 — fix --mypy

**Change**: Modified `agent.py`
**Reason**: mypy error fixes

## 2026-08-12 00:46 — fix --mypy

**Change**: Modified `agent.py`
**Reason**: mypy error fixes

## 2026-08-12 00:48 — fix --mypy

**Change**: Modified `agent.py`
**Reason**: mypy error fixes

## 2026-08-12 00:49 — fix --mypy

**Change**: Modified `agent.py`
**Reason**: mypy error fixes

## 2026-08-12 00:51 — fix --mypy

**Change**: Modified `agent.py`
**Reason**: mypy error fixes

## 2026-08-12 00:59 — fix --mypy

**Change**: Modified `agent.py`
**Reason**: mypy error fixes

## 2026-08-12 01:06 — fix --mypy

**Change**: Modified `agent.py`
**Reason**: mypy error fixes

## 2026-08-12 01:18 — fix --mypy

**Change**: Modified `agent.py`
**Reason**: mypy error fixes

## 2026-08-12 01:28 — fix --mypy

**Change**: Modified `agent.py`
**Reason**: mypy error fixes

## 2026-08-12 01:31 — fix --mypy

**Change**: Modified `agent.py`
**Reason**: mypy error fixes

## 2026-08-12 01:42 — fix --mypy

**Change**: Modified `agent.py`
**Reason**: mypy error fixes

## 2026-08-12 01:46 — fix --mypy

**Change**: Modified `agent.py`
**Reason**: mypy error fixes

## 2026-08-12 01:50 — fix --mypy

**Change**: Modified `agent.py`
**Reason**: mypy error fixes

## 2026-08-12 01:53 — fix --mypy

**Change**: Modified `agent.py`
**Reason**: mypy error fixes

## 2026-08-12 09:20 — fix --mypy

**Change**: Modified `agent.py`
**Reason**: mypy error fixes

## 2026-08-12 09:22 — fix --mypy

**Change**: Modified `agent.py`
**Reason**: mypy error fixes

## 2026-08-12 10:19 — fix --mypy

**Change**: Modified `agent.py`
**Reason**: mypy error fixes

## 2026-08-12 10:20 — optimize / fix / implement / decide / workflow

**Change**: Modified `agent_core/commands/base.py`, `fix_cmd.py`, `optimize_cmd.py`, `implement_cmd.py`, `decide_cmd.py`, `workflow_cmd.py`, `agent.py`, `patch_utils.py`, `tests/test_flow_control.py`, `tests/test_patch_application.py`, `Architecture.md`, `model.json`
**Reason**: Introduce flow-stop controls (Ctrl+C / "stop"/"q" wind down whole runs) and show what issues each patch targets at presentation time.  Also fix ambiguous hunk anchoring for common one-line patterns, clean up redundant casts and return-type annotations in agent.py tool handlers.

## 2026-08-12 13:06 — fix --mypy

**Change**: Modified `common.py`
**Reason**: mypy error fixes

## 2026-08-12 13:28 — fix --mypy

**Change**: Modified `optimize_cmd.py`
**Reason**: mypy error fixes

## 2026-08-12 13:35 — fix --mypy

**Change**: Modified `patterns.py`
**Reason**: mypy error fixes

## 2026-08-12 13:51 — fix --mypy

**Change**: Modified `patterns.py`
**Reason**: mypy error fixes

## 2026-08-12 13:55 — fix --mypy

**Change**: Modified `patterns.py`
**Reason**: mypy error fixes

## 2026-08-12 13:57 — fix --mypy

**Change**: Modified `patterns.py`
**Reason**: mypy error fixes

## 2026-08-12 14:15 — fix --mypy

**Change**: Modified `patterns.py`
**Reason**: mypy error fixes

## 2026-08-12 14:15 — fix --mypy

**Change**: Modified `patterns.py`
**Reason**: mypy error fixes

## 2026-08-12 15:08 — fix --mypy

**Change**: Modified `patterns.py`
**Reason**: mypy error fixes

## 2026-08-12 15:10 — fix --mypy

**Change**: Modified `patterns.py`
**Reason**: mypy error fixes

## 2026-08-12 15:50 — fix --mypy

**Change**: Modified `patterns.py`
**Reason**: mypy error fixes

## 2026-08-12 16:09 — fix --mypy

**Change**: Modified `retry.py`
**Reason**: mypy error fixes

## 2026-08-12 18:41 — fix --mypy

**Change**: Modified `_implement_raw.py`
**Reason**: mypy error fixes

## 2026-08-12 18:43 — fix --mypy

**Change**: Modified `_implement_raw.py`
**Reason**: mypy error fixes

## 2026-08-12 18:45 — fix --mypy

**Change**: Modified `cache_io.py`
**Reason**: mypy error fixes

## 2026-08-12 18:47 — fix --mypy

**Change**: Modified `entities.py`
**Reason**: mypy error fixes

## 2026-08-12 18:48 — fix --mypy

**Change**: Modified `_implement_raw.py`
**Reason**: mypy error fixes

## 2026-08-12 18:49 — fix --mypy

**Change**: Modified `analyze_handler.py`
**Reason**: mypy error fixes

## 2026-08-12 18:51 — fix --mypy

**Change**: Modified `analyze_handler.py`
**Reason**: mypy error fixes

## 2026-08-12 18:52 — fix --mypy

**Change**: Modified `analyze_handler.py`
**Reason**: mypy error fixes

## 2026-08-12 18:52 — fix --mypy

**Change**: Modified `context_management.py`
**Reason**: mypy error fixes

## 2026-08-12 18:54 — fix --mypy

**Change**: Modified `analyze_handler.py`
**Reason**: mypy error fixes

## 2026-08-12 18:55 — fix --mypy

**Change**: Modified `analyze_handler.py`
**Reason**: mypy error fixes

## 2026-08-12 18:59 — fix --mypy

**Change**: Modified `optimize_cmd.py`
**Reason**: mypy error fixes

## 2026-08-12 19:57 — fix --mypy

**Change**: Modified `fix_cmd.py`
**Reason**: mypy error fixes

## 2026-08-12 19:57 — fix --mypy

**Change**: Modified `fix_cmd.py`
**Reason**: mypy error fixes

## 2026-08-12 19:59 — fix --mypy

**Change**: Modified `lmstudio.py`
**Reason**: mypy error fixes

## 2026-08-12 19:59 — fix --mypy

**Change**: Modified `lmstudio.py`
**Reason**: mypy error fixes

## 2026-08-12 20:01 — fix --mypy

**Change**: Modified `constants.py`
**Reason**: mypy error fixes

## 2026-08-12 20:14 — fix --mypy

**Change**: Modified `fix_cmd.py`
**Reason**: mypy error fixes

## 2026-08-12 20:15 — fix --mypy

**Change**: Modified `fix_cmd.py`
**Reason**: mypy error fixes

## 2026-08-12 22:19 — fix --mypy

**Change**: Modified `optimize_cmd.py`
**Reason**: mypy error fixes

## 2026-08-12 22:43 — fix --mypy

**Change**: Modified `optimize_cmd.py`
**Reason**: mypy error fixes

## 2026-08-12 22:55 — fix --mypy

**Change**: Modified `lmstudio.py`
**Reason**: mypy error fixes

## 2026-08-12 23:07 — fix --mypy

**Change**: Modified `fix_cmd.py`
**Reason**: mypy error fixes

## 2026-08-12 23:09 — fix --mypy

**Change**: Modified `decisions.py`
**Reason**: mypy error fixes

## 2026-08-12 23:27 — fix --mypy

**Change**: Modified `lmstudio.py`
**Reason**: mypy error fixes

## 2026-08-12 23:36 — fix --mypy

**Change**: Modified `decisions.py`
**Reason**: mypy error fixes

## 2026-08-12 23:37 — fix --mypy

**Change**: Modified `fix_cmd.py`
**Reason**: mypy error fixes

## 2026-08-13 11:41 — fix --mypy

**Change**: Modified `optimize_cmd.py`
**Reason**: mypy error fixes

## 2026-08-13 11:41 — fix --mypy

**Change**: Modified `optimize_cmd.py`
**Reason**: mypy error fixes

## 2026-08-13 11:42 — fix --mypy

**Change**: Modified `lmstudio.py`
**Reason**: mypy error fixes

## 2026-08-13 11:42 — fix --mypy

**Change**: Modified `lmstudio.py`
**Reason**: mypy error fixes

## 2026-08-13 11:56 — fix --mypy

**Change**: Modified `optimize_cmd.py`
**Reason**: mypy error fixes

## 2026-08-13 12:22 — fix --mypy

**Change**: Modified `agent.py`
**Reason**: mypy error fixes

## 2026-08-13 12:24 — fix --mypy

**Change**: Modified `decisions.py`
**Reason**: mypy error fixes

## 2026-08-13 13:38 — structured NLP tool calling

**Change**: NLP natural-language turns now use native OpenAI-format tool calling instead of the ReAct <tool_call> text-tag loop.
**Reason**: The text-tag approach let the model describe an action ("I will read the file...") without taking it. Native tool_calls force the model to either call a tool or answer in text, every call is shown as [tool] name(args) with its result, writes are followed by a py_compile verification summary, and tool errors are fed back for retry.
**Files**: agent.py (chat_nlp, _execute_tool_call, _verify_file), agent_core/tool_schemas.py (new), agent_core/llm/tool_loop.py (iterate + display + last-text), agent_core/commands/analyze_cmd.py, tests/test_tool_loop_nlp.py (new)


## 2026-08-13 13:57 — tool loop deadline: forced final synthesis

**Change**: The NLP tool loop no longer dies silently at the iteration cap. ToolLoopRunner now injects a BUDGET WARNING system note when only a few iterations remain, and if the cap is hit while tool calls are still pending, one final tool-less LLM call forces the model to synthesize the answer from the gathered tool results. chat_nlp iteration cap raised from 8 to 20.
**Reason**: A multi-step task (e.g. 'analyze the whole project and write a report') burned all iterations on exploration, the loop exited mid-task, and the model's intermediate narration was printed as if it were the final answer.
**Files**: agent_core/llm/tool_loop.py (deadline note, forced synthesis, deadline_window), agent.py (max_iterations 8 -> 20), tests/test_tool_loop_nlp.py (+3 tests), Architecture.md (component table).


## 2026-08-13 14:10 — tool loop deadline: forced final synthesis

**Change**: The NLP tool loop no longer dies silently at the iteration cap. ToolLoopRunner now injects a BUDGET WARNING system note when only a few iterations remain, and if the cap is hit while tool calls are still pending, one final tool-less LLM call forces the model to synthesize the answer from the gathered tool results. chat_nlp iteration cap raised from 8 to 20.
**Reason**: A multi-step task (e.g. 'analyze the whole project and write a report') burned all iterations on exploration, the loop exited mid-task, and the model's intermediate narration was printed as if it were the final answer.
**Files**: agent_core/llm/tool_loop.py (deadline note, forced synthesis, deadline_window), agent.py (max_iterations 8 -> 20), tests/test_tool_loop_nlp.py (+3 tests), Architecture.md (component table).


## 2026-08-13 14:10 — fix: Git-Bash workspace broke subprocess tools

**Change**: Agent.DEFAULT_WORKSPACE was the Git-Bash path "/c/Dev/Agent1", which is not a valid Windows directory. Every subprocess-based NLP tool (git, run, diff, py_compile verification) ran with cwd=ws_dir and failed with WinError 267 "The directory name is invalid", while read/search/list_files kept working. The default is now derived from the script location, the constructor translates Git-Bash-style paths via to_windows_path, and _execute_tool_call/_verify_file fall back to a valid cwd if _nlp_workspace is invalid.
**Files**: agent.py (DEFAULT_WORKSPACE, __init__, _execute_tool_call ws_dir, _verify_file cwd), tests/test_tool_loop_nlp.py (+2 regression tests)


## 2026-08-13 14:20 — persistent NLP chat history across sessions

**Change**: The natural-language conversation now survives REPL restarts. After every chat_nlp turn the history is saved to chat_history.json (project root, git-ignored) and loaded back on startup, so the next session continues where the previous one left off. History is capped at 60 messages (system prompt + tail) to keep context bounded, corrupt files fall back to empty, and the clear command deletes the file too.
**Reason**: Asking 'Hvor nåede vi til sidst?' after a restart had no context — the in-memory chat history was reset on every launch.
**Files**: agent_core/constants.py (CHAT_HISTORY_JSON_PATH), agent.py (_load_chat_history, _save_chat_history, _trim_chat_history, clear_history), .gitignore, USAGE.md, tests/test_tool_loop_nlp.py (+5 tests)


## 2026-08-13 15:47 — security: NLP tools no longer shell-injectable

**Change**: The command-injection surface in the NLP tool loop is closed. git/diff/tests tools switched from shell-string construction (shell=True, raw model args appended) to arg-list subprocess execution — shell metacharacters in model-supplied args are now literal arguments. The run tool (intentionally a shell tool) got a hardened word-boundary, case-insensitive destructive-pattern blocklist replacing the old substring blacklist. Legacy tool_router.py ShellCommandHandler now enforces a binary allow-list plus metacharacter rejection (Windows builtins like echo require the shell, so metachar blocking is the safe middle ground).
**Correction**: the stale project_plan.md claims ('_execute_nlp_tool' in implement_cmd.py, 'test_implement_safety.py missing') were false — the function is _execute_tool_call, implement_cmd.py never used shell=True, and test_implement_safety.py exists.
**Files**: agent.py (_blocked_shell_command + run/git/diff/tests handlers), tool_router.py (ShellCommandHandler), tests/test_tool_loop_nlp.py (+5 injection/blocklist tests), tests/test_tool_router.py (green)


## 2026-08-13 16:46 — search tool overhaul: no more char-per-line or state-file noise

**Change**: The NLP search tool now returns usable results. Root causes fixed: (1) the handler iterated the result STRING char-by-char (results[:30] on a string) producing one line per character; (2) FileSearcher fell back to a blind os.walk that matched .git logs, mypy_cache .db binaries and chat_history.json. The searcher is now a pure-Python walker that excludes git-ignored state/caches/binary files, returns path:lineno: content lines capped at 50, and resolves relative paths against the agent workspace (never the process CWD).
**Change**: ToolLoopRunner detects consecutive identical tool calls — a repeat is not re-executed; the model gets a note with the previous result, and a third identical call in a row gets a hard 'stop repeating' note. This kills the 14x-identical-search loop seen in real sessions.
**Change**: chat_nlp system prompt now requires verified repo facts (project_*.md marked as historical), verified numbers, and explicitly 'if a search finds nothing in source, state the symbol does not exist — never repeat the search'.
**Files**: agent_core/file_searcher.py (rewrite), agent.py (_resolve_nlp_path workspace scoping, search handler, system prompt), agent_core/llm/tool_loop.py (duplicate detection), tests/test_file_searcher.py (new, 8 tests), tests/test_tool_loop_nlp.py (+2 loop tests)


## 2026-08-13 16:58 — stuck-model guard: forced synthesis after 3rd identical call

**Change**: When the model repeats the exact same tool call three times in a row, ToolLoopRunner now stops the loop immediately and runs the forced tool-less synthesis (dedicated 'you are stuck' note) — it no longer waits for the iteration cap while the model keeps repeating (observed: 8+ identical read calls in a real session despite steering notes).
**Change**: read tool gained optional offset/limit parameters and appends '[truncated — use read with offset=N to continue]' so the model can page through large files instead of re-reading the same first 5000 chars — the root cause of the repeated-read loop in the '_execute_nlp_tool' session.
**Files**: agent_core/llm/tool_loop.py (_STUCK_SYNTHESIS_NOTE, stuck detection), agent_core/tool_schemas.py (read offset/limit), agent.py (read handler pagination), tests/test_tool_loop_nlp.py (+4 tests)


## 2026-08-13 17:18 — REPL search command shares the fixed FileSearcher

**Change**: The REPL \search\ command (Agent.search_file) still used the old duplicated findstr + blind os.walk fallback — matching .git logs, chat_history.json, mypy_cache .db and __pycache__ binaries without line numbers, while the NLP search tool had already been fixed. Agent.search_file now delegates to the shared FileSearcher (exclusions + path:lineno: content), and the dead _search_files/_fallback_search duplicates plus the now-unused platform import were removed.
**Files**: agent.py (search_file delegation, dead code removal), no new tests needed (existing 595 cover the searcher)

## 2026-08-13 18:29 — project_*.md no longer pollutes search results

**Change**: project_*.md (spec/analysis/plan/tasks/entities) are temporary outputs of the workflow command, not source of truth — they are now excluded from search, git-ignored, and untracked. A symbol that only lives in them (e.g. '_execute_nlp_tool') no longer appears as a code match, so the agent concludes 'not in code' instead of chasing a phantom implementation.
**Files**: agent_core/file_searcher.py (_is_ignored_file project_*.md rule), .gitignore (project_*.md), git rm --cached of the five tracked docs, tests/test_file_searcher.py (+1 regression test: the exact _execute_nlp_tool scenario)


## 2026-08-13 18:46 — fix: persisted history no longer anchors new prompts to old topics

**Change**: The agent kept producing the IDENTICAL stale answer regardless of the new prompt. Root cause: the persisted chat_history.json had grown to 60 messages dominated by an old '_execute_nlp_tool' exchange (including loop steering notes and an old user request), and every new prompt was appended to it — at temperature 0.1 the same context yields the same output, so the model re-answered the OLD request every time.
**Change**: History is now PROJECTED before saving/loading: only the system prompt + the exchange starting at the last user message survive; loop steering NOTE tool messages and empty assistant placeholders are dropped. Old topics can never anchor a fresh session again.
**Files**: agent.py (_project_chat_history used by _save/_load_chat_history), tests/test_tool_loop_nlp.py (+2 projection tests, 1 updated)


## 2026-08-13 20:40 — run tool shell awareness + generated test fixes

**Change**: The NLP run tool now detects and reports the active shell. cmd.exe fails Unix-style pipelines (e.g. 'mypy ... 2>&1 | tail -40') silently with rc 255 and no output, and errors like "'tail' is not recognized" previously produced '(no output)'. The tool now appends a hint telling the model it is on cmd.exe (no tail/grep/ls; use 'python -m <tool>' or Python one-liners; output is auto-truncated), both on 'not recognized' stderr and on silent rc-255 pipeline failures. The system prompt in chat_nlp states the detected shell up front.
**Change**: Fixed tests generated by the agent session: test_evolution_metrics.py called a pytest fixture directly (16 failures) and used a method-style helper with (self, metrics); test_llm_executor.py had an unmatched ')' and a fake tool-call handler reading call.function.arguments instead of the ToolCallResult keys tool/arguments.
**Files**: agent.py (_detect_shell, _unix_command_hint, run tool, system prompt), tests/test_tool_loop_nlp.py (+2 shell tests), tests/test_evolution_metrics.py (new, fixed), tests/test_llm_executor.py (extended, fixed)


## 2026-08-14 00:38 — console shows full tool results (display cap 200 -> 1500)

**Change**: Every [result] line in the NLP tool loop was truncated to 200 chars for the human observer, so read/search output looked 'cut off' even though the model receives the full result. Display cap raised to 1500 chars (_RESULT_DISPLAY_LIMIT in tool_loop.py).
**Note**: the model was also reading files in tiny 90-char windows (limit=90), compounding the effect — the read schema's default remains 5000 chars.
**Files**: agent_core/llm/tool_loop.py


## 2026-08-14 11:17 — auto-continue: the agent no longer stops on tool budget

**Change**: The tool loop used to end when its iteration budget ran out (or the model repeated calls), forcing the end-user to say 'continue' over and over — the model cannot predict its own budget. chat_nlp now chains runs automatically: if a run ends on the iteration cap, on stuck repetition, or with an answer that signals unfinished work (marker heuristics like 'budget exhausted', 'would need', 'remaining'), a fresh run starts with a fresh budget and a 'CONTINUE THE TASK' note (max 3 chains as an infinite-loop guard). ToolLoopRunner now exposes termination_reason ('answer'/'cap'/'stuck'); iteration budget raised 20 -> 40. Continuation notes are stripped before persisting history so a finished task is never resumed by a future session.
**Files**: agent.py (chat_nlp chaining, _CONTINUE_NOTE, _looks_incomplete), agent_core/llm/tool_loop.py (termination_reason), tests/test_tool_loop_nlp.py (+4 auto-continue tests)


## 2026-08-14 11:33 — progress-based loop termination instead of call-count caps

**Change**: A fixed iteration budget is the wrong control: a model that spawns many small tool calls (tiny read slices, one-off run probes) exhausts any artificial cap even though it is making real progress. The loop now terminates on PROGRESS, not on call count:
- per-run budget raised to 150 iterations (was 40); ToolLoopRunner.termination_reason adds 'no_progress'
- progress guard: after 30 consecutive non-mutating calls (read/search/list/run/git/diff/tests/analyze) a nudge is injected ('exploration alone does not finish tasks — edit/write/fix or answer'); at 50 the loop stops with forced synthesis; write/edit/fix reset the counter
- auto-continue chains raised 3 -> 6 (safe now: every chained run has its own no-progress/stuck/deadline guards)
- system prompt: 'effectively unlimited budget — do not rush or plan around a budget' and 'never request tiny read slices (limit < 1000)'
**Files**: agent_core/llm/tool_loop.py (MUTATING_TOOLS, nudge/force notes, counter), agent.py (_MAX_CHAINED_RUNS, prompt rules), tests/test_tool_loop_nlp.py (+3 progress-guard tests, capped-chain updated)


## 2026-08-14 14:58 — recovery from stuck state: no chaining after stuck/no_progress

**Change**: A session spun endlessly: after a run ended 'stuck' (repeated identical calls), auto-continue started a fresh run whose duplicate counter was reset — the model re-ran the SAME probe in every run (observed: 5 chained runs of an identical `run(python -c ...)`). Two fixes: (1) chat_nlp now auto-continues ONLY on 'cap' (genuine budget run-out while progressing) or an 'answer' that signals unfinished work — 'stuck' and 'no_progress' are explicit no-progress verdicts and end the turn with a [stopped] hint instead; (2) ToolLoopRunner gained a cross-run `seen_calls` registry (shared across chained runs) so the duplicate note reports the TOTAL executions of a call, flagging repeated probes as known dead-ends even after a restart.
**Change**: Test strings translated to English (project convention).
**Files**: agent.py (continuation policy, seen_calls, [stopped] hint), agent_core/llm/tool_loop.py (seen_calls registry + note), tests/test_tool_loop_nlp.py (+2 no-continuation tests, cap test uses distinct calls)


## 2026-08-14 15:07 — [model: ...] status label printed once per session

**Change**: LMStudioProvider.chat() printed the '[model: X | profile=... t=... tok=...]' label before EVERY LLM call, so a tool loop showed it dozens of times. The label is now printed once per provider instance (session) and only re-printed when it changes mid-session (model/profile/temperature/max_tokens switch).
**Files**: agent_core/llm/lmstudio.py (_last_label dedup), tests/test_lmstudio_label.py (new: once-per-session, profile change, model change)


## 2026-08-14 22:04 — colors in REPL output + sanitizer fix + qwen session cleanup

**Change (agent session)**: New agent_core/colors.py (ANSI helpers, degrades off-TTY/NO_COLOR, __all__ fix) wired into the REPL banner, command list, tool-loop [tool]/[result] lines (colorize_result heuristic) and auto-continue/[stopped] messages. Sanitizer hardened: the malformed pipe pattern `r"|\\s*"` (matched a literal backslash, never pipes) is replaced by `r"|\s*\S.*"` so '| ls -la' is stripped entirely; separators now strip separator+command instead of leaving fragments. +24 tests (tests/test_sanitizer.py).
**Change (cleanup)**: nlp_parser.py was left corrupted by the session (duplicate import re + orphaned IntentType docstring -> IndentationError, breaking the whole agent) — restored. Paste command-list typo fixed ('[--workspace <path>]'). Unused imports removed (config.py Any/_json, agent.py red/colorize_result). tests tool timeout raised 120 -> 300s (the full suite takes ~2.5min; 120s made the agent split runs). tool_loop color fallback simplified. 10 probe temp files at repo root deleted.
**Files**: agent_core/colors.py (new), tests/test_sanitizer.py (new), agent.py, agent_core/llm/tool_loop.py, agent_core/security/sanitizer.py, agent_core/nlp_parser.py, agent_core/config.py


## 2026-08-14 22:39 — fix: QUIET display mode no longer swallows the final answer

**Change**: The display-mode contract says QUIET hides only intermediate tool output ('only the final answer is printed'), but chat_nlp guarded the final print with `display_mode != QUIET` — so in QUIET mode the user saw NO output at all between prompts. The final answer is now printed in every mode; [auto-continue]/[stopped] status lines are gated on non-QUIET (headless/piping cleanliness). Also: .env had AGENT_DISPLAY_MODE=quiet left behind by the qwen session's `display quiet` test — removed so the default VERBOSE applies again.
**Files**: agent.py (final-answer print + status-line gating), .env (runtime, not committed), tests/test_tool_loop_nlp.py (QUIET test now asserts the answer is actually printed)


## 2026-08-15 00:40 — implement --review: 'y' deletes the whole orphaned component

**Change**: Answering 'y' to the review delete prompt only removed the directly flagged files, leaving sibling orphans behind (e.g. a base module whose only importers were deleted) plus empty generated packages. Now the delete set is expanded to its TRANSITIVE CLOSURE (_unwired_closure): a generated file referenced only by other to-be-deleted generated files joins the set; files referenced by surviving generated code or by real project code stay pinned. Empty packages left behind are pruned (_prune_empty_dirs: removes __init__.py markers and rmdirs empty dirs; pre-existing packages with other content are untouched).
**Change**: Remaining Danish comments/docstrings in tests translated to English; two phantom 'fix --mypy' CHANGES entries (referencing a nonexistent agent_20260729_201150.py) removed.
**Files**: agent_core/commands/implement_cmd.py (_unwired_closure, _prune_empty_dirs, delete block), tests/test_implement_safety.py (+4 tests), tests/test_fix_helpers.py (English), agent_core/diff/semantic_parser.py + agent_core/utils/common.py (unused-import cleanups from the workflow's fix phase, verified)


## 2026-08-15 08:58 — prevent superfluous generated files (3 layers)

**Change**: A `workflow . --desc ...` run generated 8 modules that duplicated existing ones (shell_allowlist vs allowlist.py, path_guard vs path_utils.py, tool_schema vs tool_schemas.py, sanitizer_fix/output_sanitizer vs sanitizer.py). Prevention now works in three layers before any code is written:
1. **Taskplan gate (implement)**: planned NEW modules are checked against existing modules before generation — near-duplicates abort the run with a clear message ('extend the existing module instead'); --force overrides. This would have BLOCKED 6 of the 8 files in the failing run.
2. **Pre-generation skip (implement)**: the same check runs per file in the generation loop; duplicates are skipped with a message instead of written.
3. **Prompt policy + module inventory (workflow)**: plan/taskplan prompts now state 'prefer [MODIFY] over [NEW], name the closest existing module' and receive a compact inventory of existing modules (rel path + first docstring line) so the LLM cannot invent near-duplicate names ('shell_allowlist' when 'security/allowlist.py' is listed).
**Change**: detect_module_collisions gains a shared-name-token check (catches path_guard vs path_utils, which fuzzy ratios miss); generic tokens (cmd/core/util/...) are ignored.
**Files**: agent_core/patterns.py (_shared_name_tokens), agent_core/commands/implement_cmd.py (_check_planned_duplicates, Layer 1 gate, Layer 2 skip), agent_core/commands/workflow_cmd.py (_module_inventory, MODULE POLICY in prompt_context + plan prompt), tests (+4)


## 2026-08-15 09:32 — quality gates: analysis verification + MODIFY-filter + wire-in

**A — Analysis verification gate (workflow_cmd.py)**: _write_verified_analysis now returns (text, checked, flagged), and a new _analysis_flag_gate pauses the run for confirmation when ANY code claim could not be verified (fabricated paths/symbols poison the downstream plan); the flagged claims are shown (top 10); --force downgrades the gate to a warning; EOF defaults to halt. Applied at all three analyze branches.
**B — MODIFY-filter in the Layer-1 gate (implement_cmd.py)**: instead of a hard abort on planned near-duplicates, the gate now offers [m]odify-existing (default: drop the duplicate planned files, print 'extend the existing module instead', refresh the cache, continue with the rest), [f]orce-generate anyway, or [a]bort.
**C — Wire-in offer (implement_cmd.py)**: kept-but-unwired modules now get a deterministic consumer suggestion (_suggest_consumers: token matching + concept map + agent.py fallback) and an optional one-shot LLM wiring pass (_wire_in_modules: [PATCH:] blocks applied per patch with py_compile verification, max 3 files, honest 'left for manual integration' reporting). Legacy 'Use model profile use to wire them' message removed.
**Files**: agent_core/commands/workflow_cmd.py, agent_core/commands/implement_cmd.py, tests/test_workflow_cmd.py (+5), tests/test_implement_safety.py (+4), tests/test_analysis_verifier.py (updated for tuple return)


## 2026-08-15 11:29 — workflow docs move to .docs/<timestamp>/ (git-ignored, root no longer polluted)

**Change**: Workflow pipeline docs (spec/analysis/plan/entities/tasks) are no longer written to the workspace root. Each `workflow` run creates a fresh `.docs/<YYYY-MM-DD_HH-MM-SS>/` folder (one per run, so runs stay apart) and writes all its docs there; the same applies to bare output filenames of the standalone `plan`/`entities`/`taskplan` commands (explicit paths are still respected). Readers (`implement`, `fix`, `decide extract`) resolve doc names against the newest run folder first, with the workspace root kept as a legacy fallback — so `implement project_tasks.md` still works. The skip-if-exists logic now carries docs over from the previous run folder (copied into the new one, so each run folder stays self-contained). `.docs/` is git-ignored and excluded from file search.
**Migration**: the five existing root `project_*.md` files were moved to `.docs/2026-08-15_11-17-11/` (one-off, not automatic).
**Files**: agent_core/commands/doc_paths.py (new), agent_core/commands/workflow_cmd.py (run folder + carry-over), agent_core/commands/plan_cmd.py, entities_cmd.py, taskplan_cmd.py (bare output -> .docs, input fallback), agent_core/commands/implement_cmd.py (input fallback), agent_core/commands/fix_cmd.py (find_doc), agent_core/commands/decide_cmd.py (extract fallback), agent_core/file_searcher.py (.docs excluded), .gitignore (.docs/), tests/test_doc_paths.py (new, 16 tests), tests/test_file_searcher.py (+1)


## 2026-08-15 11:52 — geometric duplicate detection (TF-IDF + optional embeddings)

**Change**: New agent_core/utils/module_similarity.py adds a precision-first semantic layer to the planned-module gates: TF-IDF cosine over module docstrings + task-plan descriptions (the strongest signal), with an optional LM Studio /v1/embeddings backend (AGENT_EMBEDDING_MODEL; probed once, never silently degrades — evidence reports which signal fired). Calibrated against ground truth from the failing workflow run: true pairs score 0.64-0.71, false pairs 0.27-0.49, threshold 0.55 with self-exclusion and a production-only corpus (tests/benchmarks excluded). _check_planned_duplicates now combines name gates + geometric findings, each with evidence ('TF-IDF top-1 0.704 → path_utils.py'). Corpus is cached per workspace, invalidated by max-mtime (identical precision, faster repeats). Embeddings A/B-verified against the same fixtures before activation.
**Files**: agent_core/utils/module_similarity.py (new), agent_core/commands/implement_cmd.py (gate integration + task descriptions), tests/test_module_similarity.py (new, 9 tests), tests/test_implement_safety.py (+1 semantic-gate test)


## 2026-08-15 12:19 — spec-aware carry-over in .docs workflow runs

**Change**: The .docs/<timestamp>/ run-folder carry-over copied plan/entities/taskplan from the PREVIOUS run unconditionally — a new `workflow --desc <different task>` without --force silently reused the old task's plan and would have implemented the wrong files. _reuse now only carries over when the previous run's project_spec.md content matches the current spec (true resume); a different spec prints 'Previous run ... has a DIFFERENT spec — regenerating (no carry-over)' and regenerates all docs. New module-level _specs_match helper (tested).
**Files**: agent_core/commands/workflow_cmd.py (_reuse gate + _specs_match), tests/test_workflow_cmd.py (+3 tests); stale .docs/2026-08-15_12-03-09 run folder (spec-only, wrong-task leftovers) removed


## 2026-08-15 12:56 — autonomous mode + tailored next command

**Change**: One autonomy layer in agent_core/commands/base.py — is_autonomous() (AGENT_AUTONOMOUS env, read at call time), set_autonomous() (per-command --auto flag overrides env, None clears), and auto_choice() (interactive = read_input as today; autonomous = safe default without prompting, auto_default is ALWAYS the safe option — never auto-approval).
**Change**: workflow --auto: the 'Next: implement ...' message is now TAILORED to the current codebase (_tailored_implement_parts): positionals in implement's order incl. analysis, a duplicate pre-check via the existing gate (warning + no --force so the MODIFY filter can run), and --keep only on a matching implement cache. After the hint, 'Run this command now? (y/N)' — default N interactively, auto-runs inline in autonomous mode (_run_next wraps ImplementCommand exactly like the REPL, incl. flow-stop). The analysis verification gate auto-DENIES in autonomous mode. All three Next sites unified via _offer_next.
**Change**: implement's three interactive gates (Layer-1 options, delete y/N, wire-in y/N) now use auto_choice with safe auto-defaults (m/n/n) — interactive behaviour is byte-identical; only autonomous runs change.
**Files**: agent_core/commands/base.py, agent_core/commands/workflow_cmd.py, agent_core/commands/implement_cmd.py, tests/test_autonomy.py (new, 11 tests)


## 2026-08-15 15:12 — fix --desc

**Change**: Modified `implement_cmd.py`
**Reason**: In agent_core/commands/implement_cmd.py, EVERY call to agent.llm.chat(...)
must pass disable_thinking=True, exactly like fix_cmd.py and optimize_cmd.py
already do. The model currently burns the whole 

## 2026-08-15 15:12 — fix --desc

**Change**: Modified `decisions.py`
**Reason**: In agent_core/commands/implement_cmd.py, EVERY call to agent.llm.chat(...)
must pass disable_thinking=True, exactly like fix_cmd.py and optimize_cmd.py
already do. The model currently burns the whole 

## 2026-08-15 15:44 — canonical workspace paths + implement length-fix verification

**Change (paths)**: Relative paths are now canonicalized against the WORKSPACE (never the process CWD) before any processing, with the absolute form always derivable. decisions.py: normalize_affected_files stores affected_files as canonical workspace-relative forms at record time (non-existent/escape entries like '../ReactAgent' dropped; doc basenames fall back to the newest .docs/<ts>/ run); load_decisions normalizes legacy entries so matching is on ONE form; find_decisions and find_overlaps match canonically (absolute caller paths included). doc_paths.find_input now resolves relative inputs against the workspace first and always returns an absolute path. decide_cmd --from/default resolves via find_input. .decisions.json persisted with canonical forms (#003-006; '../ReactAgent' removed, 'project_spec.md' resolved into .docs/2026-08-15_12-06-06/).
**Change (implement length fix, from the fix --stdin run)**: verified and repaired — all 8 agent.llm.chat call sites now carry disable_thinking=True (generation, file-list, mypy retry, fix applications, review, wire-in), the retry loop appends an 'Answer immediately — no reasoning' note on reasoning errors, and the fix pipeline's <tool_call>-token stripping was repaired (system prompt prohibition and the tool-call detection condition restored; a stripped token had turned the condition into a permanent true). check_contradictions/resolve_contradictions intentionally keep thinking enabled (analysis quality).
**Tests**: test_decision_paths.py (new, 12): canonical_rel, normalize_affected_files, find_input contract, decision matching; test_autonomy.py: meta-guards asserting every implement/decisions-extraction chat call carries disable_thinking=True and that <tool_call> tokens survive; test_doc_paths.py updated to the always-absolute contract.


## 2026-08-15 17:59 — web_search NLP tool + MODIFY-rewrite guard

**Change (web_search)**: new agent_core/tools/web_search.py — dependency-free DuckDuckGo client (urllib + html.duckduckgo.com, adapted from ReactAgent), query-only (decision #006, no URL fetching / SSRF), results formatted with an [UNTRUSTED WEB CONTENT] marker (decision #005). web_search schema appended to NLP_TOOL_SCHEMAS (auto-derives into NLP_TOOL_NAMES) and a handler branch added to agent.py::_execute_tool_call (query required, max_results 1-10, output capped). Live-verified against DuckDuckGo.
**Change (guard)**: implement now REFUSES wholesale rewrites of existing files in its MODIFY path — the broken run replaced tool_loop.py's 416 lines with a 4-line stub and gutted tool_schemas.py; a similarity < 0.5 on an existing file rejects the batch with a clear message (--allow-rewrite opts in). Truncation-retry path guarded too.
**Change (recovery)**: the broken implement run's damage was reverted (tool_loop/tool_schemas/tool_dispatcher/tools __init__ restored from HEAD; stray agent1/agent.py and .generated_manifest.json removed). The rerun of the user's implement command hung silently for 45 min and was aborted — web_search was then implemented directly and verified instead.
**Files**: agent_core/tools/web_search.py (new), agent_core/tool_schemas.py, agent.py, agent_core/commands/implement_cmd.py (--allow-rewrite guard), tests/test_web_search.py (new, 11 tests)


## 2026-08-15 22:07 — multi-provider LLM architecture + opencode-go provider

**Change (architecture, decisions #007-#012)**: agent_core/llm/provider.py is the ONE abstraction — provider_for() selects by model prefix (opencode-go/... → opencode) then the llm_provider setting; build_provider() is the configured factory. AgentSettings gains llm_provider, opencode_server_url, opencode_password, opencode_model (env + .env fallbacks); LM Studio remains the default. resolve_model is provider-aware (no LM Studio query for opencode) and persist_model_choice stores the provider in model.json. LLMClient builds its provider via the factory.
**Change (opencode provider)**: new agent_core/llm/opencode_provider.py — talks to `opencode serve` (port 4096, optional basic auth) implementing the SAME chat contract as LM Studio (plain text or OpenAI-style tool_calls JSON, [Error ...] strings, temperature/max_tokens/_profile_name attributes). opencode's message API executes its own built-in tools server-side, so the agent's tool loop is text-mediated: pending tool parts map via a fixed table (bash→run, read→read, write→write, edit→edit, list→list_files, grep→search, webfetch→web_search) and tool results are fed back as text parts (decision #011 — LM Studio knobs never reach other providers).
**Change (model command)**: `model list` now shows the models of EVERY LLM provider — [opencode] models from the server (/config/providers) plus [lmstudio] VRAM/models — with active-provider markers; `model provider [lmstudio|opencode]` switches the provider; `model name opencode-go/<m>` selects an opencode model directly.
**Files**: agent_core/config.py, agent_core/llm/provider.py, agent_core/llm/opencode_provider.py (new), agent_core/constants.py, agent.py (LLMClient), agent_core/commands/model_cmd.py, .env.example, tests/test_opencode_provider.py (new, 14 tests)

## 2026-08-17 — feat: implement --modify diff-apply mode for existing modules + tests

**Change**: `implement <taskplan> --modify` merges generated content into existing compile-OK `.py` modules as a reviewed unified diff instead of skipping them (default `--keep`) or overwriting them wholesale (`--force`). The diff is produced via difflib, applied through the shared tolerant patch_utils machinery (strict + anchored fallback), py_compile verified, shown with show_file_diff, and applied only after approval — safe auto-default declines in autonomous mode. Wholesale rewrites are rejected unless similarity ≥ 0.5; `--allow-rewrite` opts in. Unchanged content is skipped; non-.py files and unreadable files are skipped with a note. Flow-stop (Ctrl+C / "stop"/"q") honored during the approval prompt.

**Reason**: The gap between skip (`--keep`) and overwrite (`--force`) left no middle ground for extending existing modules — `--modify` lets the LLM propose a minimal reviewed diff so the user sees exactly what changes before it lands, with compile + similarity guards preventing accidental gut rewrites (the broken run that replaced tool_loop.py's 416 lines with a 4-line stub is now impossible).

**Files**: agent_core/commands/implement_cmd.py (`_apply_modify_diff`, modify_mode flag, modify_target gate in the generation loop, help text), tests/test_implement_safety.py (`TestModifyMode`: apply-after-approval, decline-without-approval, unchanged-skip, wholesale-rewrite-rejected, allow-rewrite-applies)

## 2026-08-17 — fix: verifier symbol-line pairing no longer bleeds across bullet points

**Change**: `agent_core/commands/analysis_verifier.py` `_pair_symbol_with_line` paired a verified symbol with the nearest `line N` claim by absolute offset distance (≤200 chars) on the same file — crossing analysis bullet boundaries and contradicting `_iter_lines`'s "avoiding bleed between bullet points" design. A `line ~86` mention in one bullet could be falsely attributed to a `_safe_path` symbol verified in an adjacent bullet, producing misleading `[UNVERIFIED] \`line 86\` — _safe_path is defined at agent.py:554 (claimed line 86)` reports even though the two claims were unrelated. Pairing now requires `segment_start` equality — symbols and lines must share the same source segment before any mismatch flagging; distance filtering removed in favor of exact-segment scoping. `_Claim` gains a `segment_start` field populated by every claim constructor in `_extract_claims`.

**Reason**: The verifier's stated design scopes file/line context only to claims on the same analysis line, but `_pair_symbol_with_line` ignored that and paired globally — turning unrelated adjacent bullets into false positives (observed: 6 flagged where 5 were genuine against real source). Segment-scoped pairing restores intent.

**Files**: agent_core/commands/analysis_verifier.py (`_Claim.segment_start`, all constructors in `_extract_claims`, `_pair_symbol_with_line` rewrite), tests/test_analysis_verifier.py (`test_symbol_line_does_not_bleed_across_bullets` new; fixture gains `FileSystem.read`/`_safe_path`; `_flagged_lines` helper)

