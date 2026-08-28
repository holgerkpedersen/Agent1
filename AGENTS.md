# AGENTS.md — Agent1 (self-improving AI coder agent)

Project memory for AI-agent sessions. Read this first; it encodes hard-won invariants
that cost a deleted `agent.py` to learn.

## What this is

A Python agent system (REPL + workflow pipeline) that uses local LM Studio models
or the hosted opencode-go API (persisted model/provider choice in `model.json`;
currently `qwen/qwen3.8-27b` under `deep-analysis`) to analyze, plan, implement,
and test code changes in its own workspace. It records every design decision and
is being extended to audit its own file effects (self-improvement).

## Architecture map

- `agent.py` — entry point; REPL `run_interactive`; the `Agent` class: `chat_nlp`
  tool-call loop, memory stores (`_files_read`, `_semantic_index`, `_knowledge_graph`,
  `_working_memory`, `_history`, `_file_mtimes`), and persistence via
  `chat_history.json` + `agent_memory.json` (repo root, gitignored).
- `agent_core/commands/*_cmd.py` — registry commands (REPL): `read, write, search,
  analyze, plan, entities, taskplan, implement, fix, workflow, decide, clear, model,
  run, self_heal, optimize, perf, paste, paste_image, display, cleanup, review,
  reconstruct, mode`; `agent_core/commands/plan_verifier.py` — deterministic
  regression gate over generated plan/entities/taskplan docs (decision #076).
- `agent_core/symbol_intel.py` — pure-AST code intelligence backing the NLP
  `definitions`/`references` tools (signatures + line spans; capped whole-word
  reference search); both are read-only and allowed in plan mode.
- `agent_core/modes.py` — session modes (`build` default, `plan` read-only);
  enforced at schema level AND in `_execute_tool_call` (decision #077).
- `agent_core/llm/tool_loop.py` — `ToolLoopRunner`: NLP tool-call execution loop.
- `agent_core/security/` — sanitizers, command allowlist, secrets store (OS keyring
  + encrypted-file fallback); `agent_core/file_system.py`, `path_utils.py` — real
  path utilities (`to_windows_path`, `normalize_path`, `safe_path`, `resolve_path`);
  `agent_core/commands/freshness.py` — REPL stale-module guard (warns when loaded
  `agent_core`/`harnessfix` files change on disk mid-session); **do NOT invent new ones**.
- `agent_core/mcp/` — MCP (Model Context Protocol) consumer: `jsonrpc.py`
  (typed JSON-RPC 2.0 framing), `transports.py` (Stdio argv-list Popen +
  HTTP/SSE, every request wall-clock capped), `config.py` (`mcp.json`,
  gitignored; secrets only via `secret:<keyring>` / `${VAR}` refs, fail-closed),
  `client.py` (schema-validated tool calls, 20k-char result cap), `manager.py`
  (lock-guarded lifecycle, process-wide singleton). REPL `mcp` command
  (`mcp_cmd.py`) is the ONLY writer of mcp.json; the dashboard `/mcp` page
  can connect/disconnect/call but never read or write config; POST endpoints
  enforce exact-loopback Origin+Host checks. LLM bridge tools
  `mcp_tools`/`mcp_call` fire only for servers with `expose_to_llm: true`
  (default false), re-checked under the manager lock at call time.
- `harnessfix/` — self-improvement consumers: `tracing.py` (`TraceWriter`, ON by
  default, opt-OUT via `AGENT_NO_TRACE=1`; `agent.py` already wires it into the
  live tool loop so every run leaves a trace under `reports/traces/`),
  `reader`, `diagnose`, `gates`, `loop`, `htir`, `links`, `corpus`, `review.py`
  (human verification ledger + diagnosis-pinning regression export), `autoreview.py`
  (evidence-rule auto-labeling behind `review auto`), `history.py` (trace-index +
  execution-ledger queries and PAST EXECUTION NOTES formatters that implement/fix
  inject into prompts).
- `scripts/autonomous_self_improve.py` — fully autonomous self-improvement driver.
  Repeats the HarnessFix `loop` with `auto_approve` (machine-gated: applies a
  repair, commits ONLY if test + security + benchmark gates all pass; never merges
  on ambiguity). Requires `AGENT_AUTONOMOUS=1` (or `--auto`) to engage; halts on a
  `STOP_AUTONOMOUS` file/env kill-switch; leaves a git checkpoint before each
  iteration so any change is a one-step `git revert`.
- `tests/` — pytest, **1503 collected** (~2 min full run with `--no-cov`;
  `testpaths=["tests"]`).
- `agent_core/tests/` — entry-point/component test package (31 tests, runs only when
  targeted: `python -m pytest agent_core/tests -q --no-cov`); reconstructed 2026-08-19
  after the source was deleted uncommitted (decision #058).
- `.decisions.json` — decision ledger (76 records, gaps ok; latest #077).
- `.docs/<timestamp>/` — one folder per workflow run (spec/analysis/plan/entities/tasks).
- `backups/` — implement's pre-run copies of existing targets (timestamped).

## Workflow pipeline

`workflow <target> [--brainstorm] [--desc "..."] [--auto] [--continue] [--force]` →
`.docs/<ts>/project_{spec,analysis,plan,entities,tasks}.md` → prompts to continue
into `implement <tasks> <analysis> <plan> <entities> --workspace . --modify`
(`--auto` runs the tailored implement inline; `--continue` resumes the newest run).

## CRITICAL INVARIANTS (do not regress)

1. **Implement must NEVER delete a pre-existing file it did not write.**
   History (2026-08-18): the post-loop dependency cascade unlinked an untouched
   `agent.py` because it imported a wholesale-rewrite-rejected `tool_loop.py`.
   Now: only `written_files` may be removed; pre-existing importers are KEPT
   (`KEPT: ... existing file left untouched`); originals are backed up to `backups/`
   before any write. Regression tests: `TestDependencyCascadeSafety`.
2. **Wholesale-rewrite guard**: existing files are rejected unless `--allow-rewrite`
   (or `--force`). Only files actually applied count as implemented.
3. **`[FILE:]` names must match the planned batch**; foreign names are ignored with a
   warning (a `secrets.py` batch once returned a `sanitizer.py` block).
 4. **Commit after every session.** Uncommitted work is unrecoverable — the deleted
    `agent.py`'s uncommitted changes existed nowhere (not even git objects).
    History (2026-08-19): the 5 `agent_core/tests/*` files written by the 2026-08-18
    23:18 implement run were deleted before commit; only the `__pycache__/*.pyc`
    survived, and the tests were reconstructed from the marshalled code objects
    (names/docstrings/constants) — decision #058.
5. **Extend existing modules, don't regenerate them.** Planned new modules are checked
   against workspace reality (`_check_planned_duplicates`). Phantom modules exist in
   the wild — planned files that were never written (the original `secrets.py`
   phantom was later implemented as the 2026-08-17 secret manager; the check still
   guards against new phantoms).
6. **`_ensure_package_inits` must skip `tests/` and `src/` trees** (PEP 420 namespace
   packages; `tests/__init__.py` breaks pytest sibling imports).
7. **Memory persistence**: `agent_memory.json` holds files-read/semantic-index/
   knowledge-graph/working-memory; `clear` deletes both `chat_history.json` and
   `agent_memory.json`. Mtimes are session-scoped (never persisted).
8. **Chat history projection** keeps system prompt + bounded multi-exchange window
   (60 msgs) — the old last-exchange-only projection was removed deliberately.

## Verification commands

- Full suite: `python -m pytest -q --no-cov` (~3.5 min; use `--no-cov` for speed).
- Targeted: `python -m pytest tests/test_implement_safety.py tests/test_tool_loop_nlp.py -q --no-cov`.
- mypy: `python -m mypy <file>`. **Known baseline: 22 pre-existing errors in 6 files**
  (implement_cmd.py 14; reconstruct_cmd.py 2; self_heal_cmd.py 2;
  security/secrets.py 2; cleanup_cmd.py 1; demo_data_cmd.py 1).
  Do not silently "fix" them; do NOT introduce new errors.
- ruff config lives in `pyproject.toml` (`[tool.ruff]`); CI runs
  `ruff check agent_core harnessfix fixcommand tests performance_dashboard`
  with a **blocking** `--select F821` gate plus an advisory full-lint step.
- A local pre-commit hook (`.git/hooks/pre-commit`, tracked copy in
  `.githooks/pre-commit`) mirrors CI: it blocks commits that introduce
  F821/parse errors in staged files and prints the advisory lint. Install once
  with `git config core.hooksPath .githooks` (or copy
  `.githooks/pre-commit` to `.git/hooks/`). Run `ruff check --fix` yourself
  before committing if you want the advisory issues cleaned too.
- Implement auto-runs `py_compile` on every written file.

## Conventions

- Commit style: short `fix:` / `feat:` / `docs:` subjects (see `git log`).
- **Every bug fix ships with a regression test** (e.g. `TestDependencyCascadeSafety`,
  `TestPersistentMemory`, `TestAnalysisVerifier`).
- Decisions are recorded via the `decide` step; candidates carrying unverified claims
  require explicit confirmation before recording.
- Windows shell is `cmd.exe` — no grep/tail; use `python -c "..."` one-liners inside
  the agent REPL; normalize paths via `to_windows_path`.
- Analysis claims are verified against real files/symbols (`analysis_verifier`):
  typed annotated attrs (`self._x: T = ...`), segment-scoped symbol resolution,
  dotted-module references all count as verified.
- **No emojis or pictographs in repo text files** (decision #079). Plain-text
  status markers instead (`[DONE]`, `[Q]` quick win, `[S]` strategic). Monochrome
  CLI glyphs are exempt (check/cross/warning marks, box drawing — they are
  load-bearing terminal output asserted by tests). Enforced as audit check 6:
  `python scripts/audit_invariants.py` fails on findings (`agent_core/text_policy.py`).
- Runtime state files (`chat_history.json`, `agent_memory.json`) and the
  `.docs/`, `backups/`, `reports/` trees are exempt from content scans.

## Issues ledger (.issues.json) — systematic, autonomous-handled work

Decisions live in `.decisions.json` (*why*); concrete, locatable work items
live in the committed `.issues.json` (*what to fix*). The two ledgers never
overlap. The autonomous driver consumes `.issues.json` and may work items
incrementally as its capability grows — human stays in control.

- **REPL command `issue`** (`agent_core/commands/issue_cmd.py`): `issue add
  <category> <file:line> [--title ...] [--approach ...] [--level N]
  [--severity S]`, `list`, `show <id>`, `resolve <id> [resolved|deferred|
  wontfix]`, `promote <id> <0|1|2>`, `autonomy`. Mutating subcommands are
  blocked in plan mode.
- **Collector** `scripts/collect_issues.py`: scans the repo (excluding
  `reports/ backups/ .docs/ generated/` etc.) via two detectors in
  `harnessfix/issue_loop.py` — `duplication` (duplicate/unreachable `except`
  handlers) and `best-effort-except` (inline log-and-swallow `except Exception:
  logger.<level>(..., traceback.format_exc())`). Seeds `.issues.json`
  idempotently (stable ids from category+location), never overwriting a
  human-set status/level. Safe to run in pre-commit/CI.
- **Autonomy levels** (per issue): `0` human-only, `1` auto-safe (tests +
  security gates pass), `2` benchmark-required (explicit `issue promote`).
  New issues default to `1`. The driver's `AGENT_AUTONOMY_LEVEL` env caps what
  it may touch; raise it gradually as categories prove safe.
- **Resolution engine** `harnessfix/issue_loop.resolve_issue`: verifies via the
  SAME detector that raised the issue (acceptance = detector no longer flags
  its files), generates the fix through the existing `fix` command (so the
  AGENTS.md file-safety invariants hold), then runs the existing
  `harnessfix.gates` (test + security + optional benchmark). Fail-closed: any
  ambiguity leaves the tree untouched and stops (no merge). The autonomous
  driver (`scripts/autonomous_self_improve.py`, `--source issues|catalog|both`)
  commits ONLY the issue's files plus `.issues.json`, with `STOP_AUTONOMOUS`
  and a per-iteration git checkpoint intact.

## Roadmap (recorded, in .decisions.json / .docs/2026-08-18_10-45-11/)

- #047 — sanitize shell commands and file contents before trace persistence.
- #048 — instrumentation invisible unless tracing enabled (`AGENT_NO_TRACE=1`).
- **#049 — files-affected recording per tool/nlp (DONE)**: `ToolLoopRunner`
  gains `effects_fn` (`(tool_name, args) -> [paths]`); `tool_result`/`tool_error`
  events carry `affected_files`; only invoked when a trace sink exists. `Agent`
  arms `_pending_effects` only while a `TraceWriter` exists, notes
  read/write/edit/fix targets. REPL registry **commands** still uninstrumented
  (next increment); trace consumers (`harnessfix/reader`) not yet reading
  `affected_files`.
- **#050-#056 — verification gate increment (DONE, 2026-08-18)**:
  - #050 — traces self-describing: `TraceWriter(meta={model, profile})` stamps
    every record; `task_begin` event carries the user prompt (`PROMPT_CAP=500`);
    `chat_nlp` wraps the loop in `CorrelationIdContext`; dashboard shows
    prompt/model/profile/affected-files per task.
  - #051 — collision guard no longer self-blocks: `GUARD_TEST_FILENAMES`
    (guard fixture tests) excluded by default; hits recorded as
    `ignored_guard_test_hits`; real pinning tests still block.
  - #052 — interrupted runs (no `loop_end`, >=3 events) count as failed;
    abandonment diagnosis uses `affected_files` ("task ended non-completed
    after mutating N file(s)"); stuck mechanism names the repeating tool.
    Refined: guard-terminated runs (stuck/cap/no_progress) that still
    delivered a substantive final answer count as DELIVERED, not failed
    (`TraceGraph.has_final_answer`); diagnosis signatures never match inside
    `tool_result` text (file contents — a read mentioning "truncation"
    caused a bogus context diagnosis on task a669a26e...).
  - #053 — **human verification gate**: `harnessfix/review.py` + REPL `review`
    command (`refresh/list/show/label/export`); ledger
    `reports/harnessfix/review.json` (gitignored); dispositions
    bug|regression|noise|ok; `export` writes diagnosis-pinning pytest files.
  - #054 — `decide review`: `find_stale_decisions` (non-mutating) flags
    decisions whose `affected_files` no longer exist; reports open
    contradictions.
  - #055 — benchmark keyed `model|profile` (`--profile` flag, gate reads
    `model|profile` key, list-form report parsing fixed); `--max-tokens`
    default 2048 (reasoning models starved at 512 → empty content).
    **Baseline: qwen/qwen3.8-27b|deep-analysis = 84.7%** (coding 93.3,
    avg 12.3s).
  - #056 — `scripts/audit_invariants.py` (git-dirty, paired memory files,
    phantom modules from latest `.docs/`, trace health, backups/; `--strict`
    escalates git-dirty to ERROR).
- **#060 — history-assisted implement/fix (DONE, 2026-08-19)**: new
  `harnessfix/history.py` builds a process-cached index over `reports/traces/`
  plus a structured execution ledger `reports/history/executions.jsonl`
  (gitignored) and renders compact PAST EXECUTION NOTES blocks. `implement`
  injects them per batch (next to the decisions block; `--no-history` opt-out)
  and appends a structured summary per run; `fix` injects per-file history in
  both `_fix_traceback` and `--desc` modes and appends run summaries.
  Recording got richer too: `search`/`list_files`/`analyze` now record
  `affected_files` in traces (read/write/edit/fix already did). Matching
  handles old-format traces via args-path suffix (abs→rel) and new-format
  `affected_files`; directory args only match direct children to avoid
  workspace-wide noise. 20 tests in `tests/test_harnessfix_history.py`.
  - **Manual next steps for the user**: `review refresh` to build the ledger
    over the ~64 real traces; label the first batch; the benchmark gate now
    works with the qwen3.8-27b baseline in `reports/benchmark_harnessfix.json`.
- **Trace-based file recovery (DONE, 2026-08-20)**: `reconstruct
  [--start <file>] [--end <file>] [--search <query>] [--dry-run] [--force]`
  scans `reports/traces/*.jsonl` for write/edit tool ops, groups them by target
  path, and replays them in timestamp order to rebuild the final state of each
  file — the recovery path for the #058 incident (pyc reconstruction was the
  fallback when it happened). Edits whose `old_text` no longer matches are
  skipped with a warning.
- **#077 — Plan mode (opencode-style session modes, DONE, 2026-08-24)**:
  `agent_core/modes.py` defines `build` (default) and `plan` (read-only
  research) session modes; the `mode` REPL command switches them. In plan
  mode the NLP tool loop only offers the verified read-only tools
  (`search`, `read`, `list_files`, `diff`, `web_search`) — mutating schemas
  are filtered out of the LLM toolset AND rejected at `_execute_tool_call`
  (the choke point shared by `chat_nlp` and `multillm`), so no file changes.
  A system-prompt suffix + per-turn note steer the model to end with a plan
  as text. Tests: `tests/test_plan_mode.py`.
- **#076 — Regression gate for generated plan docs (DONE, 2026-08-24)**:  `agent_core/commands/plan_verifier.py` runs deterministic checks over
  freshly generated `plan`/`entities`/`taskplan` docs (zero LLM tokens):
  backticked paths must exist or be marked new (`[NEW]` tag / create-add
  wording), `[MODIFY]` targets must already exist, entities python fences must
  parse (`ast.parse`) with unique top-level names, taskplan-referenced existing
  modules must not duplicate top-level definitions in the same directory.
  Findings are appended as a `## Verification Report` (analysis_verifier
  style); flagged docs pause for confirmation unless `--force`, and autonomous
  mode auto-DECLINES (safe default). Wired into all workflow inline write
  sites via `_plan_doc_gate()` and into the standalone plan/entities/taskplan
  commands. Tests: `tests/test_plan_verifier.py`.
- **agent.py DRY refactor (2026-08-24)**: monolithic `_execute_tool_call`
  if-chain replaced by `_nlp_tool_handlers()` dispatch table — one small
  `_nlp_*` method per tool; shared helpers `_truncate_output`,
  `_run_subprocess_captured`, `_shape_run_stderr`, `_save_verify_note`,
  `_run_command_quietly`, `_effective_ws_dir`; `_SYSTEM_PROMPT` hoisted to a
  module constant; REPL banner derives its command list from the registry
  itself (`_build_registry`) so it cannot drift; both dashboard entrypoints
  share `_build_dashboard`. Handler exceptions are contained per call (a bad
  tool call returns an error string instead of killing the turn). Tests:
  `tests/test_agent_improvements.py`, `tests/test_agent_dry_refactor.py`.
- Taskplan-time phantom-module gate (plan-time existence check for planned files).
- **Agentic quick-win batch 2 (DONE, 2026-08-25)**: #6 post-mutation
  self-review note (`chat_nlp` prints `[self-review] <files>` after turns
  whose write/edit results carry py_compile verification lines; extraction
  via `_mutating_files_this_turn`, display via `_print_self_review_note`)
  and #19 uncommitted-changes reminder at every REPL shutdown path
  (`_warn_uncommitted`: invariant-#4 nudge listing up to 5 paths when
  `git status --porcelain` is non-empty; silent on clean repos / outside
  git). Tests: `tests/test_quickwins_batch2.py` (12). Full suite:
  1482 passed / 2 skipped.
- **#8 symbol-level tools (DONE, 2026-08-25)**: new `agent_core/symbol_intel.py`
  + two NLP tools: `definitions(path)` (every class/function with compact
  signature and line span via pure AST) and `references(symbol, max_results)`
  (capped file:line list of uses across workspace .py files; whole-word,
  attribute-aware matching so `run` never hits `run_interactive`; oversized
  files skipped). Both read-only → in `PLAN_MODE_TOOLS`. System prompt
  steers the model to prefer them over grep+read paging. Tests:
  `tests/test_symbol_intel.py` (17). Full suite: 1501 passed / 2 skipped.
- **Agentic quick-win batch (DONE, 2026-08-25)**: improvement plan in
  `docs/AGENTIC_IMPROVEMENT_PLAN.md` (19 audited items, progress log there).
  Landed: #1 decisions block injected into the chat_nlp system message
  (`_decision_constraints_block`, rebuilt per turn via
  `_strip_dynamic_system_blocks` so blocks never accumulate); #5 char-budget
  chat-history trimming (`_HISTORY_CHAR_BUDGET` = 75k chars + the 60-message
  cap; oldest-first contiguous trim, compaction note, assistant/tool pairs
  never split); #7 LM Studio now retries transient HTTP 429/5xx
  (`TransientHTTPError` from `_open_chat`, matching opencode's taxonomy);
  #14 plan-mode answers persisted to `.docs/<ts>/plan_proposed.md`;
  #15 `multillm --synthesize`; #18 db_io duplicate imports +
  `llm/config.ProfileType` unified onto `llm_types`. Tests:
  `tests/test_llm_retry_policy.py`, `tests/test_quickwins_2026_08_25.py`.
  Full suite: 1470 passed / 2 skipped.
- **No-emoji policy + audit gate (decision #079, DONE, 2026-08-25)**: new
  `agent_core/text_policy.py` (stdlib-only emoji/pictograph detector with an explicit
  monochrome-glyph allowlist; `scan_tree` skips runtime-state files) wired into
  `scripts/audit_invariants.py` as check 6 (findings are ERRORS). Cleaned AGENTS.md
  ([DONE] markers), the improvement plan ([Q]/[S] tags) and repaired a mojibake
  byte in CHANGES.md. Regression found by the new tests: `_mutating_files_this_turn`
  scanned restored history from previous sessions — fixed with a per-turn boundary
  (`Agent._turn_start_index`). Tests: `tests/test_text_policy.py` (26),
  `tests/test_quickwins_batch2.py::TestTurnBoundaryAfterRestart` (3).

## Git / remote auth (non-interactive)

`git push`/`ls-remote` must NOT prompt for credentials (no human at the keyboard).
Auth is supplied by a local credential helper that reads `GITHUB_TOKEN` from the
gitignored `.env` — the token is never written into `.git/config` or the remote URL.

- Helper: `scripts/git_credential_helper.py` (reads `.env`, emits `git`/token for
  `protocol=https host=github.com`).
- Wired at repo scope in `.git/config`:
  `credential.helper=!D:/Dev/Agent1/scripts/git_credential_helper.cmd` (placed
  before the global `manager`), plus `credential.interactive never`.
- If a push hangs on a credential prompt, run `git config --local --get-regexp
  credential` to confirm the helper is present, and verify `.env` has a live
  `GITHUB_TOKEN`.
