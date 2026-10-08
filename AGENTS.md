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
- `agent_core/tools/git_merge.py` — the NLP `merge` tool's state machine
  (`status`/`start`/`continue`/`abort`/`quit`). Owns merges instead of
  forwarding a string to the generic `git` tool because `git merge --continue`
  blocks forever on an editor with no `GIT_EDITOR`, and rejects every argument
  (`--continue expects no arguments`). Always runs non-interactively
  (`GIT_EDITOR` no-op + `GIT_MERGE_AUTOEDIT=no` + `stdin=DEVNULL` + hard
  timeout), inspects real state (`MERGE_HEAD`, unmerged paths) so no-op calls
  answer with guidance rather than a wasted git call, and is mutating — so
  plan mode blocks it.
- `agent_core/modes.py` — session modes (`build` default, `plan` read-only);
  enforced at schema level AND in `_execute_tool_call` (decision #077).
- `agent_core/commands/speculate_cmd.py` — REPL `speculate`: N independent
  speculative branches run on the thread pool, each carrying the agent's LIVE
  system prompt (`_chat_history[0]` after `_refresh_system_message`) and a
  bounded READ-ONLY tool loop (`_BRANCH_TOOLS` = search/read/list_files/
  definitions/references/web_search; any other name is refused BEFORE the
  executor, so parallel branches cannot mutate) to ground its answer; a judge
  LLM scores each and `ProbabilisticOrchestrator.run_speculative` COMMITs the
  best at `--threshold` or REFUSEs. Branch-dispatch timeout default 300s.
  A branch whose tool-call syntax leaks through as TEXT (e.g. gemma's
  `<|tool_call>call:run{...}`) is failed, not committed, and the judge scores a
  non-answer 0.0 without asking the model. `--judge jev` scores with the
  dedicated small model while the reasoning model still generates the branches
  (the smart split: big model thinks, small model decides); `--branch-model jev`
  also generates the branches on the Jev model (cheap/offline, but a 1.5B
  cannot ground a repo question — expect REFUSE); `--escalate` opts into
  gray-band escalation to the chat-model judge; `--judge both` averages the
  two judges. The header prints `branch_model=` / `jev_model=` / `escalate=`.
  Grounding: a repo/code question (`_is_repo_question`) requires every branch
  to execute at least one tool whose EVIDENCE mentions a subject term
  (`_question_subject_terms` / `_grounded_in_subject`; `grounding=on`;
  `--no-grounding` disables) — calling an unrelated tool does not count — and
  the answer must carry a `file:line` citation. Corroboration: `--agree N`
  (default 2) requires N independent branches to cite the SAME evidence files
  (`_evidence_signature`) before COMMIT. Calibration: `--threshold auto` uses
  the threshold fitted from labeled outcomes (`jev label <id>
  correct|incorrect` -> `harnessfix/jev_telemetry.load_suggested_threshold`).
  Before COMMIT, every `file[:line]` claim in the winning answer is verified
  against the workspace (`_verify_claims`: missing file / line past EOF / bare
  basename resolved by rglob / a backticked symbol next to the citation must
  appear within ±3 lines of the cited line) and the command REFUSEs on a
  mismatch — all guards are deterministic and free. Also exposed as the
  read-only NLP tool `speculate` (LLM calls it on demand via
  `Agent._nlp_speculate`; JSON args question/branches/judge/branch_model/
  threshold; output is captured into the tool result, nothing leaks to REPL
  stdout) and in `PLAN_MODE_TOOLS` — branches can never mutate. Field notes
  from live runs on local providers: a SLOW local chat model can push a full
  deliberation past the default 600s branch-dispatch timeout (every branch
  carries the whole system prompt plus several tool round-trips) — the run
  times out instead of committing; and a small local Jev model cannot ground
  repo questions (`branch_model='jev'`), so expect REFUSE via the
  deterministic guards (fabricated tool output / ungrounded claims) rather
  than a commit. That is by design: refuse over fabricate — use `speculate`
  for fast corroborated repo answers on providers where chat-mode branches
  finish in seconds, not minutes.
- `agent_core/jev_engine.py` — the Jev decision engine: a TYPED, probabilistic
  micro-decision (`yesno` -> P(yes)/P(no)/TRUE-FALSE-UNKNOWN, `choice` ->
  distribution over options/argmax-UNDECIDED, `score` -> 0-100 + spread) run on
  a DEDICATED small model (`settings.jev_model`, default
  `qwen2.5-coder-1.5b-instruct` via catalog `_defaults.jev_model`; env
  `AGENT_JEV_MODEL`) — it builds its OWN   provider with `single=True`: ONE
  pinned provider, NO failover chain, and it never touches `agent.llm`. (A
  failover chain was the bug: a Jev call silently drifted to DeepSeek/
  OpenRouter/the 27B when LM Studio was unreachable.) Every prompt carries the
  current local date/time and a NAMED time of day (`_now_context` /
  `_time_of_day`: "time of day: night"), so time-dependent questions
  ("is it evening?") get a real probability instead of a guess. Before the
  first request
  the engine AUTO-SELECTS its model (`LMStudioProvider.ensure_model_loaded`,
  run once per engine) so a `model` switch that evicted the small model from
  VRAM cannot break a Jev command. NOTE: pick a
  NON-thinking model — `google/gemma-4-e4b` is a
  THINKING model whose reasoning-off knobs do NOT work (it needs
  `jev_max_tokens` >= 512 to reach the verdict, and it is a bad speculate
  branch model). `qwen2.5-coder-1.5b-instruct` answers terse and verified 5/5
  on every primitive. Mechanisms: `logprobs` (one call via
  `LMStudioProvider.chat_logprobs`) and `vote` (N samples;
  agreement IS the probability), `auto` = logprobs then vote. Provider errors,
  timeouts, empty output, unparseable tokens and leaked tool calls are
  ABSTENTIONS (majority -> UNKNOWN), never fabricated votes. Consumers: REPL
  `jev` (`agent_core/commands/jev_cmd.py`) and the read-only NLP tool
  `jev_decide` (`PLAN_MODE_TOOLS`). Configured with `model jev [<name>]`
  (shows the resolved provider + LM Studio status; `model jev <name>` persists
  `AGENT_JEV_MODEL` to model.json + .env) — the Jev model is independent of
  the selected chat model and `speculate` prints `jev_model=<name>` in its
  header so the judge model is visible before the branches run.
- `harnessfix/jev_telemetry.py` — Jev decision telemetry + calibration: every
  `JevEngine.decide` appends one record (source/model/kind/mechanism/
  probabilities/decision/threshold) to `reports/history/jev.jsonl`
  (gitignored, same tree as the execution ledger); best-effort (never breaks a
  decision) and opt-out via `AGENT_NO_JEV_LOG=1` (tests set it so fake
  providers never pollute the ledger). `record_outcome(id, correct|incorrect)`
  labels a decision (REPL: `jev label <id> correct|incorrect`); `jev stats
  [--last N] [--json]` renders counts, predicted-vs-observed calibration bins
  and the threshold that best separates correct from incorrect yesno decisions
  (`load_suggested_threshold` is the consumer `speculate --threshold auto`
  uses) — the measurement loop that makes the hardcoded 0.7/0.3 thresholds
  calibratable from real outcomes.
- `_nlp_read` (in `agent.py`) — paging is line-based and ALWAYS honored: the
  AST `definitions` summary is returned only for a BARE read (no `offset`/
  `limit`) of a `.py` file over `_CONTEXT_AST_THRESHOLD_KB` (50); returning it
  for every offset made the model loop forever on large files. A directory
  path is rejected with a clear `Not a file (directory)` message (never a raw
  `[Errno 13]`); `analyze` by contrast accepts a FILE or DIRECTORY (a folder is
  summarized from a bounded 40-file / 200k-char sample) and no path = the whole
  workspace.  A `read` call may also pass **`paths`** (array) to fetch several
  files in ONE call (`_nlp_read_many`, capped at `_MAX_READ_FILES`), so a model
  asked to "read two files" no longer refuses with "I can only read one file
  at a time".
- `agent_core/llm/tool_loop.py` — `ToolLoopRunner`: NLP tool-call execution loop.
- `agent_core/llm/lemonade_provider.py` — first-class AMD Lemonade (NPU)
  provider. Lemonade is AMD's OpenAI-compatible local server that runs LLMs on
  the Ryzen AI XDNA NPU (FastFlowLM / Ryzen AI LLM) or the iGPU; the agent talks
  to it exactly like LM Studio (no model management — Lemonade owns loading).
  Model ids are namespaced `lemonade/<id>`, `model list` shows a `[lemonade]`
  section, and `model lemonade/<id>` / `model <name> -p lemonade` switch to it.
  `LemonadeProvider.health_check()` probes `/models`; base URL comes from
  `model_catalog.json` `_defaults.lemonade_base_url` (default
  `http://localhost:13305/api/v1`, override `LEMONADE_API_URL`; optional default
  model `AGENT_LEMONADE_MODEL`). Install/run with `lemonade run <model>`
  (lemonade-server.ai). Tests: `tests/test_lemonade_provider.py` (26).
- `agent_core/llm/engines.py` — engine/accelerator abstraction: a named engine
  (`npu`→lemonade, `igpu`→lmstudio, `llama`, `cloud`→opencode/openrouter) maps
  to a provider + `device`; `resolve_engines` picks one model per engine from
  the live catalogs.  `agent_core/llm/context_budget.py` — per-provider context
  budgeting (`provider_context_limit` / `trim_messages_to_context`): the oldest
  turns are trimmed to fit a small NPU window so a prompt that fits the iGPU
  cannot overflow the NPU.  `run_parallel` gained `provider_overrides` + `warm`
  and every `ParallelResult` carries `device`; `multillm --engines npu,igpu
  [--warm]` runs one model per accelerator at once.  `LemonadeProvider` exposes
  `context_length()` / `ensure_model_loaded()` / `unload_model()` (residency),
  and the Jev engine can run on the NPU (`model jev lemonade/<small-FLM>`)
  since it votes when the provider has no `chat_logprobs`.
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
- `tests/` — pytest, **2572 collected** (~2 min full run with `--no-cov`;
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

- **Test efficiently — never pay for the whole suite twice.**
  - **Full-suite gate (2026-09-30)** `agent_core/pytest_gate.py`: a
    WHOLE-SUITE run (no test path, or `tests`/`.` spelled out) is REFUSED by
    the `run`/`tests` tools unless (a) a REAL tracked source file differs from
    the session baseline (`git status --porcelain -uall`; scratch paths
    `_tmp_*`, `tmp/`, `*.bak`, `reports/`, `.docs/`, `__pycache__` never
    count) and (b) the per-session budget `AGENT_MAX_FULL_PYTEST_RUNS`
    (default **1**) is not spent. The refusal text names the cheap lanes, so
    the model self-corrects from the tool result. Escape hatch:
    `AGENT_PYTEST_FULL_RUN_GATE=off` (or raise the budget) for a deliberate
    second pass. Also fixed here: `--lf` / `--nf` / `--new-first` /
    `--testmon` select a SUBSET and are no longer classified as full runs by
    either classifier (`agent._is_full_pytest_command`, `conftest._is_full_run`)
    — they were being given the whole-suite budget *and* the gate.
  - Full suite (only before commit/push, and only via the one earned run):
    `python -m pytest -q --no-cov`.
    Bare `pytest` and the whole tree spelled out (`pytest tests/`, `pytest .`)
    are all treated as full runs: they record their elapsed time and get a
    budget of `max(PYTEST_FULL_SUITE_TIMEOUT, last * 1.50)`. An overrunning run
    records its abort time, so the next budget grows instead of repeating.
    The NLP `run`/`tests` tools inject that same budget, so `pytest tests/`
    can no longer be killed at a guessed 120/300s.
  - Re-run only what failed: `python -m pytest --lf -q --no-cov`.
  - Only what changed: `python -m pytest --testmon -q --no-cov` (affected
    tests only; first run builds the map).
  - Fast lane: `python -m pytest -q --no-cov -x -m "not integration and not harnessfix_self_test" --ignore=tests/performance --ignore=tests/integration`.
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
- A `commit-msg` hook (tracked at `.githooks/commit-msg`, active via the same
  `core.hooksPath`) enforces the commit-subject convention below. It loads
  `agent_core/commit_policy.py` **by path** (never via the `agent_core`
  package, which would drag the whole agent stack into a git hook) and is
  stdlib-only. Vacuous subjects (`"commit changes"`, `wip`, a bare filename)
  are **errors** and block the commit; format deviations (no type prefix,
  unknown type, trailing period, >72 chars) are **warnings** and never block
  you. Covered by `tests/test_commit_policy.py`.
- Implement auto-runs `py_compile` on every written file.

## Conventions

- Commit style: short `fix:` / `feat:` / `docs:` subjects (see `git log`).
  **Enforced** by `.githooks/commit-msg` via `agent_core/commit_policy.py`:
  write WHAT changed and WHY. `"commit changes"` is rejected — don't pass a
  lazy subject straight to `git commit -m`.
- **Every bug fix ships with a regression test** (e.g. `TestDependencyCascadeSafety`,
  `TestPersistentMemory`, `TestAnalysisVerifier`).
- Decisions are recorded via the `decide` step; candidates carrying unverified claims
  require explicit confirmation before recording.
- Windows shell is `cmd.exe` — no grep/tail; use `python -c "..."` one-liners inside
  the agent REPL; normalize paths via `to_windows_path`.
- The NLP `search` tool's `path` argument is a **DIRECTORY to search recursively**,
  not a filename filter. Mechanism (`agent_core/file_searcher.py::_walk_search`,
  the `if not os.path.isdir(local_path): return matches` guard): a non-directory
  short-circuits to zero matches, which `_nlp_search` renders as the misleading
  `No files found matching that query.` So pointing it at a FILE
  (`path="types.py"`, `path="agent_core/memory/types.py"`) returns no hits even
  though the query matches that very file. Both separators work
  (`agent_core/memory` and `agent_core\memory` both match), so a forward slash is
  never the problem; a non-existent directory is (e.g. bare `memory` — paths
  resolve against the workspace root, not CWD). Do NOT misdiagnose this as
  filename filtering. For "where is X defined/used?", prefer `references`
  (one call, whole-word, capped) over `search` — it cannot misfire this way.
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
- **read paging + speculative tool loop (DONE, 2026-09-24)**: `read` on a
  >50 KB `.py` file honours an explicit `offset`/`limit` (returns lines); only a
  BARE read yields the AST `definitions` summary (now with a how-to-page note).
  Regression: `tests/test_agent_robustness.py`. `speculate` branches now carry
  the agent's system prompt and run a read-only tool loop (hard allowlist;
  mutating tools refused before the executor); branch-dispatch timeout
  60 -> 300s. Tests: `tests/test_speculate_cmd.py` (8).
- **LM Studio prefill-aware socket timeout (DONE, 2026-09-24)**: `_scaled_timeout`
  sizes the socket timeout to the estimated prompt PREFILL (tokens, ~3.5
  chars/token; `LMSTUDIO_PREFILL_TOKENS_PER_SEC` default 15, pessimistic)
  instead of ~1s per 50 KB, so a >600s prefill no longer trips LM Studio's
  "Client disconnected. Stopping generation...". Floor `LMSTUDIO_CHAT_TIMEOUT`
  (600), cap 3600s. Tests: `tests/test_lmstudio_payload.py`.
- **Reasoning-budget auto-recovery (DONE, 2026-09-27)**: `LMStudioProvider.chat`
  retries ONCE with thinking disabled AND an escalated `max_tokens`
  (`_escalated_max_tokens`: floor 4096, x4, cap 32768) when a reasoning model
  returns only `reasoning_content` and no content (`_is_thinking_budget_error`).
  Before, this surfaced `[Error: model consumed N reasoning bytes with no
  output]` and ended the turn.  Live (2026-09-27): `qwen/qwen3.5-9b` IGNORES the
  reasoning-off knob, so the larger budget is what actually recovers it (250-token
  request -> retry at 4096 -> full answer).  Tests: `tests/test_llm_retry_policy.py`.
- **Tool-path drop-tools recovery (DONE, 2026-09-27)**: when ``tools`` are
  sent to a model whose LM Studio TOOL path crashes the engine, `chat` detects
  it (`_is_tool_path_engine_error`: peg-native grammar, ``bad allocation``/
  out-of-memory, ``channel error``, ``terminated``, engine predict stream/request
  failure) and retries the SAME prompt WITHOUT tools so the turn completes as
  plain chat — then remembers the model in `_tools_unsupported` so later turns
  skip the doomed attempt.  Proven live with `llama-4-scout-17b-16e-instruct`:
  with tools the engine dies (`bad allocation`/`terminated`), without tools the
  full agent system prompt answers fine; LM Studio's own chat works because it
  sends no tools.  `_engine_server_error` EXCLUDES peg/OOM from transient
  retries (deterministic).  If no tools were sent or the no-tools retry also
  fails, the error propagates and the failover chain takes over.  Server-side
  root cause is a llama.cpp/LM Studio parser/alloc bug; mitigations are a newer
  LM Studio / higher quant (Q4_K_XL, not Q4_K_S).  Tests:
  `tests/test_llm_retry_policy.py::TestToolGrammarRecovery`,
  `::TestToolPathDropTools`.
- **Engine OOM recovery (DONE, 2026-09-27)**: `LMStudioProvider._open_chat`
  detects a `std::bad_alloc` engine 400 (`_is_engine_oom_error`: "bad
  allocation"/"out of memory"/...) and raises a RuntimeError with an actionable
  message ("free VRAM/RAM or use a smaller model/quant") instead of a transient
  error — OOM is deterministic, so retrying is wasted.  `_engine_server_error`
  excludes OOM, and `_CONNECTION_FAILURE_RE` gained `out of memory|bad
  allocation|failed to allocate|insufficient memory` so the chain fails over.
  Live: `llama-4-scout-17b-16e-instruct` (~63 GB GGUF) intermittently OOMs on
  load/alloc; the turn now degrades to failover.  Tests:
  `tests/test_llm_retry_policy.py::TestEngineOomRecovery`.
- **Jev decision engine (DONE, 2026-09-25)**: `agent_core/jev_engine.py` +
  REPL `jev` + NLP `jev_decide` + `speculate --judge jev|both`. Typed
  yesno/choice/score decisions on a DEDICATED small model
  (`settings.jev_model`, default `qwen2.5-coder-1.5b-instruct`) via logprobs or sample
  votes; abstains rather than fabricating a probability. New settings
  (`jev_model`/`jev_provider`/`jev_samples`/`jev_temperature`/`jev_max_tokens`/
  `jev_timeout`), `LMStudioProvider.chat_logprobs`, catalog
  `_defaults.jev_model`. Tests: `tests/test_jev_engine.py` (43),
  `tests/test_jev_cmd.py` (16), `tests/test_lmstudio_payload.py` (26),
  `tests/test_speculate_cmd.py` (16).
- **Jev telemetry + calibration (DONE, 2026-09-25)**: `harnessfix/jev_telemetry.py`
  records every decision to `reports/history/jev.jsonl` (best-effort, opt-out
  `AGENT_NO_JEV_LOG=1`), `record_outcome(id, ...)` labels it, and `jev stats`
  reports counts, predicted-vs-observed calibration bins and a suggested yesno
  threshold. `speculate --judge jev` also gained the gray-band cascade
  (`--low`, default 0.3): confident accept/reject costs one small-model call,
  only `low < P < threshold` escalates that candidate to the LLM judge — but
  escalation is now opt-in via `--escalate`, because `--judge jev` runs the
  whole command (branches + judge) on the dedicated model and must not touch
  the selected chat model by default.
  Tests: `tests/test_jev_telemetry.py` (13), `tests/test_jev_cmd.py` (20),
  `tests/test_speculate_cmd.py` (19).
- **Lemonade / AMD NPU provider (DONE, 2026-09-27)**: new
  `agent_core/llm/lemonade_provider.py` (`LemonadeProvider`) + catalog
  `_defaults.lemonade_base_url` + `_routing.lemonade` + `AgentSettings.
  lemonade_api_url`/`lemonade_model` + `_LLM_PROVIDERS` entry.  Routing is
  namespaced `lemonade/<id>` (the `lemonade` ROUTER key sits BEFORE the LM
  Studio family keys so `lemonade/qwen…` is not hijacked by the `qwen` key);
  `model list` gains a `[lemonade]` section, `model <name> -p lemonade` and
  `model lemonade/<id>` switch to it, and the provider exposes
  `health_check()` + `list_models()`.  Fetch errors retry 429/5xx and fall over
  via the shared `(connection error)` marker.  Tests:
  `tests/test_lemonade_provider.py` (26).
- **NPU+iGPU parallelism foundation (DONE, 2026-09-27)**: engine/device
  abstraction (`agent_core/llm/engines.py`), per-provider context budgeting
  (`agent_core/llm/context_budget.py`), model residency on Lemonade
  (`context_length`/`ensure_model_loaded`/`unload_model`), and device-aware
  parallel dispatch (`run_parallel(provider_overrides=..., warm=...)`,
  `ParallelResult.device`, `multillm --engines npu,igpu [--warm]`).  NPU Jev
  offload works out of the box (`model jev lemonade/<small-FLM>` → vote
  mechanism).  Tests: `tests/test_engines.py`, `tests/test_context_budget.py`,
  `tests/test_parallel_engines.py`, `tests/test_lemonade_provider.py`.
  Fix (2026-09-27): `build_provider` with a multi-provider chain now adds a
  SYNTHETIC FRONT ENTRY when the active model routes to a provider the chain
  does not list (e.g. a `lemonade/…` model with the default cloud-only chain).
  Before, the model was silently dropped and the chain's first entry answered —
  `multillm` ran `opencode-go/deepseek` instead of the NPU for a lemonade model,
  because only `model lemonade/…` (which passes `provider_override`) had
  reached Lemonade.  Now the NPU is tried first, the chain stays as fallback.
- **#080 — collision guard distinguishes pins from producers (DONE, 2026-09-27)**:
  `harnessfix/repairs/collisions.py` reported every test-suite occurrence of a
  repair-affected fragment, including tests that merely *produce* it.  With the
  accepted `tool-interface-error-detail` repair already in the tree, the two
  executor doubles in `tests/test_tool_loop_nlp.py` (`return "Tool error: boom"`)
  made every loop iteration end `skipped_test_collision` — the loop deadlocked
  reporting a collision the file's real assertion no longer had (it pins the NEW
  `"Tool error (<ExcType>): "` form).  The guard now groups lines into logical
  statements (`_logical_statements`, bracket-depth, so a wrapped `assert (` …
  `) in out` stays ONE statement) and reports a hit only when the statement
  PINS the fragment (`_statement_pins`: no `return`/`yield`/assignment, and an
  `assert`/comparison/`in`/`.startswith(`/`.count(` marker).  Classification errs
  toward MISSING a pin — a missed pin costs one gate run (the test gate reverts
  the repair), a false collision blocks the repair forever.  Verified: 0
  non-guard pins for all three catalog fragments, `test_tool_loop_nlp.py` clean,
  real pins still caught.  Tests: `tests/test_harnessfix_collisions.py` (12).

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
