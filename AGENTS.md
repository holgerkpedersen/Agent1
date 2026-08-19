# AGENTS.md — Agent1 (self-improving AI coder agent)

Project memory for AI-agent sessions. Read this first; it encodes hard-won invariants
that cost a deleted `agent.py` to learn.

## What this is

A Python agent system (REPL + workflow pipeline) that uses LM Studio models
(currently `laguna-s-2.1` under `deep-analysis` or `fast-codegen` profile) to analyze,
plan, implement, and test code changes in its own workspace. It records every
design decision and is being extended to audit its own file effects (self-improvement).

## Architecture map

- `agent.py` — entry point; REPL `run_interactive`; the `Agent` class: `chat_nlp`
  tool-call loop, memory stores (`_files_read`, `_semantic_index`, `_knowledge_graph`,
  `_working_memory`, `_history`, `_file_mtimes`), and persistence via
  `chat_history.json` + `agent_memory.json` (repo root, gitignored).
- `agent_core/commands/*_cmd.py` — registry commands (REPL): `read, write, search,
  analyze, plan, entities, taskplan, implement, fix, workflow, decide, clear, model,
  run, self_heal, optimize, perf, paste, display, cleanup`.
- `agent_core/llm/tool_loop.py` — `ToolLoopRunner`: NLP tool-call execution loop.
- `agent_core/security/` — sanitizers; `agent_core/file_system.py`, `path_utils.py` —
  real path utilities (`to_windows_path`, `normalize_path`); **do NOT invent new ones**.
- `harnessfix/` — self-improvement consumers: `tracing.py` (opt-in `TraceWriter`),
  `reader`, `diagnose`, `gates`, `loop`, `htir`, `links`, `corpus`, `history.py`
  (trace-index + execution-ledger queries and PAST EXECUTION NOTES formatters that
  implement/fix inject into prompts).
- `tests/` — pytest, **1128 passed, 2 skipped** (~3.5 min full run; `testpaths=["tests"]`).
- `agent_core/tests/` — entry-point/component test package (31 tests, runs only when
  targeted: `python -m pytest agent_core/tests -q --no-cov`); reconstructed 2026-08-19
  after the source was deleted uncommitted (decision #058).
- `.decisions.json` — decision ledger (57 records, gaps ok; latest #057/#058).
- `.docs/<timestamp>/` — one folder per workflow run (spec/analysis/plan/entities/tasks).
- `backups/` — implement's pre-run copies of existing targets (timestamped).

## Workflow pipeline

`workflow <target> [--brainstorm] [--desc "..."] [--force]` →
`.docs/<ts>/project_{spec,analysis,plan,entities,tasks}.md` → prompts to continue
into `implement <tasks> <analysis> <plan> <entities> --workspace . --modify`.

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
   the wild: `agent_core/security/secrets.py` was planned but does not exist.
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
- mypy: `python -m mypy <file>`. **Known baseline: 34 pre-existing errors in 5 files**
  (`agent.py:88-90,500-501`; `implement_cmd.py:2273,2307...`; `model_cmd.py:168,525-527`).
  Do not silently "fix" them; do NOT introduce new errors.
- No ruff config; pyproject.toml configures mypy, pytest, coverage only.
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

## Roadmap (recorded, in .decisions.json / .docs/2026-08-18_10-45-11/)

- #047 — sanitize shell commands and file contents before trace persistence.
- #048 — instrumentation invisible unless tracing enabled (`AGENT_NO_TRACE=1`).
- ✅ **#049 — files-affected recording per tool/nlp (DONE)**: `ToolLoopRunner`
  gains `effects_fn` (`(tool_name, args) -> [paths]`); `tool_result`/`tool_error`
  events carry `affected_files`; only invoked when a trace sink exists. `Agent`
  arms `_pending_effects` only while a `TraceWriter` exists, notes
  read/write/edit/fix targets. REPL registry **commands** still uninstrumented
  (next increment); trace consumers (`harnessfix/reader`) not yet reading
  `affected_files`.
- ✅ **#050-#056 — verification gate increment (DONE, 2026-08-18)**:
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
- ✅ **#060 — history-assisted implement/fix (DONE, 2026-08-19)**: new
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
- Taskplan-time phantom-module gate (plan-time existence check for planned files).
