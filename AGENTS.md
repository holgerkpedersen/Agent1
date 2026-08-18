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
  `reader`, `diagnose`, `gates`, `loop`, `htir`, `links`, `corpus`.
- `tests/` — pytest, **1066 collected / 1064 passed, 2 skipped** (~3.5 min full run).
- `.decisions.json` — decision ledger (47 records, gaps ok; latest #047/#048).
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
- Files-affected recording per tool/nlp/command in `agent.py` (the original
  self-improvement goal — analysis exists, implementation still pending).
- Taskplan-time phantom-module gate (plan-time existence check for planned files).
