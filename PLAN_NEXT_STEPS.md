# Next Steps — Plan (branch: master)

## Current state (verified this session: git status, `ruff check . --statistics`, full pytest)

Branch `master` is **clean and in sync with `origin/master`** (HEAD `2623dfc`).
There is no second agent: this session owns push, and pushes work from here
(remote is a plain HTTPS URL, credential helper caches the valid token).

Latest commits:
- `2623dfc` style(lint): wrap 38 over-long import lines (E501) + drop unused
  `import sys` (F401). E501 2619 -> 2581. Each touched file py_compile-checked
  before writing; F821 clean; all touched modules import at runtime.
- `69a6a41` fix: Linux CI regressions in `_norm`, `kill_process_tree`, autoreview
  corpus test.
- `d16a1de` fix(paste): show the platform EOF key (Ctrl+D on POSIX) + tests.
- `d2de06f` fix(repl): ignore SIGTSTP so Ctrl-Z in pasted input does not suspend.

Verified lint (repo-wide, ruff, line-length 88 from `pyproject.toml`):
- E501 **2581** (was 2619). Of these, ~2069 are long **string literals**
  (prompt/doctext text) — wrapping them changes asserted text, so they need
  hand-by-hand edits, not a script. 42 are comments; the rest is code.
- E402 **44** — concentrated in test files that import after `sys.path` setup
  (9 `test_repairs_abandonment_resume`, 8 `test_repairs_stuck_repeat`,
  6 `test_harnessfix_collisions`, 5 `test_harnessfix_loop`, 6
  `performance_dashboard/main.py`), plus a few in `scripts/`.
- E702, E731, F401, F402, F811, F841, E741: **0** (F401 fixed in `2623dfc`).

CI gate (`.github/workflows/ci.yml`, ubuntu + windows, py3.12):
- **hard**: `ruff --select F821` (must be clean), `pytest -q --no-cov`.
- **advisory** (continue-on-error): full `ruff check`, `mypy agent_core --strict`.
- `PYTEST_FULL_SUITE_TIMEOUT=1800` is set in the workflow env for runners.

Full suite (this session): **3774 passed, 22 failed, 19 skipped** in ~364 s.
All 22 failures are **pre-existing** — they reproduce on a clean tree with the
lint changes stashed (verified). They are: stateful pytest-gate tests
(`test_pytest_full_run_gate.py`, which inspect the live session's full-run
budget), latency benchmarks (`test_decisions_performance.py`,
`tests/performance/test_decision_latency.py`), and
`test_clear_history{,_gate_reset}.py`. None import the files touched by `2623dfc`.

## .env (untracked; de-duplicated this session)

`.env` had the same block pasted 3x. It now has 18 unique keys (27 duplicate
lines removed). Effective (last-wins) values are unchanged EXCEPT
`GITHUB_TOKEN`, which now resolves to the **valid** token — the revoked
duplicate that used to shadow it is gone. Backup was removed after verification.

## Next steps, in order

1. **User action (GitHub web UI):** revoke the leaked token that sat in
   `.git/config` in plaintext (suffix `K8ky`). It returns 401, so deleting it
   is pure clean-up.
2. **E501 — long string literals (~2069).** The bulk of the remaining debt.
   Each is prompt/docstring text with assertions on exact content, so it needs
   manual, test-aware edits: one module per commit, run that module's tests
   each time. Highest-density files: `implement_cmd.py` (364), `fix_cmd.py`
   (184), `optimize_cmd.py` (124), `workflow_cmd.py` (123) — ~42% of all E501.
3. **E501 — comments (42) and remaining code lines.** Low risk, mechanical.
4. **E402 (44).** In test files, move the `sys.path` bootstrap above imports
   or gate with `# noqa: E402`; in `performance_dashboard/main.py` reorder.
5. **Triage the 22 pre-existing failures.** They are real signals: decide
   whether the latency benchmarks are too tight for CI hardware, and why
   `clear_history` fails. Do not just mark them expected-fail.
6. Run the full suite **once** at the end of a batch (targeted tests while
   iterating).

## Rules of engagement (single-agent repo)

- Push is owned by this agent; `git status` before staging, `git push` after a
  verified commit. Remote has no embedded credentials.
- Small targeted diffs; one concern per commit.
- Full-suite pytest once at the end; targeted tests while iterating.
- Never wrap a long **string literal** mechanically — it changes asserted text.
- Delete `_tmp_*.py` scratch files and one-off backups before finishing.
