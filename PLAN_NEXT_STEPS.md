# Next Steps — Plan (branch: master)

## Current state (verified with git log / ruff this session)

Local commits, NOT pushed (the other agent owns push). Oldest first:
- `468d25d`, `ab93510`, `7412dec` — E501 wrapping in `agent_core/agent.py`.
- `7277ffa` refactor(agent): replace bootstrap star import with explicit names.
- `119099b` fix(agent): drop duplicate logger and TYPE_CHECKING redefinitions (F811).
- `a6f48a9` fix(lmstudio): run context probe in worker thread so parallel calls
  do not serialize. Fixes the previously failing
  `tests/test_parallel_llm.py::TestParallelConcurrency::test_blocking_providers_do_not_serialize`.
  Verified this session: `test_parallel_llm.py` 43 passed; testmon lane 547 passed
  (1066 deselected). The full-suite count of 3797 was NOT re-verified; it came from
  an earlier note and should be treated as unconfirmed.
- `d19625c` refactor(agent): move helpers below imports — E402 in `agent.py` is 0.
  About 181 targeted tests pass.
- `bb07e4c` refactor(lint): rename ambiguous loop variable `l` to `ln` in 13 files
  (E741). Only targeted tests (~474) were run, not the full suite. Review this one
  first.

Working tree: clean. Branch: 8 commits ahead of `origin/master`.

Repo-wide lint still open (ruff): E501 about 2,670 lines; E402 50 (outside agent.py).
Other rules in the deferred backlog (E702, F401, E731, F402, F811, F841) were not
rechecked after the commits above.

## Next steps, in order

1. Coordinate with the other agent before any push. Confirm they are not editing
   the same files (`agent_core/agent.py`, `agent_core/commands/*`, `agent_core/handlers/nlp_tools.py`).
2. DONE: Reviewed `bb07e4c` (the `l` -> `ln` rename). Ruff (E741, F821, F841,
   E402, F811) is clean on the 7 touched source files, and 109 tests in the 6
   related test files pass. Keep it; revert with a new commit only if needed.
3. Run the full suite once more at the end of the batch (not per commit).
4. E501 package-wide, one module per commit, one concern per commit.
5. Remaining E402 (50 repo-wide) and the deferred backlog (E702, F401, E731, F402,
   F811, F841). Coordinate first.

## Rules of engagement (multi-agent repo)

- Never `git push` (the other agent pushes).
- `git status` before every edit; small targeted diffs; one concern per commit.
- Full-suite pytest once at the end; targeted tests while iterating.
- Delete `_tmp_*.py` scratch files before finishing.
