# Next Steps — Plan (branch: master)

## Current state (verified)

Local commits, NOT pushed (the other agent owns push):
- `468d25d` chore(agent): wrap remaining long lines in agent_core/agent.py (E501)
  — E501 in agent.py is 0.
- `7277ffa` refactor(agent): replace bootstrap star import with explicit names
  — `from agent_core.bootstrap import *` replaced by an explicit list of the
  136 names in `bootstrap.__all__`. Module namespace verified identical
  before/after (no missing, no added names).
- `119099b` fix(agent): drop duplicate logger and TYPE_CHECKING from explicit import
  — removed two F811 redefinitions introduced by the explicit list.

Verification:
- `ruff --select F403,F405,F811,F821,F401` on agent.py: clean.
- `tests/test_agent_module_surface.py` + `tests/test_agent_dry_refactor.py`: 24 passed.
- Full suite (`python -m pytest -q --no-cov`): 3795 passed, 7 skipped, 1 failed.
  - Failure: `tests/test_parallel_llm.py::TestParallelConcurrency::test_blocking_providers_do_not_serialize`
    (elapsed ~4.6s vs < 0.35s). It FAILS ON HEAD TOO (stashed the change,
    re-ran: 4.69s), so it is pre-existing and not caused by the import change.
    Needs its own investigation (likely `run_parallel` not using
    `asyncio.to_thread` for the blocking `urlopen` path).

Advisory lint in agent.py: 20 E402 (module-level imports after code). Pre-existing;
they come from the import block being split around helper definitions.

## Next steps, in order

1. Investigate the failing parallel test (pre-existing). Start at
   `agent_core/llm/parallel.py` — confirm blocking provider calls are dispatched
   via `asyncio.to_thread`. Fix the implementation, do not weaken the assertion.
   Add/keep a regression test.
2. Consolidate the E402 imports in `agent_core/agent.py` by moving the late
   imports up into the top import block (verify namespace identical, as above).
3. Package-wide E501 (~1560 lines) — one module at a time, one concern per commit.
4. Deferred backlog (coordinate first): E702 (14), F401 (38), E731 (22),
   E741 (13), F402 (9), F811 (15), F841 (11).

## Rules of engagement (multi-agent repo)

- Never `git push` (the other agent pushes).
- `git status` before every edit; small targeted diffs; one concern per commit.
- Full-suite pytest once at the end; targeted tests while iterating.
- Delete `_tmp_*.py` scratch files before finishing.
