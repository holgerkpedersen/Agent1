# `agent.py` decomposition — baseline and step log

Plan: `.docs/2026-10-16_21-37-29/plan_proposed.md` (13 tasks, 38 dependency edges).
This file is the working log: one section per step, with the measured numbers
before and after. Every number below was produced by a command in this session.

## Baseline (measured 2026-10-16, before t02)

| Metric | Value | How measured |
| --- | --- | --- |
| `agent.py` lines | **5565** | `len(read().splitlines())` |
| Classes in `agent.py` | **2** — `LLMClient`, `Agent` | `ast` walk of module body |
| Module-level functions | **52** | `ast` walk |
| Methods on the 2 classes | **50** | `ast` walk |
| Public names importable from `agent` | **146** | `dir(agent)` minus `_`-prefixed |
| Tests collected | **3686** | `pytest --collect-only -q --no-cov` |
| mypy errors, `agent.py` itself | **12** | `mypy agent.py`, lines starting `agent.py:` |
| mypy errors, transitive total | **106** in 31 files | same run, full output |

mypy error codes on `agent.py` itself:
`unused-ignore` 3, `arg-type` 3, `attr-defined` 2, `redundant-cast` 2,
`union-attr` 1, `assignment` 1.

Full mypy output is kept in `mypy_baseline_out.txt` for later diffing.

## The guard

`tests/test_agent_module_surface.py` (166 tests, all passing at baseline) pins:

1. `import agent` keeps working.
2. All 146 baseline public names stay reachable from `agent`.
3. A named subset that the plan relocates (`Agent`, `LLMClient`,
   `run_interactive`, `main`, `NLP_TOOL_SCHEMAS`, …) keeps its re-export.
4. `agent.py` never exceeds 5565 lines — a step that grows it copied
   code instead of moving it.

`EXPECTED_PUBLIC_NAMES` includes re-exported stdlib modules (`os`, `json`,
`sys`, …). They are part of the observed public surface because other modules
do `from agent import os`-style imports, so the guard keeps them pinned.

## Step log

| Step | Scope | Status | `agent.py` after |
| --- | --- | --- | --- |
| t01 | Baseline + surface guard test | done | 5565 (untouched) |
| t02 | imports + bootstrap side effects → `agent_core/bootstrap.py` | not started | |
| t03 | history compaction (3944–5249) → `agent_core/history_compaction.py` | not started | |
| t04 | turn heuristics + tracing (4083–4409) → `agent_core/turn_telemetry.py` | not started | |
| t05 | shell/subprocess helpers → `agent_core/shell_intel.py`, `agent_core/pytest_cmd_policy.py` | not started | |
| t06 | `_SYSTEM_PROMPT` + shell hints → `agent_core/prompts/system_prompt.py` | not started | |
| t07 | `LLMClient` (351–526) → `agent_core/llm/client.py` | not started | |
| t08 | signals + interruptible input (245–348) → `agent_core/cli/signals.py` | not started | |
| t09 | REPL layer (5271–5558) → `agent_core/cli/registry_setup.py`, `interactive.py` | not started | |
| t10 | `_nlp_*` handlers (862–2037) → `agent_core/handlers/nlp_tools.py` | not started | |
| t11 | plan tools / context blocks / persistence | not started | |
| t12 | turn loop (2653–3231) → `agent_core/turn_loop.py` | not started | |
| t13 | consolidation, lint guard, docs, commit | not started | |

## Rules each step must follow

1. One step = one commit. Move code, do not change behaviour.
2. After each move: `python -c "import agent"` plus
   `pytest tests/test_agent_module_surface.py` must pass.
3. Mutable module globals move **together with the functions that mutate
   them**. A function that mutates a global it no longer owns is the main
   hazard of this refactor.
4. Import order is load-bearing: `OPENBLAS_NUM_THREADS` must be set before
   `numpy`/`scipy` are imported.
5. Do not "fix" the 12 pre-existing mypy errors while moving code — record
   them, keep the count flat, clean them in t13.
