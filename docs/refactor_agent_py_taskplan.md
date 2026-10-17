# Taskplan — decompose `agent.py` in small, reviewable steps

**Target:** `C:\Dev\Agent1\agent.py`
**Status:** steps t01–t02 done; this plan covers the remaining **t03–t15**.
**Supersedes:** `.docs/2026-10-16_21-37-29/plan_executed_20261017_004945.md`
(line numbers there are stale; every number below was re-measured this session).

---

## 1. Measured current state

| Metric | Value | How measured |
| --- | --- | --- |
| `agent.py` lines | **5416** | `len(open("agent.py").read().splitlines())` |
| Classes | **2** — `LLMClient` 202–377, `Agent` 400–3791 | `ast` walk of module body |
| `Agent` methods | **100** | `ast` walk |
| Module-level functions | **54** (1329 lines) | `ast` walk |
| Module-level constants | **40** | `ast` walk |
| Already extracted | `agent_core/bootstrap.py` (import block + OpenBLAS + UTF-8 stdout) | commit `7dac445` |

Committed history for this refactor:

| Commit | Step | Effect |
| --- | --- | --- |
| `b45d5c8` | t01 | baseline recorded + `tests/test_agent_module_surface.py` guard |
| `7dac445` | t02 | imports and bootstrap side effects → `agent_core/bootstrap.py` |

The guard (`tests/test_agent_module_surface.py`, 403 lines) freezes the module
surface in two frozensets and fails loudly when a name moves without an
announced edit:

* `EXPECTED_DEFINITIONS` — names `agent.py`'s **own body** must keep defining.
* `EXPECTED_IMPORTED` — names reachable through `agent.py`'s **import block**.

Its docstring states the rule for every step:

> When a step moves code out, remove the name from this list in the SAME commit
> and add it to `EXPECTED_IMPORTED` (the re-export).

---

## 2. Three hard constraints

1. **`agent.py` is a public import surface.** 83 files do `from agent import Agent`,
   5 import `LLMClient`, and ~30 test files import private helpers by name
   (`from agent import _trim_chat_history`, `_MAX_CHAINED_RUNS`, …). Every moved
   name must stay importable from `agent` through an explicit re-export.
2. **No new module may `import agent` at module level.** `agent_core.decisions` and
   `agent_core.subagent` already import `agent`; a module-level import back would
   close the cycle. Import inside the function body when a call-time lookup is
   needed (see §3).
3. **Tests monkeypatch configuration on the `agent` module.** `conftest.py:168-196`
   plus ~30 test files do `monkeypatch.setattr(agent, "NAME", …)`. Moving a
   function changes which module a *bare global read* resolves against — see §3.

---

## 3. The one rule that makes every step safe

**A moved function must not read a monkeypatched name as a bare global.**

A bare read resolves against the enclosing module's globals. After the move the
enclosing module is the new one, so a patch on `agent.NAME` is silently ignored
and the test still passes or fails for the wrong reason.

Verified bare-global reads of test-patched names inside `agent.py`:

| Name read as a bare global | Read at line(s) | Enclosing function | Patched by |
| --- | --- | --- | --- |
| `_MAX_CHAINED_RUNS` | 2686 | `Agent._run_chained_tool_loop` | 4 test files |
| `_CONTEXT_AST_THRESHOLD_KB` | 1330, 1333 | `Agent` (read/page) | 1 |
| `_DELEGATE_TIMEOUT_S` | 792, 796, 874, 877 | `Agent` (delegate, parallel) | 1 |
| `_ENV_FILE_PATH` | 4537 | `_read_env_value` | 1 |
| `TURN_LOG_PATH` | 4219 | `_append_turn_log` | 3 |
| `CHAT_HISTORY_JSON_PATH` | 3221, 3514, 3553, 3602, 3606 | `Agent` | 13 |
| `AGENT_MEMORY_JSON_PATH` | 134, 3222, 3624, 3673, 3677, 3749 | `_signal_break_with_cleanup`, `Agent` | 15 |
| `load_agent_settings` | 215, 558, 4019, 4039 | `LLMClient`, `Agent`, `_resolve_stream_fn`, `_resolve_display_mode` | 1 |
| `resolve_model` | 210, 413 | `LLMClient`, `Agent.__init__` | 1 |
| `_resolve_display_mode` | 2481, 2792, 3007, 3053, 3064, 3081, 3300 | `Agent` | 3 |
| `_kill_process_tree` | 1494 | `Agent._nlp_run` | 1 |
| `_looks_incomplete` | 2683 | `Agent._run_chained_tool_loop` | 1 |
| `_pytest_full_suite_timeout` | 1463, 1505, 1711 | `Agent` | 2 |
| `record_full_run_seconds` | 1500, 1518, 1721 | `Agent` | 1 |
| `record_turn_outcome` | 2872 | `Agent._finish_turn` | 2 |
| `decisions_as_system_prompt` | 3248 | `Agent._decision_constraints_block` | 1 |
| `recommend_profile` | 2993 | `Agent.chat_nlp` | 1 |
| `ToolLoopRunner` | 2510, 2621, 2816 | `Agent` | 1 |
| `LLMClient` | 495 | `Agent.__init__` | 5 |
| `_start_kanban_inbound` | 5398 | `main` | 3 |
| `run_interactive` | 5409 | `main` | 3 |
| `run_dashboard_server` | 5402 | `main` | 1 |
| `start_dashboard_thread` | 5407 | `main` | 1 |

**Mitigation — one tiny helper, used by all moved code.** Create it in t03 and
reuse it in every later step:

```python
# agent_core/runtime_globals.py
"""Call-time reads of the `agent` module's globals.

Extracted code lives in agent_core, but tests and conftest.py monkeypatch
configuration on the `agent` module. A bare global read inside a moved
function would resolve against the new module and silently ignore those
patches, so moved code reads patchable configuration through `live()`.
"""

def live(name: str, default: Any = None) -> Any:
    """Read *name* from the `agent` module at CALL time (never at import)."""
    import agent  # function-scope: keeps the import graph acyclic
    return getattr(agent, name, default)
```

Then a moved function writes `continuations < live("_MAX_CHAINED_RUNS", 6)`
instead of `continuations < _MAX_CHAINED_RUNS`. Every existing
`monkeypatch.setattr(agent, "_MAX_CHAINED_RUNS", 0)` keeps working, and **no test
file needs editing**. This is the whole trick: the patches stay where they are,
the code reads them from where they are.

---

## 4. Per-step checklist (definition of done)

Copy this list into each step's commit message. A step is done only when all
eight pass, in this order.

1. **Pure move.** Copy the functions verbatim. No renames, no signature changes,
   no logic edits, no "while I'm here" cleanups.
2. **Patchable reads go through `live()`.** Check the §3 table for names the
   moved functions read.
3. **Re-export explicitly** in `agent.py`: `from agent_core.<mod> import (…)  # noqa: F401`.
4. **Update the guard in the same commit**: delete each moved name from
   `EXPECTED_DEFINITIONS` and add it to `EXPECTED_IMPORTED` in
   `tests/test_agent_module_surface.py`.
5. `python -m compileall agent.py agent_core/<new>.py`
6. `python -c "import agent; print(agent.Agent, agent.LLMClient)"`
7. `python -m pytest tests/test_agent_module_surface.py -q` **plus** the targeted
   tests for the moved region.
8. `git add -A` then
   `git commit -m "refactor: extract <region> to agent_core/<module> (step tNN/15)"`.

**Never let `agent.py` grow.** A step that increases its line count copied code
instead of moving it.

### Extraction style for `Agent` methods (steps t11–t14)

Extract each method as a **module-level function that takes the agent first**,
and keep a one-line delegating method on `Agent` under the same name:

```python
# agent_core/handlers/nlp_tools.py
async def nlp_run(agent: "Agent", args: dict[str, Any]) -> str: ...

# agent.py, inside class Agent — name and signature unchanged
async def _nlp_run(self, args: dict[str, Any]) -> str:
    return nlp_tools.nlp_run(self, args)
```

Why this and not collaborator objects: every existing call site keeps working
(including `_nlp_tool_handlers()`'s dispatch table, which binds `self._nlp_run`),
every `monkeypatch.setattr(agent_instance, "_nlp_run", …)` keeps working, and each
delegator is two lines a reviewer can read at a glance. Use
`TYPE_CHECKING` for the `Agent` annotation to avoid an import cycle.

---

## 5. The steps

Ordered by increasing risk: module-level helpers first (no `self`, no instance
state), `Agent` decomposition last (all 100 methods share instance state, so
those steps are strictly serial).

### Phase A — module-level helpers

**t03 — history compaction** → `agent_core/history_compaction.py` *(deps: t01)*
Highest test coverage of any region, so do it first and cheaply.
Functions 4694–5100: `_drop_orphan_tool_messages`, `_repair_unanswered_tool_calls`,
`_strip_tool_args`, `_message_size`, `_content_tokens`, `_compaction_note_goal`,
`_is_compaction_note`, `_strip_turn_wrappers`, `_recover_original_goal`,
`_first_user_message_text`, `_trim_chat_history` (93 lines),
`_strip_image_blocks`, `_project_chat_history`.
Constants: `_MAX_CHAT_MESSAGES` (3823), `_HISTORY_CHAR_BUDGET`,
`_HISTORY_TRIM_NOTE` (3832–3836), `_HISTORY_TRIM_GOAL_CAP`.
Also creates `agent_core/runtime_globals.py` (§3) — nothing here needs it yet.
Tests: `tests/test_tool_loop_nlp.py`, `tests/test_quickwins_2026_08_25.py`,
`tests/test_paste_image.py`, `tests/test_file_history.py`.

**t04 — turn telemetry** → `agent_core/turn_telemetry.py` *(deps: t01)*
Functions 3934–4260: `_looks_incomplete`, `_unverified_action_claim`,
`_resolve_stream_fn`, `_resolve_display_mode`, `_final_answer_fallback`,
`_turn_quality_score`, `_begin_trace_turn`, `_append_turn_log`,
`_strip_dynamic_system_blocks`.
Constants: `_INCOMPLETE_MARKERS`, `_COMPLETE_MARKERS`, `_ACTION_VERB_RE`,
`_QUESTION_LEAD_RE`, `_CONTINUE_NOTE`, `_CONTINUE_NOTE_TAG`,
`_CONTINUE_NOTE_TAG_KEY`, `_READ_LOOP_NOTE`, `_VERIFY_NOTE`.
⚠ `_resolve_stream_fn` reads `load_agent_settings` at 4019 and
`_resolve_display_mode` at 4039, and `_append_turn_log` reads `TURN_LOG_PATH` at
4219 → all three must go through `live()`.

**t05 — shell and subprocess helpers** → `agent_core/shell_intel.py` *(deps: t01)*
Functions: `_current_git_branch` (164–176), `_unix_command_hint` (4056),
`_windows_command_hint` (4069), `_shell_command_hint` (4080),
`_kill_process_tree` (4107–4125), `_git_branch` (4128–4144),
`_detect_shell` (4263–4279), `_shell_name_token` (4282–4297),
`_truncate_output` (4433–4448), `_run_subprocess_captured` (4451–4474),
`_shape_run_stderr` (4477–4505), `_blocked_shell_command` (4508–4520).
Constants: `_MAX_RUN_TIMEOUT_S`, and keep
`_DANGEROUS_SHELL_PATTERNS = DESTRUCTIVE_SHELL_PATTERNS` as a **deliberate alias
in `agent.py`** (it is re-exported and referenced by tests).
⚠ `_kill_process_tree` is monkeypatched as `agent._kill_process_tree` in 1 test file.

**t06 — pytest command policy** → `agent_core/pytest_cmd_policy.py` *(deps: t01)*
Functions: `_read_env_value` (4532–4547), `_pytest_full_suite_timeout` (4550–4573),
`_is_full_pytest_command` (4609–4651), `_is_pytest_command` (4654–4669),
`_is_subset_pytest_command` (4672–4691).
Constants: `_PYTEST_FULL_SUITE_TIMEOUT_KEY`, `_PYTEST_LAST_FULL_RUN_KEY`,
`_FULL_SUITE_MARGIN`, `_PYTEST_BARE_TOKENS`, `_PYTEST_VALUE_FLAGS`,
`_PYTEST_SUBSET_FLAGS`.
⚠ `_read_env_value` reads `_ENV_FILE_PATH` at 4537 (patched by 1 test file).

**t07 — system prompt** → `agent_core/prompts/system_prompt.py` *(deps: t05)*
`_SYSTEM_PROMPT` (4309–4430, 122 lines) and `_SHELL_NAME` (4302).
⚠ `_SHELL_NAME = _shell_name_token()` is an import-time side effect; keep the
assignment in `agent.py` (or in the new module) but **exactly once** — a
duplicate evaluation would call `_detect_shell` twice.
Tests: 4 files import `_SYSTEM_PROMPT`.

**t08 — `LLMClient`** → `agent_core/llm/client.py` *(deps: t02)*
Class body 202–377 (176 lines, 8 methods).
⚠ Reads `resolve_model` at 210 and `load_agent_settings` at 215 as bare globals
→ `live()`. `agentic_bench.py` and `agent_core/subagent.py` import it from
`agent`, so the re-export is mandatory.

**t09 — signals and interruptible input** → `agent_core/cli/signals.py` *(deps: t02)*
Functions: `_suppress_and_log` (81–91), `_signal_break` (106–112),
`_signal_break_with_cleanup` (122–137), `_install_signal_handlers` (140–161),
`_interruptible_input` (179–199).
**Move the mutable globals with the functions that write them** —
`_SHUTDOWN_FLAG`, `_shutdown_save_memory`, `_shutdown_trace_close`,
`_input_executor` — so every `global` statement keeps targeting one module.
⚠ `_signal_break_with_cleanup` reads `AGENT_MEMORY_JSON_PATH` at 134 → `live()`.
The guard test exempts the three `None`-initialised placeholders (line 361), so
the surface edit is smaller than it looks.

**t10 — REPL / CLI layer** → `agent_core/cli/registry_setup.py` + `agent_core/cli/interactive.py` *(deps: t08, t09)*
`registry_setup.py`: `_register_commands` (5122–5170), `_build_registry` (5173–5177),
`_warn_uncommitted` (5180–5211), `_build_chat_prompt` (5214–5232).
`interactive.py`: `CHAT_INPUT_PROMPT` (5237), `run_interactive` (5240–5370),
`_start_kanban_inbound` (5373–5388), `main` (5391–5409).
⚠ `main` reads four patched names as bare globals (lines 5398–5409) → `live()`.
**Keep `if __name__ == "__main__"` behaviour byte-identical** — this is the file
users launch.

### Phase B — `Agent` decomposition (strictly serial)

**t11 — legacy tool aliases and file operations** → `agent_core/agent_files.py` *(deps: t03–t10)*
Methods 2221–2408: `_tool_read_file`, `_tool_write_file`, `_tool_apply_patch`,
`_tool_edit_file`, `_tool_search`, `_tool_list_files`, `_tool_delete_file`,
`_tool_analyze_file`, `_tool_llm_analyze`, `execute_tool`, `_normalize_path`,
`_safe_path`, `read_file`, `write_file`, `apply_patch`, `edit_file`,
`_build_semantic_index`, `_cleanup_semantic_index`, `search_file`.
Small (111 method-lines) and self-contained — the warm-up for Phase B.
Keep the delegators per §4.

**t12 — NLP tool handlers** → `agent_core/handlers/nlp_tools.py` *(deps: t03–t10)*
28 methods, 749–1888, 1023 method-lines — the single biggest win.
**Split into per-cluster commits** (each cluster = one commit, same checklist):

* 12a — `agent_core/tool_call_repair.py`: `_escape_json_string_controls` (18–54),
  `_repair_tool_arguments` (57–77).
* 12b — file/search: `_nlp_read`, `_read_page`, `_nlp_read_many`, `_nlp_read_skill`,
  `_nlp_search`, `_nlp_list_files`, `_nlp_definitions`, `_nlp_references`.
* 12c — exec: `_nlp_run`, `_nlp_git`, `_nlp_merge`, `_nlp_diff`, `_nlp_edit`,
  `_nlp_write`, `_nlp_fix`, `_nlp_tests`.
* 12d — delegation: `_nlp_delegate`, `_nlp_delegate_batch`, `_nlp_create_subagent`,
  `_nlp_run_subagent_task`.
* 12e — reasoning/misc: `_nlp_jev_decide`, `_nlp_speculate`, `_nlp_analyze`,
  `_nlp_web_search`, `_nlp_get_current_datetime`, `_nlp_hue_control`.
* 12f — MCP: `_nlp_mcp_tools`, `_nlp_mcp_call`.

⚠ `_nlp_tool_handlers()` (977–1022) is the wiring point and **stays on `Agent`**;
it must return the same mapping. `_execute_tool_call` also stays.
⚠ Reads to route through `live()`: `_CONTEXT_AST_THRESHOLD_KB` (1330, 1333),
`_DELEGATE_TIMEOUT_S` (874, 877), `_kill_process_tree` (1494),
`_pytest_full_suite_timeout` (1463, 1505, 1711),
`record_full_run_seconds` (1500, 1518, 1721).

**t13 — plan tools, prompt blocks, persistence** *(deps: t03, t12)*
Three modules, three commits:

* `agent_core/handlers/plan_tools.py` — 1892–2217: `_get_plan_state`,
  `_nlp_plan_status`, `_nlp_plan_start`, `_nlp_plan_step`, `_nlp_plan_finish`.
* `agent_core/prompts/context_blocks.py` — 3235–3538:
  `_decision_constraints_block`, `_skill_index_block`, `_skill_hint_block`,
  `_habits_block`, `_file_history_block`, `_semantic_memory_block`,
  `_ensure_original_goal`, `_recent_user_texts`, `_prompts_related`,
  `_suspend_history_for_new_task`, `_original_goal_block`.
  ⚠ `_decision_constraints_block` reads `decisions_as_system_prompt` at 3248.
* `agent_core/memory/persistence.py` — 3544–3791 plus `_read_json_quarantining`
  (3795–3820): `_load_chat_history`, `_save_chat_history`, `_load_memory`,
  `_save_memory`, `_record_llm_experience`.
  ⚠ The heaviest patch surface: `CHAT_HISTORY_JSON_PATH` (5 reads),
  `AGENT_MEMORY_JSON_PATH` (5 reads). **This step is the reason `live()` exists.**
  (Correction vs the old plan: `_read_json_quarantining` sits at 3795, not in the
  compaction block, so it belongs here.)

**t14 — the turn loop** → `agent_core/turn_loop.py` *(deps: t03–t13)*
Methods 2504–3161: `_run_chained_tool_loop` (253 lines), `_reset_failover_notice`,
`_report_failover`, `_finish_turn` (125), `chat_nlp` (145),
`_mutating_files_this_turn`, `_print_self_review_note`, `_persist_plan_answer`.
**Last and most delicate:** `chat_nlp` is the entry point of the whole NLP path and
owns continuation chaining, retry/backoff and the trace writer lifecycle.
Extract only after everything it calls has settled.
⚠ Reads to route through `live()`: `_MAX_CHAINED_RUNS` (2686),
`_looks_incomplete` (2683), `record_turn_outcome` (2872),
`recommend_profile` (2993), `_resolve_display_mode` (7 sites), `ToolLoopRunner`
(3 sites).

### Phase C — consolidation

**t15 — final consolidation** *(deps: t03–t14, role: reviewer + integrator)*
1. Delete imports that became dead in `agent.py`.
2. Run `agent_core/utils/lint_guard.py` if it is wired, plus the project linter.
3. Confirm `agent.py` is a composition root: bootstrap import, re-exports,
   `Agent.__init__` wiring, delegators, entry point.
4. Update the module map in `Architecture.md` / `AGENTS.md` and the step log in
   `docs/refactor_agent_py.md`.
5. Delete scratch files; **one** full `python -m pytest` run; commit.

---

## 6. Expected shape

| Milestone | `agent.py` lines | After |
| --- | --- | --- |
| Baseline (t01) | 5565 | — |
| After t02 (actual) | **5416** | t02 |
| Module-level helpers gone (~1560 moved out, ~90 re-export lines in) | ≈ 3950 | t10 |
| `Agent` methods extracted (~2600 out, ~200 delegator lines in) | ≈ 1550 | t14 |
| Composition root | ≈ 1300–1500 | t15 |

The last number is larger than the old plan's 600–900 because the delegators stay
on `Agent` by design. That is the deliberate trade: ~200 extra lines in exchange
for every call site, dispatch entry and monkeypatch target staying valid — which
is what makes each step easy to understand.

---

## 7. Ordering and parallelism

* Safe in parallel (independent regions, no shared state): **t03, t04, t05, t06**.
* Safe after t02: **t08, t09**.
* **Serial:** t10 → t11 → t12 → t13 → t14. All of Phase B touches `Agent`
  instance state; running two of them at once produces conflicting diffs.
* t12 and t13 are multi-commit by design; each commit still satisfies §4.

---

## 8. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| A missed re-export breaks a `from agent import X` caller | Guard test from t01; update both frozensets in the same commit |
| A monkeypatch silently stops applying after a move | §3 table + `live()` for every listed read |
| Circular import `agent_core.*` → `agent` | `import agent` only inside `live()`, never at module level |
| Import-time side effects reorder (OpenBLAS, UTF-8 stdout, `_SHELL_NAME`) | Side effects stay in `bootstrap.py`; `_SHELL_NAME` evaluated once |
| A step changes behaviour while "cleaning up" | Pure-move rule in §4 item 1; behaviour-preserving is the acceptance bar |
| Diff too large to review | t12/t13 split per cluster; one concern per commit |
| `agent.py` grows instead of shrinking | §4 closing rule: a growing step copied code |

---

## Tasks

- [t03] Extract history compaction (agent.py 4694-5100 + constants) to agent_core/history_compaction.py, add agent_core/runtime_globals.py live() accessor, re-export every name (role: implementer) deps: t01
- [t04] Extract turn telemetry (agent.py 3934-4260 + marker constants) to agent_core/turn_telemetry.py, routing load_agent_settings and TURN_LOG_PATH reads through live() (role: implementer) deps: t01
- [t05] Extract shell and subprocess helpers (agent.py 164-176, 4056-4144, 4263-4297, 4433-4520) to agent_core/shell_intel.py, keeping the _DANGEROUS_SHELL_PATTERNS alias in agent.py (role: implementer) deps: t01
- [t06] Extract pytest command policy (agent.py 4532-4691 + _PYTEST_* constants) to agent_core/pytest_cmd_policy.py, routing _ENV_FILE_PATH through live() (role: implementer) deps: t01
- [t07] Extract _SYSTEM_PROMPT (agent.py 4309-4430) and _SHELL_NAME to agent_core/prompts/system_prompt.py with re-export (role: implementer) deps: t05
- [t08] Extract class LLMClient (agent.py 202-377) to agent_core/llm/client.py, routing resolve_model and load_agent_settings through live(), with re-export (role: implementer) deps: t02
- [t09] Extract signal handling and interruptible input (agent.py 81-199) to agent_core/cli/signals.py, moving the mutable shutdown globals with their mutators (role: implementer) deps: t02
- [t10] Extract the REPL layer (agent.py 5122-5409) to agent_core/cli/registry_setup.py and agent_core/cli/interactive.py, keeping the __main__ entry behaviour identical (role: implementer) deps: t08, t09
- [t11] Extract legacy _tool_* aliases and file operations (agent.py 2221-2408) to agent_core/agent_files.py, keeping one-line delegators on Agent (role: implementer) deps: t03, t04, t05, t06, t07, t08, t09, t10
- [t12] Extract the _nlp_* tool handlers (agent.py 749-1888) to agent_core/handlers/nlp_tools.py in six per-cluster commits, keeping the _nlp_tool_handlers wiring on Agent unchanged (role: implementer) deps: t11
- [t13] Extract plan tools to agent_core/handlers/plan_tools.py, prompt context blocks to agent_core/prompts/context_blocks.py and chat/memory persistence to agent_core/memory/persistence.py (role: implementer) deps: t03, t12
- [t14] Extract the turn loop (agent.py 2504-3161) to agent_core/turn_loop.py, routing _MAX_CHAINED_RUNS and the other patched reads through live() (role: implementer) deps: t03, t04, t05, t06, t07, t08, t09, t10, t11, t12, t13
- [t15] Final consolidation: prune dead imports, run the lint guard, one full test run, update Architecture.md and AGENTS.md module map and docs/refactor_agent_py.md (role: integrator) deps: t03, t04, t05, t06, t07, t08, t09, t10, t11, t12, t13, t14
